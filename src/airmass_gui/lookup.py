"""Target name lookup via the CDS SIMBAD service (and its official mirrors).

The main CDS SIMBAD service occasionally goes down; the documented mirrors
(https://simbad.u-strasbg.fr/) are tried in order.
"""
from __future__ import annotations

import re

import requests

# Mirrors tried in order (CDS SIMBAD + the two documented mirror sites).
SIMBAD_ENDPOINTS = (
    "https://simbad.cfa.harvard.edu/simbad/sim-id",
    "https://simbad.idia.ac.za/simbad/sim-id",
    "https://simbad.u-strasbg.fr/simbad/sim-id",
)

# The ICRS coordinate line of a SIMBAD ASCII record, e.g.
# "Coordinates(ICRS,ep=J2000,eq=2000): 18 36 56.33635  +38 47 01.2802 (...)".
COORD_RE = re.compile(
    r"Coordinates\(ICRS,ep=J2000,eq=2000\):\s+"
    r"(\d+)\s+(\d+)\s+([\d.]+)\s+"
    r"([+-]?\d+)\s+(\d+)\s+([\d.]+)"
)

# "Proper motions: 200.94 286.23 [0.32 0.40 0] A 2007A&A...474..653V";
# components in mas/yr, the third bracket value is the ignore flag.
PM_RE = re.compile(
    r"^Proper motions:\s*([+-]?[\d.]+)\s+([+-]?[\d.]+)\s*"
    r"\[([+-]?[\d.]+)\s+([+-]?[\d.]+)"
)
# "Parallax: 130.23 [0.36] A ...", mas; bracket content '~' means no error.
PLX_RE = re.compile(r"^Parallax:\s*([\d.]+)\s*\[([~\d.]+)")
# "Radial Velocity: -13.5 [0.4] C ...", km/s.
RV_RE = re.compile(r"^Radial Velocity:\s*(-?[\d.]+)")
# "Flux U : 0.03 [~] C ..." one line per band; '~' means no error.
FLUX_RE = re.compile(r"^Flux\s+([A-Za-z0-9]+)\s*:\s*(-?[\d.]+)\s*\[([~\d.]+)")
# "Identifiers (65):" header of the fixed-column identifier block.
IDENTS_RE = re.compile(r"^Identifiers \((\d+)\):")

def _first_match(pattern: re.Pattern, lines: list[str]) -> re.Match | None:
    """First record line matching pattern; None when none does.

    Parameters
    ----------
    pattern : re.Pattern
        Regular expression anchored at the line start.
    lines : list of str
        Record lines to scan.

    Returns
    -------
    re.Match or None
    """
    for line in lines:
        match = pattern.match(line)
        if match:
            return match
    return None


def simbad_lookup(name: str, timeout: float = 30.0) -> dict:
    """Look up a target name in SIMBAD, trying each mirror in turn.

    Parameters
    ----------
    name : str
        Object name or identifier as sent in the Ident parameter.
    timeout : float, default 30.0
        Per-request timeout in seconds.

    Returns
    -------
    dict
        name, ra (ICRS, J2000, degrees), dec (degrees), matches (int),
        mag (V magnitude, None if absent), spec (spectral type string,
        None if absent), plx (parallax in mas, None if absent),
        plxErr (parallax uncertainty in mas, None when absent),
        pmRa/pmDec (proper motion in mas/yr, None if absent),
        pmRaErr/pmDecErr (their uncertainties in mas/yr), rv (radial
        velocity in km/s, None if absent), fluxes (band -> {"value",
        "err"} magnitudes per "Flux X" line), idents (alternative
        identifiers) and idTotal (their count), and info (a short
        ", "-joined summary for display).

    Raises
    ------
    LookupError
        If every mirror fails or returns no coordinates.
    """
    last_error: Exception | None = None
    for endpoint in SIMBAD_ENDPOINTS:
        try:
            response = requests.get(
                endpoint,
                params={"Ident": name, "output.service": "bibformat", "output.format": "ASCII"},
                timeout=timeout,
            )
            response.raise_for_status()
            result = _parse_ascii_record(response.text)
            if result is None:
                last_error = LookupError(f"no coordinates in Simbad response for {name!r}")
                continue
            result["matches"] = 1
            return result
        except requests.RequestException as exc:
            last_error = exc
            continue
    raise LookupError(f"Simbad lookup failed for {name!r}: {last_error}")


def _parse_ascii_record(text: str) -> dict | None:
    """Parse a SIMBAD ASCII object record (output.format=ASCII).

    Parameters
    ----------
    text : str
        Full text of the ASCII record.

    Returns
    -------
    dict or None
        name, ra (J2000 degrees), dec (degrees), mag (V magnitude, None if
        the record has no "Flux V" line), spec (first token of the
        "Spectral type:" line, None if absent), plx (parallax in mas, None
        if absent), plxErr (parallax uncertainty in mas, None when the
        record carries no error), pmRa/pmDec (proper motion components in
        mas/yr, None if absent), pmRaErr/pmDecErr (their uncertainties in
        mas/yr), rv (radial velocity in km/s, None if absent), fluxes
        (band -> {"value", "err"} magnitudes per "Flux X" line, err None
        when the record marks it '~'), idents (identifiers from the
        "Identifiers (n):" block) and idTotal (n, 0 when absent), and info
        (display summary). None when the record has no ICRS coordinate
        line.
    """
    lines = text.splitlines()

    # Object name: first non-blank line of the header that is not a C.D.S banner.
    name = None
    for line in lines[:4]:
        stripped = line.strip()
        if stripped and not stripped.startswith("C.D.S"):
            name = stripped
            break

    for line in lines:
        match = COORD_RE.search(line)
        if not match:
            continue
        ra_hours = float(match.group(1)) + float(match.group(2)) / 60.0 + float(match.group(3)) / 3600.0
        dec_degrees = (
            abs(float(match.group(4))) + float(match.group(5)) / 60.0 + float(match.group(6)) / 3600.0
        )
        if match.group(4).startswith("-"):
            dec_degrees = -dec_degrees
        pm = _proper_motion(lines)
        plx, plx_err = _parallax(lines)
        idents, id_total = _identifiers(lines)
        return {
            "name": name or "Unknown",
            "ra": ra_hours * 15.0,
            "dec": dec_degrees,
            "mag": _line_value(lines, r"^Flux V\s*:\s*(-?[\d.]+)"),
            "spec": _line_value(lines, r"^Spectral type:\s*(\S+)"),
            "plx": plx,
            "plxErr": plx_err,
            "pmRa": pm[0], "pmDec": pm[1],
            "pmRaErr": pm[2], "pmDecErr": pm[3],
            "rv": _line_value(lines, r"^Radial Velocity:\s*(-?[\d.]+)"),
            "fluxes": _fluxes(lines),
            "idents": idents,
            "idTotal": id_total,
            "info": _record_info(lines, plx, plx_err, pm, id_total),
        }
    return None


def _line_value(lines: list[str], pattern: str) -> str | None:
    """First value on the first line matching pattern; None if none.

    Parameters
    ----------
    lines : list of str
        Record lines to scan.
    pattern : str
        Regular expression whose group(1) is the value. A line whose value
        is "~" (SIMBAD's "no data" marker) is skipped.

    Returns
    -------
    str or None
    """
    regex = re.compile(pattern)
    for line in lines:
        match = regex.match(line)
        if match and match.group(1) not in ("~", ""):
            return match.group(1)
    return None


def _parallax(lines: list[str]) -> tuple[float | None, float | None]:
    """Parallax and its uncertainty from the "Parallax:" line.

    Parameters
    ----------
    lines : list of str
        Record lines.

    Returns
    -------
    tuple of float or None
        (parallax in mas, uncertainty in mas); both None when the line is
        absent. The uncertainty is None when the record marks it '~'.
    """
    match = _first_match(PLX_RE, lines)
    if match is None:
        return None, None
    err = None if match.group(2) == "~" else float(match.group(2))
    return float(match.group(1)), err


def _proper_motion(lines: list[str]) -> tuple:
    """Proper motion components and uncertainties from the record.

    Parameters
    ----------
    lines : list of str
        Record lines.

    Returns
    -------
    tuple
        (pmRa, pmDec, pmRaErr, pmDecErr) in mas/yr from the "Proper
        motions:" line, all None when it is absent.
    """
    match = _first_match(PM_RE, lines)
    if match is None:
        return (None, None, None, None)
    return tuple(float(match.group(i)) for i in range(1, 5))


def _fluxes(lines: list[str]) -> dict[str, dict]:
    """Every "Flux X" line as band -> {"value", "err"}.

    Parameters
    ----------
    lines : list of str
        Record lines.

    Returns
    -------
    dict
        Band letter -> {"value": magnitude, "err": uncertainty or None
        when the record marks it '~'}. First line per band wins; {} when
        the record has no flux lines.
    """
    out: dict[str, dict] = {}
    for line in lines:
        match = FLUX_RE.match(line)
        if match is None or match.group(1) in out:
            continue
        err = None if match.group(3) == "~" else float(match.group(3))
        out[match.group(1)] = {"value": float(match.group(2)), "err": err}
    return out


def _identifiers(lines: list[str]) -> tuple[list[str], int]:
    """The fixed-column "Identifiers (n):" block as a flat list.

    The block is three 36-character columns per indented line, ending at
    the first blank or unindented line.

    Parameters
    ----------
    lines : list of str
        Record lines.

    Returns
    -------
    tuple
        (identifiers in block order, n from the header); ([], 0) when the
        record has no identifier block.
    """
    for i, line in enumerate(lines):
        header = IDENTS_RE.match(line)
        if header is None:
            continue
        total = int(header.group(1))
        ids: list[str] = []
        for nxt in lines[i + 1:]:
            if not nxt.strip() or not nxt.startswith(" "):
                break
            body = nxt.strip()
            for col in range(3):
                piece = body[col * 36 : (col + 1) * 36].strip()
                if piece:
                    ids.append(piece)
        return ids, total
    return [], 0


def _record_info(
    lines: list[str],
    plx: float | None,
    plx_err: float | None,
    pm: tuple,
    id_total: int,
) -> str:
    """One-line summary of photometry/astrometry for display.

    Parameters
    ----------
    lines : list of str
        Record lines.
    plx : float or None
        Parallax in mas, parsed by _parallax.
    plx_err : float or None
        Parallax uncertainty in mas, or None.
    pm : tuple
        Proper motion (pmRa, pmDec, pmRaErr, pmDecErr) in mas/yr, or all
        None.
    id_total : int
        Identifier count from the "Identifiers (n):" header, 0 when
        absent.

    Returns
    -------
    str
        ", "-joined "label=value" pairs for V/B flux, spectral type,
        parallax (with uncertainty and inverted distance in pc or kpc),
        proper motion, radial velocity, and identifier count; empty
        string when the record has none of them.
    """
    parts = []
    for pattern, label in (
        (r"^Flux V\s*:\s*(-?[\d.]+)", "V mag"),
        (r"^Flux B\s*:\s*(-?[\d.]+)", "B mag"),
        (r"^Spectral type:\s*(\S+)", "spectral type"),
    ):
        value = _line_value(lines, pattern)
        if value is not None:
            parts.append(f"{label}={value}")
    if plx is not None:
        entry = f"parallax={plx:g} mas"
        if plx_err is not None:
            entry += f" (+/-{plx_err:g})"
        dist = _distance_text(plx, plx_err)
        if dist:
            entry += f", distance={dist}"
        parts.append(entry)
    if pm[0] is not None:
        parts.append(
            f"proper motion=({pm[0]:g}, {pm[1]:g}) mas/yr"
        )
    rv = _line_value(lines, r"^Radial Velocity:\s*(-?[\d.]+)")
    if rv is not None:
        parts.append(f"RV={rv} km/s")
    if id_total > 0:
        parts.append(f"identifiers={id_total}")
    return ", ".join(parts)


def _distance_text(plx: float, plx_err: float | None) -> str | None:
    """Distance from a parallax, pc below 1000 pc and kpc above.

    Parameters
    ----------
    plx : float
        Parallax in mas, > 0.
    plx_err : float or None
        Parallax uncertainty in mas, or None.

    Returns
    -------
    str or None
        e.g. "7.68 pc (+/- 0.02 pc)" or "1.24 kpc"; None when the
        parallax is not positive. The uncertainty propagates as
        d * sigma / plx.
    """
    if plx is None or plx <= 0:
        return None
    dist = 1000.0 / plx
    err = dist * plx_err / plx if plx_err else None
    if dist < 1000.0:
        text = f"{dist:.2f} pc"
        if err is not None:
            text += f" (+/- {err:.2f} pc)"
        return text
    text = f"{dist / 1000.0:.3f} kpc"
    if err is not None:
        text += f" (+/- {err / 1000.0:.3f} kpc)"
    return text
