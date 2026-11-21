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
        None if absent), plx (parallax in mas, None if absent), and
        info (a short ", "-joined summary for display).

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
        if absent), and info (display summary). None when the record has no
        ICRS coordinate line.
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
        return {
            "name": name or "Unknown",
            "ra": ra_hours * 15.0,
            "dec": dec_degrees,
            "mag": _line_value(lines, r"^Flux V\s*:\s*(-?[\d.]+)"),
            "spec": _line_value(lines, r"^Spectral type:\s*(\S+)"),
            "plx": _line_value(lines, r"^Parallax:\s*([\d.]+)"),
            "info": _record_info(lines),
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


def _record_info(lines: list[str]) -> str:
    """One-line summary of photometry/astrometry for display.

    Parameters
    ----------
    lines : list of str
        Record lines.

    Returns
    -------
    str
        ", "-joined "label=value" pairs for V/B flux, spectral type,
        parallax, and radial velocity; empty string when the record has
        none of them.
    """
    parts = []
    for pattern, label in (
        (r"^Flux V\s*:\s*(-?[\d.]+)", "V mag"),
        (r"^Flux B\s*:\s*(-?[\d.]+)", "B mag"),
        (r"^Spectral type:\s*(\S+)", "spectral type"),
        (r"^Parallax:\s*([\d.]+)", "parallax [mas]"),
        (r"^Radial Velocity:\s*(-?[\d.]+)", "RV [km/s]"),
    ):
        value = _line_value(lines, pattern)
        if value is not None:
            parts.append(f"{label}={value}")
    return ", ".join(parts)
