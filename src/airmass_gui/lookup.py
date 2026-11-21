"""Target name lookup via the CDS SIMBAD service (and its official mirrors).

The main CDS SIMBAD service occasionally goes down; the documented mirrors
(https://simbad.u-strasbg.fr/) are tried in order.
"""
from __future__ import annotations

import re

import requests

SIMBAD_ENDPOINTS = (
    "https://simbad.cfa.harvard.edu/simbad/sim-id",
    "https://simbad.idia.ac.za/simbad/sim-id",
    "https://simbad.u-strasbg.fr/simbad/sim-id",
)

COORD_RE = re.compile(
    r"Coordinates\(ICRS,ep=J2000,eq=2000\):\s+"
    r"(\d+)\s+(\d+)\s+([\d.]+)\s+"
    r"([+-]?\d+)\s+(\d+)\s+([\d.]+)"
)


def simbad_lookup(name: str, timeout: float = 30.0) -> dict:
    """Look up a target name in SIMBAD.

    Returns dict with keys: name, ra, dec (decimal degrees), matches, info.
    Raises LookupError on failure.
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
    """Parse a SIMBAD ASCII object record; None if no ICRS coordinates are present."""
    lines = text.splitlines()

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
            "info": _record_info(lines),
        }
    return None


def _line_value(lines: list[str], pattern: str) -> str | None:
    regex = re.compile(pattern)
    for line in lines:
        match = regex.match(line)
        if match and match.group(1) not in ("~", ""):
            return match.group(1)
    return None


def _record_info(lines: list[str]) -> str:
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
