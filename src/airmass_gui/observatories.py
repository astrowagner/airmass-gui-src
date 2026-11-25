"""Observatory definitions: a geodetic site, IANA time zone, and named presets."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Observatory:
    """One observatory site: geodetic coordinates, altitude, and local time zone.

    Attributes
    ----------
    name : str
        Display name.
    lat : float
        Geodetic latitude in degrees (positive north).
    lon : float
        Geodetic longitude in degrees (positive east).
    elevation_m : float
        Altitude above mean sea level in meters.
    timezone : str, default "UTC"
        IANA time-zone name used for all local-time display (e.g.
        "America/Phoenix").
    """

    name: str
    lat: float
    lon: float
    elevation_m: float
    timezone: str = "UTC"


# Named sites offered in the preset menu; keys are the display names.
# A copy of tests/fixtures/observatories_reference.csv (verified by
# sha256 in tests/test_observatories.py), whose rows cite the astropy
# site registry (github.com/astropy/astropy-data,
# coordinates/sites.json) and the Wikipedia pages named in its source
# column. The two Mount Hopkins rows use America/Phoenix because
# Arizona keeps MST all year.
PRESETS: dict[str, Observatory] = {
    obs.name: obs
    for obs in [
        Observatory("Apache Point (APO 3.5m)", 32.78000, -105.82000, 2798.0, "America/Denver"),
        Observatory("Calar Alto", 37.22361, -2.54611, 2168.0, "Europe/Madrid"),
        Observatory("Cerro La Silla (ESO)", -29.25667, -70.73000, 2347.0, "America/Santiago"),
        Observatory("Cerro Pachon (Gemini South)", -30.24074, -70.73668, 2750.0, "America/Santiago"),
        Observatory("Cerro Paranal (VLT)", -24.62744, -70.40499, 2669.0, "America/Santiago"),
        Observatory("Cerro Tololo (NOIRLab)", -30.16528, -70.81500, 2215.0, "America/Santiago"),
        Observatory("Chajnantor (ALMA)", -23.02900, -67.75500, 5000.0, "America/Santiago"),
        Observatory("Effelsberg (100m)", 50.52484, 6.88362, 417.0, "Europe/Berlin"),
        Observatory("Green Bank (GBT 100m)", 38.43306, -79.83972, 807.0, "America/New_York"),
        Observatory("Jodrell Bank (Lovell 76m)", 53.23650, -2.30840, 77.0, "Europe/London"),
        Observatory("Karoo (MeerKAT)", -30.71106, 21.44389, 1087.0, "Africa/Johannesburg"),
        Observatory("Kitt Peak (Mayall 4m)", 31.96333, -111.60000, 2120.0, "America/Phoenix"),
        Observatory("Kitt Peak (NSO)", 31.96333, -111.60000, 2120.0, "America/Phoenix"),
        Observatory("Las Campanas (Carnegie)", -29.00333, -70.70167, 2282.0, "America/Santiago"),
        Observatory("Mauna Kea (Gemini North)", 19.82380, -155.46905, 4213.0, "Pacific/Honolulu"),
        Observatory("Mauna Kea (Keck I/II)", 19.82625, -155.47448, 4160.0, "Pacific/Honolulu"),
        Observatory("Mauna Kea (Subaru)", 19.82550, -155.47602, 4163.0, "Pacific/Honolulu"),
        Observatory("Mount Graham (LBT)", 32.70131, -109.88906, 3221.0, "America/Phoenix"),
        Observatory("Mount Hamilton (Lick)", 37.34333, -121.63667, 1290.0, "America/Los_Angeles"),
        Observatory("Mount Hopkins (MMT 6.5m)", 31.68833, -110.88500, 2608.0, "America/Phoenix"),
        Observatory("Mount Hopkins (Whipple 1.5m)", 31.68094, -110.87750, 2320.0, "America/Phoenix"),
        Observatory("Mount Lemmon (Skycenter)", 32.44200, -110.78930, 2791.0, "America/Phoenix"),
        Observatory("Mount Locke (McDonald)", 30.67167, -104.02167, 2075.0, "America/Chicago"),
        Observatory("Mount Wilson (100-inch)", 34.22258, -118.06168, 1742.0, "America/Los_Angeles"),
        Observatory("Palomar Mountain (Hale 200-inch)", 33.35600, -116.86300, 1706.0, "America/Los_Angeles"),
        Observatory("Pic du Midi (2.0m)", 42.93639, 0.14278, 2877.0, "Europe/Paris"),
        Observatory("Roque de los Muchachos (IAC)", 28.75833, -17.88000, 2327.0, "Atlantic/Canary"),
        Observatory("San Pedro Martir (OAN 2.1m)", 31.02917, -115.48694, 2830.0, "America/Tijuana"),
        Observatory("Siding Spring (AAO)", -31.27704, 149.06609, 1164.0, "Australia/Sydney"),
        Observatory("Sutherland (SALT)", -32.37582, 20.81081, 1798.0, "Africa/Johannesburg"),
        Observatory("Teide (OT)", 28.30000, -16.50972, 2390.0, "Atlantic/Canary"),
        Observatory("VLA (Karl G. Jansky)", 34.07875, -107.61828, 2124.0, "America/Denver"),
        Observatory("Yerkes (40-inch)", 42.57030, -88.55670, 334.0, "America/Chicago"),
    ]
}

# Preset used when no workspace exists yet.
DEFAULT_PRESET = "Mount Graham (LBT)"

# Old -> new preset display names from the rename to "mountain (facility)".
# Workspaces, saved site states, and shared links pre-dating the rename are
# migrated to the new names on load.
RENAME_MAP: dict[str, str] = {
    "Brewster Hill (Hopkins 2.0)": "Mount Hopkins (MMT 6.5m)",
    "Brewster Hill (Whipple 4m)": "Mount Hopkins (Whipple 1.5m)",
    "CTIO (Cerro Tololo)": "Cerro Tololo (NOIRLab)",
    "Cerro Tololo (CTIO)": "Cerro Tololo (NOIRLab)",
    "Gemini North (Mauna Kea)": "Mauna Kea (Gemini North)",
    "Gemini South (Cerro Pachon)": "Cerro Pachon (Gemini South)",
    "IAC (Roque de los Muchachos)": "Roque de los Muchachos (IAC)",
    "Keck I/II (Mauna Kea)": "Mauna Kea (Keck I/II)",
    "Kitt Peak (LPL 4m)": "Kitt Peak (Mayall 4m)",
    "LBT (Large Binocular Telescope)": "Mount Graham (LBT)",
    "Lick (Mount Hamilton)": "Mount Hamilton (Lick)",
    "McDonald (MPE)": "Mount Locke (McDonald)",
    "Mount Lemmon (NOAO)": "Mount Lemmon (Skycenter)",
    "Mount McDonald (MPE)": "Mount Locke (McDonald)",
    "NOAO (Mount Lemmon)": "Mount Lemmon (Skycenter)",
    "NSO (Kitt Peak)": "Kitt Peak (NSO)",
    "Palomar": "Palomar Mountain (Hale 200-inch)",
    "SALT (SAAO)": "Sutherland (SALT)",
    "San Antonio Mountain (Palomar)": "Palomar Mountain (Hale 200-inch)",
    "San Pedro Martir (LMT 8m)": "San Pedro Martir (OAN 2.1m)",
    "Subaru (Mauna Kea)": "Mauna Kea (Subaru)",
    "VLT (Paranal)": "Cerro Paranal (VLT)",
}

# Sentinel for an observatory edited away from any preset.
CUSTOM = "Custom"


def from_dict(data: dict) -> Observatory:
    """Build an Observatory from a workspace dict, filling missing keys with defaults.

    Parameters
    ----------
    data : dict
        Observatory fields (name, lat, lon, elevation_m, timezone); any key
        may be absent.

    Returns
    -------
    Observatory
    """
    return Observatory(
        name=data.get("name", "Custom observatory"),
        lat=float(data.get("lat", 0.0)),
        lon=float(data.get("lon", 0.0)),
        elevation_m=float(data.get("elevation_m", 0.0)),
        timezone=data.get("timezone", "UTC"),
    )


def to_dict(obs: Observatory) -> dict:
    """Serialize an Observatory to a JSON-ready dict (no preset key).

    Parameters
    ----------
    obs : Observatory

    Returns
    -------
    dict
        name, lat, lon, elevation_m, timezone.
    """
    return {
        "name": obs.name,
        "lat": obs.lat,
        "lon": obs.lon,
        "elevation_m": obs.elevation_m,
        "timezone": obs.timezone,
    }
