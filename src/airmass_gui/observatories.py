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
PRESETS: dict[str, Observatory] = {
    obs.name: obs
    for obs in [
        Observatory("Apache Point (APO 3.5m)", 34.2149, -107.8376, 2789.0, "America/Denver"),
        Observatory("Brewster Hill (Hopkins 2.0)", 42.3439, -73.0553, 204.0, "America/New_York"),
        Observatory("Brewster Hill (Whipple 4m)", 42.3439, -73.0553, 204.0, "America/New_York"),
        Observatory("Calar Alto", 37.2327, -2.6009, 2189.0, "Europe/Madrid"),
        Observatory("Cerro La Silla (ESO)", -29.2563, -70.7530, 2388.0, "America/Santiago"),
        Observatory("Cerro Paranal (VLT)", -24.6275, -70.4044, 2635.0, "America/Santiago"),
        Observatory("Cerro Tololo (CTIO)", -30.1691, -70.8067, 2200.0, "America/Santiago"),
        Observatory("Chajnantor (ALMA)", -23.0233, -67.7539, 5058.0, "America/Santiago"),
        Observatory("Effelsberg (100m)", 48.9895, 6.9263, 310.0, "Europe/Berlin"),
        Observatory("Green Bank (GBT 100m)", 38.4331, -79.4664, 805.0, "America/New_York"),
        Observatory("Jodrell Bank (Lovell 76m)", 53.2444, -2.3089, 66.0, "Europe/London"),
        Observatory("Karoo (MeerKAT)", -30.7215, 21.4184, 1074.0, "Africa/Johannesburg"),
        Observatory("Kitt Peak (LPL 4m)", 31.9583, -111.6017, 2096.0, "America/Phoenix"),
        Observatory("Kitt Peak (NSO)", 31.9583, -111.6017, 2096.0, "America/Phoenix"),
        Observatory("Las Campanas (Carnegie)", -29.2944, -70.7250, 2420.0, "America/Santiago"),
        Observatory("Mauna Kea (Gemini North)", 19.8227, -155.4737, 2731.0, "Pacific/Honolulu"),
        Observatory("Mauna Kea (Keck I/II)", 19.8263, -155.4744, 4145.0, "Pacific/Honolulu"),
        Observatory("Mauna Kea (Subaru)", 19.8257, -155.4746, 4146.0, "Pacific/Honolulu"),
        Observatory(
            "Mount Graham (LBT)",
            lat=32.44278,
            lon=-111.15500,
            elevation_m=2400.0,
            timezone="America/Phoenix",
        ),
        Observatory("Mount Hamilton (Lick)", 37.3414, -121.6428, 1243.0, "America/Los_Angeles"),
        Observatory("Mount Lemmon (Skycenter)", 35.2009, -111.6075, 2167.0, "America/Phoenix"),
        Observatory("Mount McDonald (MPE)", 30.6709, -104.5612, 1710.0, "America/Chicago"),
        Observatory("Mount Wilson (100-inch)", 34.2908, -118.1006, 1742.0, "America/Los_Angeles"),
        Observatory("Pic du Midi (2.0m)", 42.9372, 0.9933, 2877.0, "Europe/Paris"),
        Observatory("Roque de los Muchachos (IAC)", 28.3127, -16.5017, 2382.0, "Europe/Madrid"),
        Observatory("San Antonio Mountain (Palomar)", 33.3564, -116.8650, 1712.0, "America/Los_Angeles"),
        Observatory("San Pedro Martir (LMT 8m)", 30.1956, -115.7953, 2730.0, "America/Mexico_City"),
        Observatory("Siding Spring (AAO)", -31.2733, 149.0644, 1165.0, "Australia/Sydney"),
        Observatory("Sutherland (SALT)", -32.376, 20.810, 1062.0, "Africa/Johannesburg"),
        Observatory("Teide (OT)", 28.0597, -16.6602, 2383.0, "Europe/Madrid"),
        Observatory("VLA (Karl G. Jansky)", 34.0785, -107.6183, 2124.0, "America/Denver"),
        Observatory("Yerkes (40-inch)", 42.5878, -89.1700, 366.0, "America/Chicago"),
    ]
}

# Preset used when no workspace exists yet.
DEFAULT_PRESET = "Mount Graham (LBT)"

# Old -> new preset display names from the rename to "mountain (facility)".
# Workspaces, saved site states, and shared links pre-dating the rename are
# migrated to the new names on load.
RENAME_MAP: dict[str, str] = {
    "LBT (Large Binocular Telescope)": "Mount Graham (LBT)",
    "Subaru (Mauna Kea)": "Mauna Kea (Subaru)",
    "Keck I/II (Mauna Kea)": "Mauna Kea (Keck I/II)",
    "Gemini North (Mauna Kea)": "Mauna Kea (Gemini North)",
    "Gemini South (Cerro Pachon)": "Cerro Pachon (Gemini South)",
    "VLT (Paranal)": "Cerro Paranal (VLT)",
    "CTIO (Cerro Tololo)": "Cerro Tololo (CTIO)",
    "IAC (Roque de los Muchachos)": "Roque de los Muchachos (IAC)",
    "Lick (Mount Hamilton)": "Mount Hamilton (Lick)",
    "McDonald (MPE)": "Mount McDonald (MPE)",
    "NOAO (Mount Lemmon)": "Mount Lemmon (Skycenter)",
    "Mount Lemmon (NOAO)": "Mount Lemmon (Skycenter)",
    "NSO (Kitt Peak)": "Kitt Peak (NSO)",
    "Palomar": "San Antonio Mountain (Palomar)",
    "SALT (SAAO)": "Sutherland (SALT)",
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
