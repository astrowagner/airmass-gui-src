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
        Observatory("Calar Alto", 37.2327, -2.6009, 2189.0, "Europe/Madrid"),
        Observatory("CTIO (Cerro Tololo)", -30.1691, -70.8067, 2200.0, "America/Santiago"),
        Observatory("Gemini North (Mauna Kea)", 19.8227, -155.4737, 2731.0, "Pacific/Honolulu"),
        Observatory("Gemini South (Cerro Pachon)", -30.2440, -70.7379, 2722.0, "America/Santiago"),
        Observatory("IAC (Roque de los Muchachos)", 28.3127, -16.5017, 2382.0, "Europe/Madrid"),
        Observatory("Keck I/II (Mauna Kea)", 19.8263, -155.4744, 4145.0, "Pacific/Honolulu"),
        Observatory("Las Campanas (Carnegie)", -29.2944, -70.7250, 2420.0, "America/Santiago"),
        Observatory(
            "LBT (Large Binocular Telescope)",
            lat=32.44278,
            lon=-111.15500,
            elevation_m=2400.0,
            timezone="America/Phoenix",
        ),
        Observatory("Lick (Mount Hamilton)", 37.3414, -121.6428, 1243.0, "America/Los_Angeles"),
        Observatory("McDonald (MPE)", 30.6709, -104.5612, 1710.0, "America/Chicago"),
        Observatory("NOAO (Mount Lemmon)", 35.2009, -111.6075, 2167.0, "America/Phoenix"),
        Observatory("NSO (Kitt Peak)", 31.9583, -111.6017, 2096.0, "America/Phoenix"),
        Observatory("Palomar", 33.3564, -116.8650, 1712.0, "America/Los_Angeles"),
        Observatory("SALT (SAAO)", -32.376, 20.810, 1062.0, "Africa/Johannesburg"),
        Observatory("Siding Spring (AAO)", -31.2733, 149.0644, 1165.0, "Australia/Sydney"),
        Observatory("Subaru (Mauna Kea)", 19.8257, -155.4746, 4146.0, "Pacific/Honolulu"),
        Observatory("VLT (Paranal)", -24.6275, -70.4044, 2635.0, "America/Santiago"),
    ]
}

# Preset used when no workspace exists yet.
DEFAULT_PRESET = "LBT (Large Binocular Telescope)"

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
