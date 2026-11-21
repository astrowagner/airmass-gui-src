"""Observatory data and presets."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Observatory:
    name: str
    lat: float
    lon: float
    elevation_m: float
    timezone: str = "UTC"


LBT = Observatory(
    "LBT (Large Binocular Telescope)",
    lat=32.44278,
    lon=-111.15500,
    elevation_m=2400.0,
    timezone="America/Phoenix",
)

PRESETS: dict[str, Observatory] = {
    obs.name: obs
    for obs in [
        LBT,
        Observatory("Keck I/II (Mauna Kea)", 19.8263, -155.4744, 4145.0, "Pacific/Honolulu"),
        Observatory("Subaru (Mauna Kea)", 19.8257, -155.4746, 4146.0, "Pacific/Honolulu"),
        Observatory("Gemini North (Mauna Kea)", 19.8227, -155.4737, 2731.0, "Pacific/Honolulu"),
        Observatory("VLT (Paranal)", -24.6275, -70.4044, 2635.0, "America/Santiago"),
        Observatory("Gemini South (Cerro Pachon)", -30.2440, -70.7379, 2722.0, "America/Santiago"),
        Observatory("CTIO (Cerro Tololo)", -30.1691, -70.8067, 2200.0, "America/Santiago"),
        Observatory("McDonald (MPE)", 30.6709, -104.5612, 1710.0, "America/Chicago"),
    ]
}

DEFAULT_PRESET = "LBT (Large Binocular Telescope)"
CUSTOM = "Custom"


def from_dict(data: dict) -> Observatory:
    return Observatory(
        name=data.get("name", "Custom observatory"),
        lat=float(data.get("lat", 0.0)),
        lon=float(data.get("lon", 0.0)),
        elevation_m=float(data.get("elevation_m", 0.0)),
        timezone=data.get("timezone", "UTC"),
    )


def to_dict(obs: Observatory) -> dict:
    return {
        "name": obs.name,
        "lat": obs.lat,
        "lon": obs.lon,
        "elevation_m": obs.elevation_m,
        "timezone": obs.timezone,
    }
