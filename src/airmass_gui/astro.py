"""Airmass and solar-altitude calculations for airmass charts.

Airmass everywhere is the Kasten & Young (1989) value normalized so that the
zenith airmass X is exactly 1. All times are astropy Time objects in UTC;
coordinates are ICRS.
"""
from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun
from astropy.time import Time

from .observatories import Observatory

#: Minutes between samples for airmass series.
STEP_MINUTES = 5


def location(obs: Observatory) -> EarthLocation:
    """Build an astropy EarthLocation for an observatory.

    Parameters
    ----------
    obs : Observatory
        Site to place; see the class doc for the lat, lon and elevation_m
        fields.

    Returns
    -------
    EarthLocation
        Geodetic location with latitude in degrees, longitude in degrees,
        and height in meters.
    """
    return EarthLocation(lat=obs.lat * u.deg, lon=obs.lon * u.deg, height=obs.elevation_m * u.m)


# Kasten & Young (1989) gives X ~= 0.9997 at the zenith; normalize so X(90 deg) = 1 exactly.
_ZENITH_X = 1.0 / (1.0 + 0.50572 * (90.0 + 6.07995) ** -1.6364)


def kasty_airmass(alt_deg):
    """Kasten & Young (1989) airmass, normalized to 1 at the zenith.

    X = 1 / (sin h + 0.50572 (h + 6.07995)^-1.6364), divided by its value at
    h = 90 deg. Finite at the horizon (X ~= 37.93 at h = 0 deg) and
    extrapolated below it.

    Parameters
    ----------
    alt_deg : float or array_like, shape (n,)
        Altitude h in degrees. Negative values (below the horizon) are
        evaluated by direct extrapolation.

    Returns
    -------
    ndarray
        Airmass X, same shape as the input (0-d for scalar input).
    """
    alt = np.asarray(alt_deg, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / (np.sin(np.deg2rad(alt)) + 0.50572 * np.power(alt + 6.07995, -1.6364))
    return raw / _ZENITH_X


def airmass_from_altitude(alt_deg):
    """Airmass X for altitude in degrees; inf below the horizon.

    Parameters
    ----------
    alt_deg : float or array_like, shape (n,)
        Altitude in degrees.

    Returns
    -------
    ndarray, shape (n,)
        Airmass X above the horizon, np.inf at or below 0 deg. Always 1-d
        (scalar input gives length 1).
    """
    alt = np.atleast_1d(np.asarray(alt_deg, dtype=float))
    return np.where(alt > 0.0, kasty_airmass(alt), np.inf)


def horizon_airmass() -> float:
    """Airmass at the horizon (altitude 0 deg), ~= 37.93.

    Returns
    -------
    float
    """
    return float(kasty_airmass(0.0))


def altitude_from_airmass(x, tol_deg: float = 1e-9, max_iter: int = 100):
    """Invert the airmass formula: X -> altitude in degrees.

    Bisection on kasty_airmass over [0, 90] deg; airmass decreases with
    altitude, so the bracket moves up when the mid airmass exceeds the
    target.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Airmass to invert. Defined for 1 <= X <= horizon_airmass();
        out-of-range input is clipped into that range.
    tol_deg : float, default 1e-9
        Bisection stopping width in degrees.
    max_iter : int, default 100
        Maximum number of bisection iterations.

    Returns
    -------
    ndarray, shape (n,)
        Altitude in degrees. Always 1-d (scalar input gives length 1).
    """
    target = np.clip(np.atleast_1d(np.asarray(x, dtype=float)), 1.0, horizon_airmass())
    out = np.where(target <= 1.0, 90.0, np.full_like(target, np.nan))
    need = target > 1.0
    if not need.any():
        return out
    # airmass decreases with altitude: X(mid) > target means we must go higher.
    # (K&Y dips ~3e-8 below 1 just under the zenith; handled by the clamp above.)
    t = target[need]
    lo = np.zeros_like(t)
    hi = np.full_like(t, 90.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if np.all(hi - lo <= tol_deg):
            break
        up = kasty_airmass(mid) > t
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    out[need] = 0.5 * (lo + hi)
    return out


def time_grid(start: Time, end: Time, minutes: int = STEP_MINUTES) -> Time:
    """Uniform time grid covering [start, end].

    Parameters
    ----------
    start : Time
        First sample (UTC).
    end : Time
        Last sample (UTC), always included.
    minutes : int, default STEP_MINUTES
        Sample spacing in minutes.

    Returns
    -------
    Time, shape (n,)
        n = ceil((end - start) / minutes) + 1 samples.
    """
    n = int(np.ceil((end - start).to_value(u.min) / minutes)) + 1
    return start + np.arange(n) * (minutes * u.min)


def airmass_series(obs: Observatory, coord: SkyCoord, start: Time, end: Time, minutes: int = STEP_MINUTES):
    """Sample altitude and airmass of a coordinate across a time span.

    Parameters
    ----------
    obs : Observatory
        Site (see location).
    coord : SkyCoord
        Object position in ICRS.
    start : Time
        First sample (UTC).
    end : Time
        Last sample (UTC).
    minutes : int, default STEP_MINUTES
        Sample spacing in minutes.

    Returns
    -------
    times : Time, shape (n,)
        Sample times.
    alt : ndarray, shape (n,)
        Altitude in degrees at each time.
    airmass : ndarray, shape (n,)
        Airmass X at each time, inf below the horizon.
    """
    times = time_grid(start, end, minutes)
    frame = AltAz(obstime=times, location=location(obs))
    alt = coord.transform_to(frame).alt.to_value(u.deg)
    return times, alt, airmass_from_altitude(alt)


def _classify(alt: float) -> str:
    """Classify sky darkness from solar altitude.

    Parameters
    ----------
    alt : float
        Solar altitude in degrees.

    Returns
    -------
    str
        One of "day", "civil", "nautical", "astro", "night", per the
        standard twilight thresholds (0, -6, -12, -18 deg).
    """
    if alt > 0.0:
        return "day"
    if alt > -6.0:
        return "civil"
    if alt > -12.0:
        return "nautical"
    if alt > -18.0:
        return "astro"
    return "night"


def sun_shading_intervals(obs: Observatory, start: Time, end: Time, minutes: int = 10):
    """Split a time span into solar-shading intervals.

    Finds every crossing of the sun's altitude through 0, -6, -12 and
    -18 deg (linear interpolation on a fine sample grid, then interval
    labels from the midpoint altitude).

    Parameters
    ----------
    obs : Observatory
        Site (see location).
    start : Time
        Span start (UTC).
    end : Time
        Span end (UTC).
    minutes : int, default 10
        Sampling spacing in minutes used to locate the crossings.

    Returns
    -------
    list of (mjd_start, mjd_end, kind)
        Contiguous intervals in modified Julian date (day) covering
        [start, end], with kind per _classify (day, civil, nautical,
        astro, night).
    """
    times = time_grid(start, end, minutes)
    frame = AltAz(obstime=times, location=location(obs))
    alt = get_sun(times).transform_to(frame).alt.to_value(u.deg)
    mjd = times.to_value("mjd")
    events: set[float] = set()
    for threshold in (0.0, -6.0, -12.0, -18.0):
        arr = alt - threshold
        idx = np.where(np.sign(arr[:-1]) != np.sign(arr[1:]))[0]
        for i in idx:
            a0, a1 = float(arr[i]), float(arr[i + 1])
            events.add(float(mjd[i] + a0 / (a0 - a1) * (mjd[i + 1] - mjd[i])))
    bounds = [float(start.to_value("mjd"))] + sorted(events) + [float(end.to_value("mjd"))]
    intervals = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 1e-6:
            continue
        alt_mid = float(np.interp((a + b) / 2.0, mjd, alt))
        intervals.append((a, b, _classify(alt_mid)))
    return intervals


def _parts(text: str) -> list[float]:
    """Split a sexagesimal entry into its numeric parts.

    Parameters
    ----------
    text : str
        Entry using colons, h/m/s or degree/arcminute/arcsecond marks, or
        plain space separators.

    Returns
    -------
    list of float
        One float per numeric part, in order.

    Raises
    ------
    ValueError
        If a part is not numeric.
    """
    s = text.strip().lower()
    for token in ("h", "m", "s", "deg", "d", "\u00b0", "\u2032", "\u2033"):
        s = s.replace(token, " ")
    return [float(part) for part in s.replace(":", " ").split()]


def parse_ra(text: str) -> float:
    """Parse a right ascension entry to decimal hours.

    Accepts h:m:s, h m s, decimal hours, or decimal degrees (values > 24
    are divided by 15).

    Parameters
    ----------
    text : str
        RA entry.

    Returns
    -------
    float
        RA in hours, wrapped to [0, 24).

    Raises
    ------
    ValueError
        If the entry is empty or malformed (wrong part count, minutes or
        seconds >= 60).
    """
    s = text.strip()
    if not s:
        raise ValueError("RA is empty")
    if ":" in s or any(ch in s.lower() for ch in "hms"):
        vals = _parts(s)
        if not 1 <= len(vals) <= 3:
            raise ValueError(f"unparseable RA: {text!r}")
        while len(vals) < 3:
            vals.append(0.0)
        hours, minutes, seconds = vals
        if abs(minutes) >= 60 or abs(seconds) >= 60:
            raise ValueError(f"unparseable RA: {text!r}")
        return (hours + minutes / 60.0 + seconds / 3600.0) % 24.0
    value = float(s)
    if value > 24.0:
        value /= 15.0  # decimal degrees -> decimal hours
    return value % 24.0


def parse_dec(text: str) -> float:
    """Parse a declination entry to decimal degrees.

    Accepts d:m:s, d m s, or decimal degrees; a negative leading degree
    part signs the whole entry.

    Parameters
    ----------
    text : str
        Dec entry.

    Returns
    -------
    float
        Declination in degrees, in [-90, 90].

    Raises
    ------
    ValueError
        If the entry is empty or malformed (wrong part count, minutes or
        seconds >= 60, |dec| > 90).
    """
    s = text.strip()
    if not s:
        raise ValueError("Dec is empty")
    if ":" in s or any(ch in s for ch in "\u00b0\u2032\u2033") or any(ch in s.lower() for ch in "dms"):
        vals = _parts(s)
        if not 1 <= len(vals) <= 3:
            raise ValueError(f"unparseable Dec: {text!r}")
        while len(vals) < 3:
            vals.append(0.0)
        deg, minutes, seconds = vals
        if abs(minutes) >= 60 or abs(seconds) >= 60:
            raise ValueError(f"unparseable Dec: {text!r}")
        sign = -1.0 if deg < 0 else 1.0
        return sign * (abs(deg) + minutes / 60.0 + seconds / 3600.0)
    value = float(s)
    if abs(value) > 90.0:
        raise ValueError(f"Dec out of range: {text!r}")
    return value


def sky_coord_from_entries(ra_text: str, dec_text: str) -> SkyCoord:
    """Build an ICRS SkyCoord from RA/Dec text entries.

    Parameters
    ----------
    ra_text : str
        RA entry, per parse_ra.
    dec_text : str
        Dec entry, per parse_dec.

    Returns
    -------
    SkyCoord
        Position in the icrs frame.

    Raises
    ------
    ValueError
        Propagated from parse_ra or parse_dec.
    """
    return SkyCoord(ra=parse_ra(ra_text) * u.hourangle, dec=parse_dec(dec_text) * u.deg, frame="icrs")


def format_ra_hours(degrees: float) -> str:
    """Format RA as 'HH:MM:SS.ss' from decimal degrees.

    Parameters
    ----------
    degrees : float
        RA in decimal degrees (J2000); divided by 15 to hours, wrapped to
        [0, 24).

    Returns
    -------
    str
        e.g. '18:36:56.34'.
    """
    hours = (degrees / 15.0) % 24.0
    h = int(hours)
    m = int((hours - h) * 60)
    s = (hours - h - m / 60.0) * 3600.0
    if s >= 59.95:
        s -= 60.0
        m += 1
    if m >= 60:
        m -= 60
        h = (h + 1) % 24
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def format_dec_dms(degrees: float) -> str:
    """Format Dec as '+DD:MM:SS.ss' from decimal degrees.

    Parameters
    ----------
    degrees : float
        Declination in decimal degrees.

    Returns
    -------
    str
        e.g. '+38:47:01.28'.
    """
    sign = "-" if degrees < 0 else "+"
    d = int(abs(degrees))
    m = int((abs(degrees) - d) * 60)
    s = (abs(degrees) - d - m / 60.0) * 3600.0
    if s >= 59.95:
        s -= 60.0
        m += 1
    if m >= 60:
        m -= 60
        d += 1
    return f"{sign}{d:02d}:{m:02d}:{s:05.2f}"
