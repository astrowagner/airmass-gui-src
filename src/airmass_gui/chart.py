"""Matplotlib rendering of airmass charts for the desktop app (backend-agnostic).

The static PNG output shares its astronomy (airmass, sun shading) and its
dark theme with the interactive site in airmass_gui.site.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from astropy import units as u
from astropy.time import Time

from .astro import (
    altitude_from_airmass,
    airmass_series,
    horizon_airmass,
    sky_coord_from_entries,
    sun_shading_intervals,
)
from .observatories import from_dict

# Axis geometry. Altitude defines the vertical scale: 0 deg (horizon) at the
# bottom, 90 deg (zenith, X = 1) at the top. Airmass values are placed on the
# left axis at their true (nonlinear) altitudes, so the airmass axis is
# effectively inverted: X = 1 at the top, growing toward the horizon.

# Curve colors for targets (tab10); shared by the GUI and the static site.
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Altitude limits (deg) and tick values for the right axis.
ALT_MIN = 0.0
ALT_MAX = 90.0
ALT_TICKS = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]

# Airmass tick values for the left axis.
AIRMASS_TICKS = [1.0, 1.5, 2.0, 4.0, 8.0, 20.0]

# Dark theme: shared by the GUI canvas and headless PNG output.
FIG_BG = "#1e1e2e"
AXES_BG = "#26263a"
FG = "#cdd6f4"
FG_DIM = "#9aa5c4"
BORDER = "#45475a"
ERROR_FG = "#f38ba8"

# Twilight shading per solar-altitude class: (color, alpha). The site
# blends the color over the active theme's axis background at runtime;
# the PNG output blends it over the dark theme: day is a light band,
# night is darker than the base.
SHADE = {
    "day": ("#e6e3d8", 0.30),
    "civil": ("#ffd966", 0.28),
    "nautical": ("#89b4d6", 0.22),
    "astro": ("#55779f", 0.22),
    "night": ("#0b101f", 0.55),
}


def _tz(name: str) -> ZoneInfo:
    """Resolve an IANA time-zone name, falling back to UTC on bad input.

    Parameters
    ----------
    name : str
        IANA time-zone name.

    Returns
    -------
    ZoneInfo
        The zone, or ZoneInfo("UTC") if the name is not recognized.
    """
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("UTC")


def _mjd_to_datetime(mjd: float) -> datetime:
    """Convert Julian Date 2400000.0-based MJD to a naive UTC datetime.

    Parameters
    ----------
    mjd : float
        Modified Julian Date (2400000.5 = 1858-11-17).

    Returns
    -------
    datetime
        Naive UTC datetime.
    """
    return datetime(1858, 11, 17) + timedelta(days=float(mjd))


def _parse_range(data: dict):
    """Extract the observatory and time range from a workspace document.

    Parameters
    ----------
    data : dict
        Workspace document.

    Returns
    -------
    tuple of (Observatory, Time, Time, ZoneInfo)
        The observatory, range start and end (UTC), and the observatory
        time zone.
    """
    obs = from_dict(data.get("observatory", {}))
    time_range = data.get("time_range", {})
    start_naive = datetime.fromisoformat(time_range.get("start", "1985-10-26 12:00"))
    tz = _tz(obs.timezone)
    start = Time(start_naive.replace(tzinfo=tz), format="datetime")
    end = start + float(time_range.get("days", 1.0)) * u.day
    return obs, start, end, tz


def _airmass_altitude(x):
    """Map airmass values to the altitude axis; below-horizon (inf) -> horizon.

    Parameters
    ----------
    x : array_like
        Airmass values; inf marks below-horizon points.

    Returns
    -------
    ndarray
        Altitude in degrees for each value, clipped to the 0-90 axis.
    """
    return altitude_from_airmass(np.where(np.isfinite(x), x, horizon_airmass()))


def _setup_axes(ax):
    """Set up the dual y axes: airmass (left, inverted) and altitude (right).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The main axes.

    Returns
    -------
    matplotlib.axes.Axes
        The twin right axes carrying the altitude scale.
    """
    ax.set_ylim(ALT_MIN, ALT_MAX)
    ax.set_yticks([float(_airmass_altitude(np.array([v]))[0]) for v in AIRMASS_TICKS])
    ax.set_yticklabels([f"{v:g}" for v in AIRMASS_TICKS])
    ax.set_ylabel("Airmass X")
    alt_ax = ax.twinx()
    alt_ax.set_ylim(ALT_MIN, ALT_MAX)
    alt_ax.set_yticks(ALT_TICKS)
    alt_ax.set_yticklabels([f"{int(t)}\u00b0" for t in ALT_TICKS])
    alt_ax.set_ylabel("Altitude")
    return alt_ax


def _dark_axes(ax, alt_ax):
    """Apply the dark theme (background, spines, ticks, labels) to the axes.

    Parameters
    ----------
    ax, alt_ax : matplotlib.axes.Axes
        The main and twin axes.
    """
    ax.set_facecolor(AXES_BG)
    for spine in list(ax.spines.values()) + list(alt_ax.spines.values()):
        spine.set_color(BORDER)
    for axis in (ax, alt_ax):
        axis.tick_params(colors=FG)
        axis.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.title.set_color(FG)

def render_figure(data: dict, out_path=None):
    """Render the airmass chart for a workspace document.

    Draws twilight shading, one curve per target (Kasten & Young airmass on
    the left axis, altitude on the right), the X = 2 reference line, and
    per-target minimum labels.

    Parameters
    ----------
    data : dict
        Workspace document.
    out_path : str or Path, optional
        If given, the figure is also saved there as PNG (150 dpi).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure. It is not registered with pyplot, so the
        caller is responsible for closing it once its canvas or PNG
        is done with it.
    """
    obs, start, end, tz = _parse_range(data)
    # A standalone Figure (not plt.subplots) so pyplot registers nothing:
    # the desktop app redraws on every change and stacked figures would
    # pile up.
    fig = Figure(figsize=(10.5, 5.2), facecolor=FIG_BG)
    ax = fig.add_subplot(111)

    for mjd_a, mjd_b, kind in sun_shading_intervals(obs, start, end):
        color, alpha = SHADE[kind]
        if color is None:
            continue
        ax.axvspan(
            mdates.date2num(_mjd_to_datetime(mjd_a)),
            mdates.date2num(_mjd_to_datetime(mjd_b)),
            facecolor=color,
            alpha=alpha,
            linewidth=0.0,
            zorder=0,
        )

    for i, obj in enumerate(data.get("objects", [])):
        name = obj.get("name", f"Target {i + 1}")
        color = obj.get("color", "#1f77b4")
        try:
            coord = sky_coord_from_entries(obj.get("ra", ""), obj.get("dec", ""))
        except ValueError as exc:
            ax.text(0.02, 0.03 + 0.05 * i, f"{name}: {exc}", transform=ax.transAxes, fontsize=8, color=ERROR_FG)
            continue
        times, _alt, x = airmass_series(obs, coord, start, end)
        xnum = mdates.date2num(times.to_datetime().astype("datetime64[s]").astype(object))
        mask = np.isfinite(x)
        if not mask.any():
            ax.text(
                0.98,
                0.94 - 0.06 * i,
                f"{name}: below horizon for whole range",
                transform=ax.transAxes,
                ha="right",
                fontsize=8,
                color=FG_DIM,
            )
            continue
        # below-horizon points (X = inf) map to the horizon and are clipped away
        ax.plot(xnum[mask], _airmass_altitude(x[mask]), color=color, linewidth=1.6, label=name)
        x_finite = np.where(mask, x, np.inf)
        imin = int(np.argmin(x_finite))
        ax.annotate(f"min {x[imin]:.2f}", xy=(xnum[imin], float(_airmass_altitude(x[imin:imin + 1])[0])),
                    xytext=(5, 5), textcoords="offset points", fontsize=8, color=color)

    ax.set_xlim(
        mdates.date2num(_mjd_to_datetime(float(start.to_value("mjd")))),
        mdates.date2num(_mjd_to_datetime(float(end.to_value("mjd")))),
    )
    alt_x2 = float(_airmass_altitude(np.array([2.0]))[0])
    ax.axhline(alt_x2, color=FG_DIM, linestyle=":", linewidth=1.0, zorder=1)
    ax.text(ax.get_xlim()[1], alt_x2, "X = 2", ha="right", va="bottom", fontsize=8, color=FG_DIM)
    alt_ax = _setup_axes(ax)
    ax.set_xlabel(f"Local time ({obs.timezone})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %H:%M", tz=tz))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=tz))
    ax.grid(True, color=BORDER, alpha=0.6)
    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        facecolor=AXES_BG,
        edgecolor=BORDER,
        labelcolor=FG,
    )

    dt_start = datetime.fromtimestamp(float(start.to_value("unix"))).astimezone(tz)
    dt_end = datetime.fromtimestamp(float(end.to_value("unix"))).astimezone(tz)
    fig.suptitle(f"Airmass chart — {obs.name}", fontsize=11, color=FG)
    ax.set_title(f"{dt_start:%Y-%m-%d %H:%M} \u2192 {dt_end:%Y-%m-%d %H:%M} local", fontsize=9, color=FG)
    _dark_axes(ax, alt_ax)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
