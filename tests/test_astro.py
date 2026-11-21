import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from airmass_gui.astro import (
    altitude_from_airmass,
    airmass_from_altitude,
    airmass_series,
    format_dec_dms,
    format_ra_hours,
    horizon_airmass,
    parse_dec,
    parse_ra,
    sky_coord_from_entries,
    sun_shading_intervals,
)
from airmass_gui.chart import AIRMASS_TICKS, ALT_TICKS, render_figure
from airmass_gui.observatories import PRESETS
from airmass_gui.workspace import new_workspace

LBT = PRESETS["LBT (Large Binocular Telescope)"]
VEGA = SkyCoord("18h36m56.30142s", "+38d47m01.4506s", frame="icrs")
PHOENIX = ZoneInfo("America/Phoenix")


def start_end(days=1.0):
    start = Time(datetime(2026, 11, 20, 12, 0, tzinfo=PHOENIX), format="datetime")
    return start, start + days * u.day


def to_local(mjd, tz):
    return Time(mjd, format="mjd").datetime.replace(tzinfo=__import__("datetime").timezone.utc).astimezone(tz)


class AirmassFormulaTest(unittest.TestCase):
    def test_zenith(self):
        self.assertAlmostEqual(float(airmass_from_altitude(np.array([90.0]))[0]), 1.0, places=3)

    def test_30_deg_altitude(self):
        x = float(airmass_from_altitude(np.array([30.0]))[0])
        self.assertAlmostEqual(x, 1.9949, places=4)

    def test_below_horizon(self):
        self.assertTrue(np.isinf(airmass_from_altitude(np.array([-0.5]))[0]))

    def test_horizon_airmass(self):
        self.assertAlmostEqual(horizon_airmass(), 37.93, places=2)

    def test_inverse_zenith(self):
        self.assertEqual(float(altitude_from_airmass(np.array([1.0]))[0]), 90.0)

    def test_inverse_horizon(self):
        self.assertAlmostEqual(float(altitude_from_airmass(np.array([horizon_airmass()]))[0]), 0.0, places=6)

    def test_inverse_known(self):
        self.assertAlmostEqual(float(altitude_from_airmass(np.array([2.0]))[0]), 29.9145, places=4)

    def test_inverse_roundtrip(self):
        alts = np.array([1.0, 5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0])
        back = altitude_from_airmass(airmass_from_altitude(alts))
        self.assertLess(float(np.max(np.abs(back - alts))), 1e-6)

    def test_inverse_clipping(self):
        self.assertAlmostEqual(float(altitude_from_airmass(np.array([99.0]))[0]), 0.0, places=6)
        self.assertEqual(float(altitude_from_airmass(np.array([0.5]))[0]), 90.0)


class VegaAtLBTTest(unittest.TestCase):
    def test_transit_airmass(self):
        start, end = start_end()
        _times, alt, x = airmass_series(LBT, VEGA, start, end)
        finite = np.isfinite(x)
        self.assertTrue(finite.any(), "Vega should be above the horizon at LBT")
        self.assertLess(float(x[finite].min()), 1.01)
        self.assertGreater(float(alt[finite].max()), 83.0)

    def test_one_rise_one_set(self):
        start, end = start_end()
        _times, alt, _x = airmass_series(LBT, VEGA, start, end)
        up = alt > 0.0
        self.assertEqual(int(np.sum(up[1:] & ~up[:-1])), 1)
        self.assertEqual(int(np.sum(~up[1:] & up[:-1])), 1)


class SunEventsTest(unittest.TestCase):
    def test_night_window(self):
        start, end = start_end()
        intervals = sun_shading_intervals(LBT, start, end)
        kinds = [kind for _a, _b, kind in intervals]
        self.assertIn("day", kinds)
        self.assertIn("night", kinds)
        night = [(a, b) for a, b, kind in intervals if kind == "night"]
        self.assertEqual(len(night), 1)
        t_start = to_local(night[0][0], PHOENIX)
        t_end = to_local(night[0][1], PHOENIX)
        self.assertEqual(t_start.date(), datetime(2026, 11, 20).date())
        self.assertGreater(t_start.hour + t_start.minute / 60.0, 15.5)
        self.assertLess(t_start.hour + t_start.minute / 60.0, 19.5)
        self.assertGreater(t_end.hour, 4.5)
        self.assertLess(t_end.hour, 9.5)


class ParseTest(unittest.TestCase):
    def test_ra_hms(self):
        self.assertAlmostEqual(parse_ra("18:36:56.30142"), 18 + 36 / 60 + 56.30142 / 3600, places=9)

    def test_ra_suffixes(self):
        self.assertAlmostEqual(parse_ra("18h36m56.3s"), 18 + 36 / 60 + 56.3 / 3600, places=6)

    def test_ra_decimal_hours(self):
        self.assertAlmostEqual(parse_ra("18.615638"), 18.615638, places=9)

    def test_ra_decimal_degrees(self):
        self.assertAlmostEqual(parse_ra("279.23458"), 279.23458 / 15, places=6)

    def test_dec_dms(self):
        self.assertAlmostEqual(parse_dec("+38:47:01.45"), 38 + 47 / 60 + 1.45 / 3600, places=9)
        self.assertAlmostEqual(parse_dec("-30:14:42"), -(30 + 14 / 60 + 42 / 3600), places=9)

    def test_dec_decimal(self):
        self.assertAlmostEqual(parse_dec("38.78374"), 38.78374, places=9)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            parse_ra("99:99:99")
        with self.assertRaises(ValueError):
            parse_dec("120")
        with self.assertRaises(ValueError):
            sky_coord_from_entries("not a ra", "not a dec")


class FormatTest(unittest.TestCase):
    def test_ra_format(self):
        self.assertEqual(format_ra_hours(279.234583), "18:36:56.30")

    def test_dec_format(self):
        self.assertEqual(format_dec_dms(38 + 47 / 60 + 1.45 / 3600), "+38:47:01.45")
        self.assertEqual(format_dec_dms(-30.244), "-30:14:38.40")


class FigureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg")

    def test_render_default_workspace(self):
        data = new_workspace()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chart.png"
            fig = render_figure(data, out)
            self.assertIsNotNone(fig)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 5000)

    def test_axes_layout(self):
        data = new_workspace()
        fig = render_figure(data)
        self.assertEqual(len(fig.axes), 2)
        ax, alt_ax = fig.axes[0], fig.axes[1]
        # Altitude defines the vertical axis: 0 deg at the bottom, 90 deg at the top.
        self.assertEqual(ax.get_ylim(), (0.0, 90.0))
        self.assertEqual(alt_ax.get_ylim(), (0.0, 90.0))
        # Left axis: airmass ticks at their true (nonlinear) altitudes, inverted
        # (X = 1 at the top, largest X near the horizon).
        expected = altitude_from_airmass(np.array(AIRMASS_TICKS))
        ticks = ax.get_yticks()
        self.assertAlmostEqual(float(max(ticks)), 90.0, places=9)
        self.assertLess(float(min(ticks)), 5.0)
        labels = [lab.get_text() for lab in ax.get_yticklabels()]
        self.assertEqual(len(ticks), len(AIRMASS_TICKS))
        for tick, label in zip(ticks, labels):
            idx = int(np.argmin(np.abs(expected - tick)))
            self.assertLess(abs(tick - expected[idx]), 1e-9)
            self.assertEqual(label, f"{AIRMASS_TICKS[idx]:g}")
        # Right axis: plain altitude scale 0-90 deg.
        self.assertTrue(np.allclose(alt_ax.get_yticks(), ALT_TICKS))
        self.assertEqual([lab.get_text() for lab in alt_ax.get_yticklabels()],
                         [f"{int(t)}\u00b0" for t in ALT_TICKS])
        # Both axes must agree: the X = 2 tick pixel reads as its true altitude.
        renderer = fig.canvas.get_renderer()
        x2_alt = float(altitude_from_airmass(np.array([2.0]))[0])
        px = ax.transData.transform((0, x2_alt))[1]
        self.assertAlmostEqual(alt_ax.transData.inverted().transform((0, px))[1], x2_alt, places=4)
        for lab in ax.get_yticklabels() + alt_ax.get_yticklabels():
            ext = lab.get_window_extent(renderer=renderer)
            self.assertGreater(ext.width, 0)
            self.assertGreater(ext.height, 0)
