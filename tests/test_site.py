"""Tests for the interactive single-file HTML site (airmass_gui.site).

The static checks always run. The numeric cross-checks evaluate the page's
embedded JS astronomy core in a real JS engine (dukpy / Duktape) and compare
against astropy, with tolerances chosen from the documented differences
(e.g. Meeus low-precision Sun vs. astropy: ~0.4 deg in RA).
"""
import json
import math
import random
import re
import tempfile
import unittest
from pathlib import Path

import html as html_mod
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, TETE, get_sun
from astropy.time import Time

from airmass_gui import astro, chart
from airmass_gui.observatories import PRESETS, to_dict
from airmass_gui.site import build_site
from airmass_gui.workspace import new_workspace

try:
    import dukpy
except ImportError:  # pragma: no cover
    dukpy = None

LBT = PRESETS["LBT (Large Binocular Telescope)"]
VEGA = SkyCoord("18h36m56.30142s", "+38d47m01.4506s", frame="icrs")

CORE_BEGIN = "/*JS-CORE-BEGIN*/"
CORE_END = "/*JS-CORE-END*/"

CORE_FUNCTIONS = [
    "wrap360", "d2r", "r2d", "jdFromUtcMs", "gmstMeeus", "lstDeg", "altitude",
    "precessIcrsToTete", "kastyAir", "airmass", "horizonX", "altFromAirmass",
    "sunRaDec", "sunAltitude", "classifySunAlt", "sunShading", "parseRa",
    "parseDec",
]

RA_OK = ["18:36:56.30142", "18.616h", "279.2346", "0", "23:59:59.9", "7.5"]
RA_BAD = ["", "abc", "1:70:0", "2:0:70", "1 2 3 4", "12.5.7", "5 55 40.8"]
DEC_OK = ["+38:47:01.4506", "-38:47:01", "45\u00b0 30\u2032 15\u2033", "-89.99", "0", "12d 30m 45s"]
DEC_BAD = ["", "abc", "91", "-95", "30:70", "1 2 3 4"]


def _extract_core(page: str) -> str:
    begin = page.index(CORE_BEGIN) + len(CORE_BEGIN)
    return page[begin : page.index(CORE_END)]


def _strip_js_comments(code: str) -> str:
    # The core has no block comments and no "//" inside string/regex literals.
    return re.sub(r"//[^\n]*", "", code)


def _embedded_json(page: str, name: str):
    """Extract the single-line JSON literal assigned to `var <name> =`."""
    marker = f"var {name} ="
    idx = page.index(marker) + len(marker)
    while page[idx] in " \t\r\n":
        idx += 1
    start = idx
    assert page[start] in "{[", f"{name} is not a JSON literal"
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(page)):
        ch = page[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return json.loads(page[start : i + 1])
    raise AssertionError(f"unbalanced JSON literal for {name}")


class SitePageTests(unittest.TestCase):
    """Static checks on the generated page (no JS engine needed)."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.data = new_workspace()
        cls.path = build_site(cls.data, Path(cls.dir.name) / "index.html")
        cls.page = cls.path.read_text(encoding="utf-8")
        cls.core = _extract_core(cls.page)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_writes_single_file(self):
        self.assertTrue(self.path.is_file())
        self.assertGreater(self.path.stat().st_size, 10_000)
        self.assertEqual([p.name for p in Path(self.dir.name).iterdir()], ["index.html"])

    def test_canvas_and_dark_theme(self):
        self.assertIn('<canvas id="chart">', self.page)
        self.assertIn(f"--bg: {chart.FIG_BG}", self.page)
        for color in (chart.AXES_BG, chart.FG, chart.FG_DIM, chart.BORDER, chart.ERROR_FG):
            self.assertIn(color, self.page)

    def test_title_uses_observatory_name(self):
        name = html_mod.escape(self.data["observatory"]["name"])
        self.assertIn(f"<title>Airmass chart &mdash; {name}</title>", self.page)

    def test_no_external_references(self):
        for token in ("http", "src=", "href=", "url(", "@import"):
            self.assertNotIn(token, self.page)

    def test_no_unreplaced_tokens(self):
        self.assertIsNone(re.search(r"__[A-Z_]+__", self.page))

    def test_embedded_data_round_trips(self):
        self.assertEqual(_embedded_json(self.page, "INITIAL"), self.data)
        self.assertEqual(
            _embedded_json(self.page, "OBSERVATORIES"),
            [to_dict(p) for p in PRESETS.values()],
        )
        self.assertEqual(_embedded_json(self.page, "PALETTE"), chart.PALETTE)

    def test_embedded_shade_colors(self):
        shade = _embedded_json(self.page, "SHADE")
        self.assertEqual(set(shade), set(chart.SHADE))
        for color in shade.values():
            self.assertRegex(color, r"^#[0-9a-f]{6}$")

    def test_js_core_markers(self):
        self.assertIn(CORE_BEGIN, self.page)
        self.assertIn(CORE_END, self.page)

    def test_js_core_is_pure(self):
        core = _strip_js_comments(self.core)
        for token in ("document", "localStorage", "new Intl", "window"):
            self.assertNotIn(token, core)

    def test_js_core_functions_present(self):
        for name in CORE_FUNCTIONS:
            self.assertRegex(self.core, rf"function {name}\(")

    def test_script_injection_is_neutralized(self):
        data = new_workspace()
        data["objects"][0]["name"] = "</script><img src=x onerror=alert(1)>"
        with tempfile.TemporaryDirectory() as scratch:
            page = build_site(data, Path(scratch) / "evil.html").read_text(encoding="utf-8")
        self.assertEqual(page.count("</script>"), 1)
        self.assertEqual(
            _embedded_json(page, "INITIAL")["objects"][0]["name"],
            data["objects"][0]["name"],
        )


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class JsCoreNumericTests(unittest.TestCase):
    """Evaluate the page's JS core in Duktape and compare against astropy."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        page = build_site(new_workspace(), Path(cls.dir.name) / "index.html").read_text(encoding="utf-8")
        cls.js = dukpy.JSInterpreter()
        cls.js.evaljs(_extract_core(page))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def _js(self, expr: str):
        return self.js.evaljs(expr)

    def test_gmst(self):
        for jd in (2440587.5, 2451545.0, 2460234.5, 2488575.0):
            expected = (
                Time(jd, format="jd")
                .sidereal_time("mean", longitude=0 * u.deg)
                .to_value(u.hourangle)
                * 15.0
                % 360.0
            )
            got = self._js(f"gmstMeeus({jd})")
            diff = abs(got - expected) % 360.0
            self.assertLessEqual(min(diff, 360.0 - diff), 0.01)

    def test_altitude(self):
        # (a) Pure geometry: feeding altitude() true-equator-of-date
        # coordinates must match astropy's refraction-free AltAz to ~0.01 deg.
        t = Time("2026-11-20 22:00:00")
        frame = AltAz(obstime=t, location=astro.location(LBT))
        expected = VEGA.transform_to(frame).alt.deg
        tete = VEGA.transform_to(TETE(obstime=t))
        got = self._js(
            f"altitude({tete.ra.to_value(u.hourangle)}, {tete.dec.deg}, {t.jd}, {LBT.lat}, {LBT.lon})"
        )
        self.assertAlmostEqual(got, expected, delta=0.01)

        # (b) End to end, as the page draws curves: ICRS input ->
        # precessIcrsToTete -> altitude. Nutation is omitted, so allow 0.05 deg.
        for iso in (
            "2000-01-01 00:00:00",
            "2026-11-20 22:00:00",
            "2050-06-01 00:00:00",
            "2090-01-01 00:00:00",
        ):
            t = Time(iso)
            expected = VEGA.transform_to(
                AltAz(obstime=t, location=astro.location(LBT))
            ).alt.deg
            tt = json.loads(
                self._js(
                    f"JSON.stringify(precessIcrsToTete({VEGA.ra.deg}, {VEGA.dec.deg}, {t.jd}))"
                )
            )
            got = self._js(
                f"altitude({tt['raDeg'] / 15}, {tt['dec']}, {t.jd}, {LBT.lat}, {LBT.lon})"
            )
            self.assertAlmostEqual(got, expected, delta=0.05, msg=iso)

    def test_precess(self):
        # Nutation is omitted, so the residuals are nutation-level. RA is
        # ill-conditioned near the pole (1/cos dec amplification), so compare
        # great-circle distance and dec, and RA only away from the pole.
        rng = random.Random(2026)

        def sep_deg(ra1, dec1, ra2, dec2):
            d1, d2 = math.radians(dec1), math.radians(dec2)
            c = (
                math.sin(d1) * math.sin(d2)
                + math.cos(d1) * math.cos(d2) * math.cos(math.radians(ra1 - ra2))
            )
            return math.degrees(math.acos(max(-1.0, min(1.0, c))))

        for iso in (
            "2000-01-01 00:00:00",
            "2026-11-20 22:00:00",
            "2050-06-01 00:00:00",
            "2090-01-01 00:00:00",
        ):
            t = Time(iso)
            for _ in range(10):
                ra = rng.uniform(0.0, 360.0)
                dec = rng.uniform(-85.0, 85.0)
                ref = SkyCoord(ra, dec, unit=("deg", "deg"), frame="icrs").transform_to(
                    TETE(obstime=t)
                )
                got = json.loads(
                    self._js(f"JSON.stringify(precessIcrsToTete({ra}, {dec}, {t.jd}))")
                )
                ctx = (iso, ra, dec)
                self.assertLessEqual(sep_deg(ref.ra.deg, ref.dec.deg, got["raDeg"], got["dec"]), 0.02, ctx)
                self.assertLessEqual(abs(got["dec"] - ref.dec.deg), 0.01, ctx)
                if abs(dec) < 60.0:
                    dra = abs(got["raDeg"] - ref.ra.deg) % 360.0
                    self.assertLessEqual(min(dra, 360.0 - dra), 0.02, ctx)

    def test_airmass(self):
        for alt in (0.5, 10.0, 30.0, 45.0, 60.0, 89.9, 90.0):
            got = self._js(f"airmass({alt})")
            expected = float(astro.airmass_from_altitude(alt)[0])
            self.assertTrue(math.isclose(got, expected, rel_tol=1e-12), (alt, got, expected))

    def test_horizon_and_inverse(self):
        self.assertAlmostEqual(self._js("horizonX()"), astro.horizon_airmass(), delta=1e-9)
        for x in (1.0, 1.2, 2.0, 5.0, 10.0, 30.0, 37.9):
            got = self._js(f"altFromAirmass({x})")
            expected = float(astro.altitude_from_airmass(x)[0])
            self.assertAlmostEqual(got, expected, delta=1e-6)

    def test_sun_ra_dec(self):
        for iso in ("2026-11-20 21:00:00", "2027-06-21 03:00:00"):
            t = Time(iso)
            sun = get_sun(t)
            got = json.loads(self._js(f"JSON.stringify(sunRaDec({float(t.jd)}))"))
            dra = abs(got["raH"] * 15.0 - sun.ra.deg) % 360.0
            self.assertLessEqual(min(dra, 360.0 - dra), 0.5)
            self.assertLessEqual(abs(got["dec"] - sun.dec.deg), 0.15)

    def test_sun_shading(self):
        start = Time("2026-11-20 00:00:00")
        end = Time("2026-11-21 00:00:00")
        expected = astro.sun_shading_intervals(LBT, start, end)
        start_ms = float(start.to_value("unix")) * 1000.0
        end_ms = float(end.to_value("unix")) * 1000.0
        got = json.loads(
            self._js(f"JSON.stringify(sunShading({start_ms}, {end_ms}, {LBT.lat}, {LBT.lon}))")
        )
        self.assertEqual(len(got), len(expected))
        for (g_a, g_b, g_kind), (p_a, p_b, p_kind) in zip(got, expected):
            self.assertEqual(g_kind, p_kind)
            self.assertAlmostEqual(g_a, (p_a - 40587.0) * 86400000.0, delta=120000.0)
            self.assertAlmostEqual(g_b, (p_b - 40587.0) * 86400000.0, delta=120000.0)

    def test_parse_ra(self):
        for text in RA_OK:
            got = self._js(f"parseRa({json.dumps(text)})")
            self.assertAlmostEqual(got, astro.parse_ra(text), delta=1e-9)
        for text in RA_BAD:
            with self.assertRaises(ValueError):
                astro.parse_ra(text)
            with self.assertRaises(Exception):
                self._js(f"parseRa({json.dumps(text)})")

    def test_parse_dec(self):
        for text in DEC_OK:
            got = self._js(f"parseDec({json.dumps(text)})")
            self.assertAlmostEqual(got, astro.parse_dec(text), delta=1e-9)
        for text in DEC_BAD:
            with self.assertRaises(ValueError):
                astro.parse_dec(text)
            with self.assertRaises(Exception):
                self._js(f"parseDec({json.dumps(text)})")


if __name__ == "__main__":
    unittest.main()
