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
from urllib.parse import quote, unquote
from pathlib import Path

import numpy as np

import html as html_mod
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, TETE, get_body, get_sun
from astropy.time import Time

from airmass_gui import astro, chart, lookup
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
    "horizonCoords", "parallacticAngle", "precessIcrsToTete", "kastyAir", "airmass", "horizonX", "altFromAirmass",
    "sunRaDec", "sunAltitude", "classifySunAlt", "sunShading", "parseRa",
    "parseDec", "simbadParseRecord",
    "encodeShareFragment", "decodeShareFragment", "htmlEscape",
    "trueRuns", "transitTimes", "twilightEvents",
    "moonRaDec", "moonPhase", "angularSep",
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

    def test_tooltip_and_view_navigation(self):
        self.assertIn('<div id="tooltip">', self.page)
        self.assertIn('<div id="xhair">', self.page)
        self.assertIn("position: relative", self.page)
        self.assertIn('"ew-resize"', self.page)
        self.assertIn("dblclick", self.page)
        self.assertIn("passive: false", self.page)

    def test_workspace_controls(self):
        for token in ('id="ws-select"', 'id="import-file"', 'id="png-btn"',
                      'id="copylink-btn"', 'accept=".json,application/json"'):
            self.assertIn(token, self.page)
        self.assertIn("function htmlEscape(", self.core)

    def test_windows_ui(self):
        self.assertIn('<input id="windows-chk"', self.page)
        self.assertIn('<input id="threshold"', self.page)
        self.assertIn('<div id="summary">', self.page)
        self.assertIn("function renderSummary()", self.page)

    def test_moon_ui(self):
        self.assertIn('<input id="moon-chk"', self.page)
        self.assertIn("function moonRaDec(", self.core)
        self.assertIn("function moonPhase(", self.core)
        self.assertIn("function angularSep(", self.core)

    def test_twilight_lines_and_gestures_ui(self):
        self.assertIn('<input id="civil-chk"', self.page)
        self.assertIn('<input id="nautical-chk"', self.page)
        self.assertIn('"touchstart"', self.page)
        self.assertIn('"keydown"', self.page)

    def test_shareable_url(self):
        self.assertIn('<button id="copylink-btn"', self.page)
        self.assertIn("function encodeShareFragment(", self.core)
        self.assertIn("function decodeShareFragment(", self.core)
        self.assertIn("history.replaceState", self.page)

    def test_title_uses_observatory_name(self):
        name = html_mod.escape(self.data["observatory"]["name"])
        self.assertIn(f"<title>Airmass chart &mdash; {name}</title>", self.page)

    def test_no_external_asset_references(self):
        # The page must be self-contained: no external assets may be loaded.
        # (The Simbad mirror URLs are a runtime opt-in API call; the mirrors
        # send Access-Control-Allow-Origin: *, so nothing is embedded.)
        for token in ("<script src", "<link ", "@import", "url(", "<img",
                      "<iframe", "<object", "<embed"):
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

    def test_horizon_coords(self):
        # Same pattern as test_altitude: feed true equator-of-date
        # coordinates and compare against astropy's refraction-free
        # AltAz to ~0.01 deg (azimuth to ~0.02 deg: the TETE-vs-mean
        # equinox nutation offset is amplified by 1/sin(alt) in azimuth).
        # Azimuth is clockwise from north.
        for iso in ("2026-11-20 22:00:00", "2026-11-20 08:30:00"):
            t = Time(iso)
            frame = AltAz(obstime=t, location=astro.location(LBT))
            ref = VEGA.transform_to(frame)
            tete = VEGA.transform_to(TETE(obstime=t))
            got = json.loads(
                self._js(
                    f"JSON.stringify(horizonCoords({tete.ra.to_value(u.hourangle)}, {tete.dec.deg}, {t.jd}, {LBT.lat}, {LBT.lon}))"
                )
            )
            self.assertAlmostEqual(got["alt"], ref.alt.deg, delta=0.01, msg=iso)
            daz = abs(got["az"] - ref.az.deg) % 360.0
            self.assertLessEqual(min(daz, 360.0 - daz), 0.02, msg=iso)

    def test_parallactic_angle(self):
        # Reference: angle at the object between the vertical circle and
        # the hour circle, from ENU unit vectors. The hour angle is taken
        # from the core's own lstDeg so LST precision cannot pollute the
        # comparison (LST accuracy is covered by test_gmst).
        rng = random.Random(42)
        checked = 0
        for _ in range(40):
            jd = rng.uniform(2451545.0, 2490000.0)
            lat = rng.uniform(-60.0, 60.0)
            lon = rng.uniform(-180.0, 180.0)
            dec = rng.uniform(-85.0, 85.0)
            if abs(dec - lat) < 3.0:
                continue  # near zenith: the angle is ill-conditioned
            ra_h = rng.uniform(0.0, 24.0)
            H = math.radians((self._js(f"lstDeg({jd}, {lon})") - ra_h * 15.0) % 360.0)
            phi = math.radians(lat)
            dec_r = math.radians(dec)
            cd, sd = math.cos(dec_r), math.sin(dec_r)
            sp, cp = math.sin(phi), math.cos(phi)
            ch, sh = math.cos(H), math.sin(H)
            u = np.array([-cd * sh, sd * cp - cd * sp * ch, sd * sp + cd * cp * ch])
            u /= np.linalg.norm(u)
            z = np.array([0.0, 0.0, 1.0])
            pole = np.array([0.0, cp, sp])
            t_vert = z - np.dot(z, u) * u
            t_vert /= np.linalg.norm(t_vert)
            t_hour = pole - np.dot(pole, u) * u
            t_hour /= np.linalg.norm(t_hour)
            q_ref = math.degrees(
                math.atan2(np.dot(u, np.cross(t_vert, t_hour)), np.dot(t_vert, t_hour))
            )
            got = self._js(f"parallacticAngle({ra_h}, {dec}, {jd}, {lat}, {lon})")
            dq = abs(got - q_ref) % 360.0
            self.assertLessEqual(
                min(dq, 360.0 - dq), 1e-5, msg=(round(jd, 4), round(lat, 2), round(dec, 2))
            )
            checked += 1
        self.assertGreaterEqual(checked, 30)

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

    def test_simbad_parse(self):
        # The page's parser must agree with the desktop app's on a real record.
        fixture = (Path(__file__).parent / "fixtures" / "simbad_vega.txt").read_text(encoding="utf-8")
        expected = lookup._parse_ascii_record(fixture)
        got = self._js(f"simbadParseRecord({json.dumps(fixture)})")
        self.assertIsNotNone(got, "JS parser found no coordinates")
        self.assertEqual(got["name"], expected["name"])
        self.assertAlmostEqual(got["raDeg"], expected["ra"], delta=1e-9)
        self.assertAlmostEqual(got["dec"], expected["dec"], delta=1e-9)
        no_coords = "C.D.S.  -  SIMBAD4\n\nSome object\n----\nno coordinates here"
        self.assertIsNone(self._js(f"simbadParseRecord({json.dumps(no_coords)})"))


    def test_share_fragment_round_trip(self):
        doc = {
            "observatory": {"name": "LBT (Large Binocular Telescope)", "lat": 31.76,
                            "lon": -111.37, "elevation_m": 2283.0,
                            "timezone": "America/Phoenix",
                            "preset": "LBT (Large Binocular Telescope)"},
            "objects": [{"name": "Vega & <friends>", "ra": "18:36:56.3", "dec": "+38:47:01",
                         "color": "#89b4fa"}],
            "time_range": {"start": "2026-11-20 00:00", "days": 2.5},
            "options": {"threshold": 1.5, "windows": True, "moon": False,
                        "civil": False, "nautical": False},
        }
        # JS-encode -> Python-decode
        frag = self._js("encodeShareFragment(" + json.dumps(json.dumps(doc)) + ")")
        self.assertTrue(frag.startswith("#w="))
        self.assertEqual(json.loads(unquote(frag[3:])), doc)
        # Python-encode -> JS-decode (round trip only: the alphabets of
        # encodeURIComponent and quote(safe="") differ on !'()* )
        got = json.loads(self._js(
            "JSON.stringify(decodeShareFragment('#w=' + " + json.dumps(quote(json.dumps(doc), safe="")) + "))"
        ))
        self.assertEqual(got, doc)
        # malformed fragments
        self.assertIsNone(self._js("decodeShareFragment('')"))
        self.assertIsNone(self._js("decodeShareFragment('#w=not json')"))
        self.assertIsNone(self._js("decodeShareFragment('#x=1')"))
        self.assertIsNone(self._js("decodeShareFragment('#w=' + " + json.dumps(quote("[1,2]")) + ")"))


    def test_html_escape(self):
        cases = [
            ("plain", "plain"),
            ("<b>", "&lt;b&gt;"),
            ('"q"', "&quot;q&quot;"),
            ("o'clock", "o&#39;clock"),
            ("a&b", "a&amp;b"),
            ("</script><img src=x>", "&lt;/script&gt;&lt;img src=x&gt;"),
        ]
        for text, want in cases:
            self.assertEqual(self._js("htmlEscape(" + json.dumps(text) + ")"), want)

    def test_true_runs(self):
        rng = random.Random(7)
        for trial in range(25):
            n = rng.randint(0, 200)
            ok = [rng.random() < 0.5 for _ in range(n)]
            start = 1.7e12
            step = 60000.0
            got = json.loads(
                self._js(f"JSON.stringify(trueRuns({json.dumps(ok)}, {start}, {step}))")
            )
            arr = np.array(ok, dtype=bool)
            padded = np.concatenate(([False], arr, [False]))
            idx = np.where(padded[:-1] != padded[1:])[0]
            runs = []
            for i in range(0, len(idx), 2):
                a = start + idx[i] * step - step / 2
                b = start + (idx[i + 1] - 1) * step + step / 2
                runs.append((a, b))
            self.assertEqual(len(got), len(runs), (trial, ok))
            for (g, r) in zip(got, runs):
                self.assertAlmostEqual(g[0], r[0], delta=1e-6)
                self.assertAlmostEqual(g[1], r[1], delta=1e-6)

    def test_transit_times(self):
        start = Time("2026-11-20 00:00:00")
        end = Time("2026-11-21 06:00:00")
        start_ms = float(start.to_value("unix")) * 1000.0
        end_ms = float(end.to_value("unix")) * 1000.0
        # Reference: 60-s scan of the mean sidereal time, linear
        # interpolation on the wrap.
        grid = np.arange(start.mjd, end.mjd + 1e-9, 60.0 / 86400.0)
        lst_deg = (
            Time(grid, format="mjd")
            .sidereal_time("mean", longitude=LBT.lon * u.deg)
            .to_value(u.deg)
        )
        for ra_h in (0.1, 6.5, 12.3, 18.6, 23.9):
            d = (lst_deg - ra_h * 15.0) % 360.0
            idx = np.where((d[:-1] > 180.0) & (d[1:] <= 180.0))[0]
            ref = []
            for i in idx:
                a0, a1 = float(d[i]), float(d[i + 1])
                frac = (360.0 - a0) / ((360.0 - a0) + a1)
                ref.append((float(grid[i]) + frac * (60.0 / 86400.0) - 40587.0) * 86400000.0)
            got = json.loads(
                self._js(f"JSON.stringify(transitTimes({ra_h}, {start_ms}, {end_ms}, {LBT.lon}))")
            )
            self.assertEqual(len(got), len(ref), ra_h)
            for g, r in zip(got, ref):
                self.assertAlmostEqual(g, r, delta=1000.0)  # 1 s

    def test_twilight_events(self):
        start = Time("2026-11-20 00:00:00")
        end = Time("2026-11-21 00:00:00")
        intervals = astro.sun_shading_intervals(LBT, start, end)
        kinds = [k for (_a, _b, k) in intervals]
        bounds = [float(start.to_value("mjd"))] + [float(b) for (_a, b, _k) in intervals]
        start_ms = float(start.to_value("unix")) * 1000.0
        end_ms = float(end.to_value("unix")) * 1000.0
        for thr, above in (
            (-6.0, {"day", "civil"}),
            (-12.0, {"day", "civil", "nautical"}),
        ):
            ref = []
            for i in range(len(kinds) - 1):
                if (kinds[i] in above) != (kinds[i + 1] in above):
                    ref.append((bounds[i + 1] - 40587.0) * 86400000.0)
            got = json.loads(
                self._js(f"JSON.stringify(twilightEvents({start_ms}, {end_ms}, {LBT.lat}, {LBT.lon}, {thr}))")
            )
            self.assertEqual(len(got), len(ref), thr)
            for g, r in zip(got, ref):
                self.assertAlmostEqual(g, r, delta=30000.0)  # 30 s

    def test_moon_position(self):
        # Truncated NAO 1954/ELP theory: ~0.015 deg from astropy's full ELP.
        for iso in ("2000-01-01 00:00:00", "2026-11-20 22:00:00",
                    "2027-06-21 03:00:00", "2050-06-01 00:00:00",
                    "2090-01-01 00:00:00"):
            t = Time(iso)
            moon = get_body("moon", t, location=EarthLocation(x=0, y=0, z=0, unit="m")).transform_to(TETE(obstime=t))
            got = json.loads(self._js(f"JSON.stringify(moonRaDec({float(t.jd)}))"))
            got_coord = SkyCoord(ra=got["raH"] * 15.0 * u.deg, dec=got["dec"] * u.deg,
                                 frame="tete", obstime=t)
            self.assertLessEqual(got_coord.separation(moon).deg, 0.03, iso)
            self.assertLessEqual(abs(got["dec"] - moon.dec.deg), 0.02, iso)
            self.assertLessEqual(abs(got["dec"]), 28.6, iso)

    def test_moon_phase(self):
        # Reference in TET (equator of date), the frame the JS core works in.
        t = Time("2026-11-20 22:00:00")
        moon = get_body("moon", t, location=EarthLocation(x=0, y=0, z=0, unit="m")).transform_to(TETE(obstime=t))
        sun = get_sun(t).transform_to(TETE(obstime=t))
        sep_ref = moon.separation(sun).deg
        got = json.loads(self._js(f"JSON.stringify(moonPhase({float(t.jd)}))"))
        self.assertLessEqual(abs(got["sepDeg"] - sep_ref), 0.05)
        self.assertLessEqual(abs(got["illum"] - (1 - math.cos(math.radians(sep_ref))) / 2), 0.005)

    def test_angular_sep(self):
        self.assertAlmostEqual(self._js("angularSep(0, 0, 6, 0)"), 90.0, delta=1e-9)
        self.assertAlmostEqual(self._js("angularSep(12, 45, 12, 45)"), 0.0, delta=1e-9)
        self.assertAlmostEqual(self._js("angularSep(0, 0, 12, 0)"), 180.0, delta=1e-9)
        # Both targets at the north celestial pole: RA is undefined there, sep 0.
        self.assertAlmostEqual(self._js("angularSep(0, 90, 12, 90)"), 0.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
