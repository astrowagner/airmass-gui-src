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

from airmass_gui import astro, chart, lookup, sn
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
    "horizonCoords", "parallacticAngle", "parallacticAngleRate", "precessIcrsToTete", "kastyAir", "airmass", "horizonX", "altFromAirmass",
    "sunRaDec", "sunAltitude", "classifySunAlt", "sunShading", "parseRa",
    "parseDec", "simbadParseRecord",
    "encodeShareFragment", "decodeShareFragment", "htmlEscape",
    "trueRuns", "transitTimes", "twilightEvents", "sunEvents",
    "moonRaDec", "moonPhase", "angularSep", "targetInfoText",
    "objCoords", "horizonsParse", "fieldRotationDeg",
    "gnomonicProject", "gnomonicUnproject", "formatRaDecHms",
    "simbadCooParse", "findingOffsetText", "findingDetailTable",
    "findingStarSize",
    "findingGridStep", "findingRadiusText", "findingRadiusFromZoom",
    "skyViewPosition", "skyViewQueryUrl",
    "snrArea", "snrMoonSep", "snrMoonDarkening", "snrSkyMag",
    "snrSourceRate", "snrSkyRate", "snrPsfPixels", "snrValue", "snrSolveTime",
]

RA_OK = ["18:36:56.30142", "18.616h", "279.2346", "0", "23:59:59.9", "7.5"]
RA_BAD = ["", "abc", "1:70:0", "2:0:70", "1 2 3 4", "12.5.7", "5 55 40.8"]
DEC_OK = ["+38:47:01.4506", "-38:47:01", "45\u00b0 30\u2032 15\u2033", "-89.99", "0", "12d 30m 45s"]
DEC_BAD = ["", "abc", "91", "-95", "30:70", "1 2 3 4"]


def _extract_core(page: str) -> str:
    """Return the JS text between the JS-CORE markers."""
    begin = page.index(CORE_BEGIN) + len(CORE_BEGIN)
    return page[begin : page.index(CORE_END)]


def _strip_js_comments(code: str) -> str:
    """Remove // line comments; the core has no block comments or // in literals."""
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


def _enu_parallactic_deg(hour_angle_rad: float, lat: float, dec: float) -> float:
    """Parallactic angle in degrees from the object's east-north-up frame.

    The angle at the object between the vertical circle and the hour
    circle, from east-north-up unit vectors: the object direction u, the
    zenith z, and the celestial pole pole = (0, cos lat, sin lat).
    q = atan2(u . (t_vert x t_hour), t_vert . t_hour), where t_vert and
    t_hour are the unit tangents of the vertical and hour circles at the
    object.

    Parameters
    ----------
    hour_angle_rad : float
        Hour angle H of the object, in radians.
    lat : float
        Observer latitude, in degrees.
    dec : float
        Object declination, in degrees.

    Returns
    -------
    float
        Parallactic angle q, in degrees.
    """
    H = hour_angle_rad
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
    return math.degrees(
        math.atan2(np.dot(u, np.cross(t_vert, t_hour)), np.dot(t_vert, t_hour))
    )


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
        """build_site emits exactly one index.html, > 10 kB."""
        self.assertTrue(self.path.is_file())
        self.assertGreater(self.path.stat().st_size, 10_000)
        self.assertEqual([p.name for p in Path(self.dir.name).iterdir()], ["index.html"])

    def test_canvas_and_dark_theme(self):
        """The chart canvas and dark-theme CSS variables are present."""
        self.assertIn('<canvas id="chart">', self.page)
        self.assertIn(f"--bg: {chart.FIG_BG}", self.page)
        for color in (chart.AXES_BG, chart.FG, chart.FG_DIM, chart.BORDER, chart.ERROR_FG):
            self.assertIn(color, self.page)

    def test_layout_uses_full_width(self):
        """Content spans 95% of the window width; nothing caps it, so the
        chart column grows with the window."""
        self.assertIn(".wrap { width: 95%; margin: 0 auto;", self.page)
        self.assertNotIn("max-width: 1400px", self.page)

    def test_tooltip_and_view_navigation(self):
        """Tooltip, crosshair, resize handle, and dblclick wiring are present."""
        self.assertIn('<div id="tooltip">', self.page)
        self.assertIn('<div id="xhair">', self.page)
        self.assertIn("position: relative", self.page)
        self.assertIn('"ew-resize"', self.page)
        self.assertIn("dblclick", self.page)
        self.assertIn("passive: false", self.page)

    def test_workspace_controls(self):
        """Workspace select, file import, and PNG/share buttons are present."""
        for token in ('id="ws-select"', 'id="import-file"', 'id="png-btn"',
                      'id="copylink-btn"', 'accept=".json,application/json"'):
            self.assertIn(token, self.page)
        self.assertIn("function htmlEscape(", self.core)

    def test_windows_ui(self):
        """Windows checkbox, threshold input, and summary div are present."""
        self.assertIn('<input id="windows-chk"', self.page)
        self.assertIn('<input id="threshold"', self.page)
        self.assertIn('<div id="summary">', self.page)
        self.assertIn("function renderSummary()", self.page)
        self.assertIn("sum-total", self.page)
        self.assertIn("any target", self.page)
        self.assertIn('<button id="csv-btn"', self.page)
        self.assertIn("function summaryData(", self.page)
        self.assertIn("function windowsCsv(", self.page)

    def test_moon_ui(self):
        """Moon checkbox and the three moon core functions are present."""
        self.assertIn('<input id="moon-chk"', self.page)
        self.assertIn("function moonRaDec(", self.core)
        self.assertIn("function moonPhase(", self.core)
        self.assertIn("function angularSep(", self.core)

    def test_twilight_lines_and_gestures_ui(self):
        """Civil/nautical checkboxes, touch, and key handlers are present."""
        self.assertIn('<input id="civil-chk"', self.page)
        self.assertIn('<input id="nautical-chk"', self.page)
        self.assertIn('"touchstart"', self.page)
        self.assertIn('"keydown"', self.page)

    def test_dawn_dusk_ticks(self):
        """Dawn/dusk ticks checkbox and the sunEvents core function are present."""
        self.assertIn('<input id="dusk-chk"', self.page)
        self.assertIn("state.options.dusk", self.page)
        self.assertIn("function sunEvents(", self.core)

    def test_footer_credit_and_details(self):
        """Footer info lives in an 'Extra details' dropdown; the credit names author, site, source, and license."""
        self.assertIn('<details id="extra-details">\n<summary>Extra details</summary>\n<ul>', self.page)
        self.assertIn("</ul>\n</details>", self.page)
        self.assertIn('class="credit"', self.page)
        self.assertIn("K. Wagner (UArizona)", self.page)
        self.assertIn('<a href="http://kevinwagner.space">kevinwagner.space</a>', self.page)
        self.assertIn('href="https://github.com/astrowagner/airmass-gui-src"', self.page)
        self.assertIn("MIT", self.page)

    def test_chart_navigation_controls(self):
        """The chart has a horizontal scrollbar, zoom buttons, a bottom-edge
        / bottom-right-corner resize, and an HTML legend row under the canvas."""
        self.assertIn('<div class="hscroll" id="hscroll"', self.page)
        self.assertIn('id="hscroll-thumb"', self.page)
        self.assertIn('<div id="legend"', self.page)
        self.assertIn("function renderLegend(", self.page)
        for btn in ("zoom-in", "zoom-out", "zoom-reset"):
            self.assertIn('id="%s"' % btn, self.page)
        self.assertIn("function updateHScroll(", self.page)
        self.assertIn("function zoomBy(", self.page)
        self.assertIn("function drawNow(", self.page)
        self.assertIn('chartBox.addEventListener("mousedown"', self.page)
        self.assertIn("function chartResizeMode(", self.page)
        self.assertIn('"nwse-resize"', self.page)
        self.assertIn('"ns-resize"', self.page)

    def test_panel_resize_handles(self):
        """Every rounded panel and the chart box expose the manual resize
        edges (right, bottom, corner) with matching cursors."""
        for pid in ("panel-observatory", "panel-time", "panel-display",
                    "panel-targets", "snr-panel", "panel-workspace",
                    "summary-panel", "calendar-panel"):
            self.assertIn('id="%s"' % pid, self.page)
        self.assertIn("function makePanelResizable(", self.page)
        self.assertIn("function applyPanelResize(", self.page)
        self.assertIn("function applyChartResize(", self.page)
        self.assertIn("function panelResizeMode(", self.page)
        self.assertIn("overflow: auto", self.page)

    def test_sidebar_column_resize(self):
        """The left column resizes from its right-edge strip; the width
        persists in localStorage and re-clamps on window resize."""
        self.assertIn('<div class="layout" id="layout">', self.page)
        self.assertIn('<aside id="sidebar">', self.page)
        self.assertIn('id="sidebar-strip"', self.page)
        self.assertIn("function applySidebarW(", self.page)
        self.assertIn("function sidebarWide(", self.page)
        self.assertIn('kind: "sidebar"', self.page)
        self.assertIn('"airmass-gui:sidebar-w"', self.page)
        self.assertIn("SIDEBAR_MIN_W", self.page)

    def test_extra_target_info(self):
        """The Extra info checkbox toggles the Simbad info column on the
        target rows and lives in the shared state."""
        self.assertIn('id="extra-chk"', self.page)
        self.assertIn("extraInfo: false", self.page)
        self.assertIn('$("extra-chk").addEventListener("change"', self.page)
        self.assertIn("#targets.extra .target-row", self.page)
        self.assertIn("#targets:not(.extra) .target-row .info", self.page)

    def test_today_button(self):
        """The Start field is plain 24-hour text and Today centers the range on tonight."""
        self.assertIn('<input id="start" type="text"', self.page)
        self.assertIn('placeholder="YYYY-MM-DD HH:MM"', self.page)
        self.assertIn('<button id="today-btn"', self.page)
        self.assertIn('$("today-btn").addEventListener("click"', self.page)

    def test_shareable_url(self):
        """Share button and the encode/decode core functions are present."""
        self.assertIn('<button id="copylink-btn"', self.page)
        self.assertIn("function encodeShareFragment(", self.core)
        self.assertIn("function decodeShareFragment(", self.core)
        self.assertIn("history.replaceState", self.page)

    def test_title_uses_observatory_name(self):
        """The page title carries the observatory name."""
        name = html_mod.escape(self.data["observatory"]["name"])
        self.assertIn(f"<title>Airmass chart &mdash; {name}</title>", self.page)

    def test_no_external_asset_references(self):
        """No external scripts, styles, fonts, or images are referenced."""
        # The page must be self-contained: no external assets may be loaded.
        # (The Simbad mirror URLs are a runtime opt-in API call; the mirrors
        # send Access-Control-Allow-Origin: *, so nothing is embedded. The
        # SkyView quick-query link in the finding chart is likewise a
        # runtime opt-in: it opens in a new tab only when the user clicks
        # it, so nothing is embedded.)
        for token in ("<script src", "<link ", "@import", "url(",
                      "<iframe", "<object", "<embed"):
            self.assertNotIn(token, self.page)
        self.assertNotIn("<img src", self.page)

    def test_no_unreplaced_tokens(self):
        """No unfilled __TOKEN__ placeholders remain."""
        self.assertIsNone(re.search(r"__[A-Z_]+__", self.page))

    def test_embedded_data_round_trips(self):
        """INITIAL, OBSERVATORIES, and PALETTE round-trip as embedded JSON."""
        self.assertEqual(_embedded_json(self.page, "INITIAL"), self.data)
        self.assertEqual(
            _embedded_json(self.page, "OBSERVATORIES"),
            [to_dict(p) for p in PRESETS.values()],
        )
        self.assertEqual(_embedded_json(self.page, "PALETTE"), chart.PALETTE)

    def test_embedded_shade_colors(self):
        """SHADE maps each level to a 6-digit hex color."""
        shade = _embedded_json(self.page, "SHADE")
        self.assertEqual(set(shade), set(chart.SHADE))
        for color in shade.values():
            self.assertRegex(color, r"^#[0-9a-f]{6}$")

    def test_js_core_markers(self):
        """The JS-CORE BEGIN/END markers are present."""
        self.assertIn(CORE_BEGIN, self.page)
        self.assertIn(CORE_END, self.page)

    def test_js_core_is_pure(self):
        """The core references no DOM, storage, Intl, or window."""
        core = _strip_js_comments(self.core)
        for token in ("document", "localStorage", "new Intl", "window"):
            self.assertNotIn(token, core)

    def test_js_core_functions_present(self):
        """Every function in CORE_FUNCTIONS is defined in the core."""
        for name in CORE_FUNCTIONS:
            self.assertRegex(self.core, rf"function {name}\(")

    def test_curve_control_present(self):
        """The Display panel offers the three curve modes and a handler."""
        self.assertIn('<select id="curve-sel">', self.page)
        for mode, label in (("airmass", "Airmass"),
                            ("altitude", "Altitude"),
                            ("azimuth", "Azimuth")):
            self.assertIn('<option value="%s">%s</option>' % (mode, label), self.page)
        self.assertIn('$("curve-sel").addEventListener("change"', self.page)
        self.assertIn('state.options.curve = this.value', self.page)

    def test_snr_control_present(self):
        """The S/N panel offers all nine inputs, a handler per one, and normalized state."""
        for field in ("ap", "fwhm", "pix", "rn", "dark", "eff", "k",
                      "target", "fixed"):
            self.assertIn('<input id="snr-%s"' % field, self.page)
        self.assertIn("state.options.snr[key] = v", self.page)
        self.assertIn("s.options.snr = normalizeSnr(s.options.snr);", self.page)

    def test_finding_chart_ui(self):
        """The Display panel offers a finding chart: checkbox, target,
        zoom slider and survey selector, a star-chart canvas, the SkyView
        page link, the Simbad coo lookup, and a resizable panel."""
        self.assertIn(
            '<input id="finding-chk" type="checkbox">', self.page)
        self.assertIn('<div class="panel" id="finding-panel"', self.page)
        self.assertIn('<select id="finding-target">', self.page)
        self.assertIn(
            '<input type="range" id="finding-zoom" min="0" max="1000"',
            self.page)
        self.assertIn('id="finding-zoom-label"', self.page)
        self.assertIn('<canvas id="finding-chart"></canvas>', self.page)
        self.assertIn('<select id="finding-survey">', self.page)
        self.assertIn('<a id="finding-skylink" href="#" target="_blank"',
                      self.page)
        self.assertIn('id="finding-panes"', self.page)
        self.assertIn('flex: 1 1 180px', self.page)
        self.assertIn('<option value="Digitized Sky Survey" selected>DSS</option>',
                      self.page)
        self.assertIn('<option value="HEAO 1 A-2">HEAO 1 (X-ray)</option>',
                      self.page)
        self.assertIn('<option value="RASS-Cnt Broad">RASS (X-ray)</option>',
                      self.page)
        self.assertIn('<option value="TESS">TESS</option>', self.page)
        self.assertIn('<option value="IRAS 100 micron">IRAS 100&mu;m</option>',
                      self.page)
        self.assertIn('<option value="0408MHz">0408 MHz</option>', self.page)
        self.assertNotIn('<select id="finding-radius">', self.page)
        self.assertIn('function findingRadiusFromZoom(', self.page)
        self.assertIn('function updateFindingZoomLabel(', self.page)
        self.assertIn('function liveFindingZoom(', self.page)
        self.assertIn('$("finding-zoom").addEventListener("input"',
                      self.page)
        self.assertIn('function skyViewPosition(', self.page)
        self.assertIn('function skyViewQueryUrl(', self.page)
        self.assertIn(
            'precessIcrsToTete(c.raDeg, c.dec, jdFromUtcMs(Date.now()))',
            self.page)
        self.assertIn('function findingRadiusText(', self.page)
        self.assertIn('skyview.gsfc.nasa.gov/current/cgi/runquery.pl',
                      self.page)
        self.assertIn('Sky image (SkyView)', self.page)
        self.assertIn('function updateFindingChart(', self.page)
        self.assertIn('id="finding-info"', self.page)
        self.assertIn("Hover the star chart for object info", self.page)
        self.assertIn('function updateFindingInfo(', self.page)
        self.assertIn('function clearFindingInfo(', self.page)
        self.assertIn('$("finding-chart").addEventListener("mousemove"',
                      self.page)
        self.assertIn('"mouseleave"', self.page)
        self.assertIn('justify-content: flex-start', self.page)
        self.assertIn('.finding-pane:first-child { flex: 0 0 auto; }',
                      self.page)
        self.assertIn('.finding-pane:last-child { flex: 1 1 auto;'
                      ' min-width: 0; }', self.page)
        self.assertIn('#finding-info { height: 8.8rem; overflow: hidden;',
                      self.page)
        self.assertIn('#finding-info table { border-collapse: collapse;',
                      self.page)
        self.assertIn('function findingOffsetText(', self.page)
        self.assertIn('function findingDetailTable(', self.page)
        self.assertNotIn('id="finding-img"', self.page)
        self.assertNotIn('alasky.cds.unistra.fr', self.page)
        self.assertNotIn('aladin.cds.unistra.fr', self.page)
        self.assertNotIn('function hpxHash(', self.page)
        self.assertNotIn('function updateFindingImage(', self.page)
        self.assertNotIn('function drawCutoutOverlay(', self.page)
        self.assertIn('findingSurvey: "Digitized Sky Survey"', self.page)
        self.assertIn('function simbadCooLookup(', self.page)
        self.assertIn('function drawFindingChart(', self.page)
        self.assertIn('function findingQueryUrl(', self.page)
        self.assertIn('"finding-panel"', self.page)
        self.assertIn("finding: false", self.page)
        self.assertIn("Finding chart", self.page)

    def test_script_injection_is_neutralized(self):
        """A hostile object name cannot open a second script tag."""
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
        """gmstMeeus agrees with astropy mean GMST to 0.01 deg."""
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
        """altitude matches astropy AltAz: pure geometry to 0.01 deg, end-to-end to 0.05 deg."""
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
        """horizonCoords matches astropy AltAz: alt to 0.01 deg, az to 0.02 deg."""
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
        """parallacticAngle matches the ENU vector definition to 1e-5 deg."""
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
            q_ref = _enu_parallactic_deg(H, lat, dec)
            got = self._js(f"parallacticAngle({ra_h}, {dec}, {jd}, {lat}, {lon})")
            dq = abs(got - q_ref) % 360.0
            self.assertLessEqual(
                min(dq, 360.0 - dq), 1e-5, msg=(round(jd, 4), round(lat, 2), round(dec, 2))
            )
            checked += 1
        self.assertGreaterEqual(checked, 30)

    def test_parallactic_angle_rate(self):
        """parallacticAngleRate matches the analytic dec = 0 derivative."""
        # For a declination-0 target the parallactic angle reduces to
        # q = atan2(cos(lat) sin H, sin(lat)); the 30-s central
        # difference should recover dq/dH times the hour-angle advance
        # per minute, where H is the hour angle in radians. The
        # advance is taken from the core's own lstDeg (sidereal rate)
        # so LST precision cannot pollute the comparison.
        rng = random.Random(7)
        checked = 0
        for _ in range(40):
            jd = rng.uniform(2451545.0, 2490000.0)
            lat = rng.uniform(-55.0, 55.0)
            lon = rng.uniform(-180.0, 180.0)
            H_deg = rng.uniform(-80.0, 80.0)
            lst = self._js(f"lstDeg({jd}, {lon})")
            ra_h = (lst - H_deg) / 15.0
            dH_min = (self._js(f"lstDeg({jd + 1 / 1440.0}, {lon})") - lst) % 360.0
            H = math.radians(H_deg)
            A = math.cos(math.radians(lat))
            B = math.sin(math.radians(lat))
            ref = dH_min * A * B * math.cos(H) / (
                (A * math.sin(H)) ** 2 + B * B
            )
            got = self._js(f"parallacticAngleRate({ra_h}, 0.0, {jd}, {lat}, {lon})")
            self.assertAlmostEqual(got, ref, delta=1e-4,
                                   msg=(round(jd, 4), round(lat, 2)))
            checked += 1
        self.assertGreaterEqual(checked, 30)

    def test_field_rotation(self):
        """fieldRotationDeg matches the unwrapped ENU parallactic angle to 0.05 deg."""
        # Reference: sum the unwrapped (+/-180 deg) increments of the
        # parallactic angle sampled on the same 15-minute grid (both
        # endpoints included) that the core uses, and take the absolute
        # value of the total. The hour angle is LST - RA, both from
        # astropy's apparent sidereal time and true equator of date, so
        # the reference is independent of the core's GMST and precession.
        # The core omits nutation, so 0.05 deg is allowed.
        ms0 = int(round((Time("2026-11-21 22:00:00").jd - 2440587.5) * 86400000))
        ms1 = int(round((Time("2026-11-22 08:00:00").jd - 2440587.5) * 86400000))
        loc = astro.location(LBT)
        prev = None
        expected = 0.0
        for ms in range(ms0, ms1 + 1, 900000):
            t = Time(ms / 1000.0, format="unix")
            lst = (
                t.sidereal_time("apparent", longitude=LBT.lon * u.deg)
                .to_value(u.hourangle)
                * 15.0
            )
            tete = VEGA.transform_to(TETE(obstime=t))
            H = math.radians(
                (lst - tete.ra.to_value(u.hourangle) * 15.0) % 360.0
            )
            q = _enu_parallactic_deg(H, LBT.lat, tete.dec.deg)
            if prev is not None:
                expected += (q - prev + 180.0) % 360.0 - 180.0
            prev = q
        expected = abs(expected)
        got = self._js(
            f"fieldRotationDeg({{ra: '18:36:56.30142', dec: '+38:47:01.4506'}}, "
            f"{ms0}, {ms1}, {LBT.lat}, {LBT.lon})"
        )
        self.assertAlmostEqual(got, expected, delta=0.05)

    def test_annual_calendar(self):
        """annualCalendarMonths matches astropy: max alt to 0.1 deg,
        usable hours to 2 h, best airmass to 0.01 and its time to 3 h,
        and usable nights to 1 day (usable = X <= 1.7 while the sun is
        at or below -12 deg)."""
        # Reference: sample the refraction-free astropy altitude on the
        # same UTC hour grid the core uses (tz 'UTC'), taking the
        # per-month max altitude and counting the hours with X <=
        # threshold during astronomical night (get_sun altitude <=
        # -12 deg), where X is the core's own kastyAir so only the
        # geometry is cross-checked. The one-hour sampling of the
        # transit phase sets the 0.1 deg max-altitude tolerance.
        # annualCalendarMonths calls tzParts, an Intl-based helper from
        # the UI layer (outside the JS core), so define a UTC-only
        # stand-in in the interpreter: the grid's tz is UTC, and the
        # Date UTC parts agree with Intl's UTC parts.
        self._js(
            "function tzParts(tz, ms) {\n"
            '  if (tz !== "UTC") throw new Error("test tzParts is UTC only");\n'
            "  var d = new Date(ms);\n"
            "  return { year: d.getUTCFullYear(), month: d.getUTCMonth() + 1,\n"
            "           day: d.getUTCDate(), hour: d.getUTCHours(),\n"
            "           minute: d.getUTCMinutes(), second: d.getUTCSeconds() };\n"
            "}"
        )
        ms0 = int(round((Time("2026-03-01 00:00:00").jd - 2440587.5) * 86400000))
        got = json.loads(
            self._js(
                f"JSON.stringify(annualCalendarMonths(function (ms) {{ return "
                f"{{raDeg: {VEGA.ra.deg}, dec: {VEGA.dec.deg}}}; }}, "
                f"'UTC', {LBT.lat}, {LBT.lon}, {ms0}, 1.7))"
            )
        )
        self.assertEqual(len(got), 12)
        self.assertEqual(got[0]["label"], "Mar 2026")
        self.assertEqual(got[11]["label"], "Feb 2027")
        loc = astro.location(LBT)
        for b in range(12):
            m = got[b]
            if m["hours"] == 0:
                self.assertIsNone(m["firstDay"])
                self.assertIsNone(m["lastDay"])
                self.assertEqual(m["nights"], 0)
            else:
                self.assertGreaterEqual(m["firstDay"], 1)
                self.assertLessEqual(m["firstDay"], m["lastDay"])
                self.assertGreaterEqual(m["lastDay"] - m["firstDay"] + 1,
                                        math.ceil(m["hours"] / 24.0))
                self.assertGreaterEqual(m["nights"], 1)
                self.assertLessEqual(m["nights"], m["lastDay"] - m["firstDay"] + 1)
                self.assertLessEqual(m["nights"], m["hours"])
            if m["maxAlt"] is None:
                self.assertIsNone(m["bestMs"])
            else:
                self.assertIsNotNone(m["bestMs"])
                self.assertGreaterEqual(m["bestMs"], ms0)
                self.assertLess(m["bestMs"], ms0 + 8784 * 3600000)
        # Month 0 in full: max altitude, usable-hour count, the
        # first/last usable day (UTC days, since the grid's tz is
        # UTC), the lowest-airmass sample and its time, and the count
        # of distinct usable days.
        ref_alt = -math.inf
        ref_hours = 0
        ref_first = ref_last = None
        ref_best_x = math.inf
        ref_best_ms = None
        ref_nights = set()
        for k in range(744):
            ms = ms0 + k * 3600000
            t = Time(ms / 1000.0, format="unix")
            alt = VEGA.transform_to(AltAz(obstime=t, location=loc)).alt.deg
            if alt > ref_alt:
                ref_alt = alt
            x = self._js(f"kastyAir({alt})")
            if alt > 0:
                if x < ref_best_x:
                    ref_best_x = x
                    ref_best_ms = ms
                dark = get_sun(t).transform_to(
                    AltAz(obstime=t, location=loc)).alt.deg <= -12
                if x <= 1.7 and dark:
                    ref_hours += 1
                    day = int(t.isot[8:10])
                    ref_nights.add(day)
                    if ref_first is None:
                        ref_first = day
                    ref_last = day
        m0 = got[0]
        self.assertAlmostEqual(m0["maxAlt"], ref_alt, delta=0.1)
        self.assertAlmostEqual(m0["hours"], ref_hours, delta=2.0)
        self.assertAlmostEqual(m0["firstDay"], ref_first, delta=1.0)
        self.assertAlmostEqual(m0["lastDay"], ref_last, delta=1.0)
        self.assertAlmostEqual(m0["minAir"], ref_best_x, delta=0.01)
        self.assertLessEqual(abs(m0["bestMs"] - ref_best_ms), 3 * 3600000)
        self.assertAlmostEqual(m0["nights"], len(ref_nights), delta=1.0)

    def test_precess(self):
        """precessIcrsToTete matches astropy TETE at nutation level (~0.02 deg)."""
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
        """airmass agrees with astro.airmass_from_altitude to 1e-12."""
        for alt in (0.5, 10.0, 30.0, 45.0, 60.0, 89.9, 90.0):
            got = self._js(f"airmass({alt})")
            expected = float(astro.airmass_from_altitude(alt)[0])
            self.assertTrue(math.isclose(got, expected, rel_tol=1e-12), (alt, got, expected))

    def test_horizon_and_inverse(self):
        """horizonX and altFromAirmass agree with the Python functions to 1e-6."""
        self.assertAlmostEqual(self._js("horizonX()"), astro.horizon_airmass(), delta=1e-9)
        for x in (1.0, 1.2, 2.0, 5.0, 10.0, 30.0, 37.9):
            got = self._js(f"altFromAirmass({x})")
            expected = float(astro.altitude_from_airmass(x)[0])
            self.assertAlmostEqual(got, expected, delta=1e-6)

    def test_sun_ra_dec(self):
        """sunRaDec matches astropy get_sun to 0.5 deg RA, 0.15 deg dec."""
        for iso in ("2026-11-20 21:00:00", "2027-06-21 03:00:00"):
            t = Time(iso)
            sun = get_sun(t)
            got = json.loads(self._js(f"JSON.stringify(sunRaDec({float(t.jd)}))"))
            dra = abs(got["raH"] * 15.0 - sun.ra.deg) % 360.0
            self.assertLessEqual(min(dra, 360.0 - dra), 0.5)
            self.assertLessEqual(abs(got["dec"] - sun.dec.deg), 0.15)

    def test_sun_shading(self):
        """sunShading boundaries match astro.sun_shading_intervals to 120 s."""
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
        """parseRa agrees with astro.parse_ra on good and bad strings."""
        for text in RA_OK:
            got = self._js(f"parseRa({json.dumps(text)})")
            self.assertAlmostEqual(got, astro.parse_ra(text), delta=1e-9)
        for text in RA_BAD:
            with self.assertRaises(ValueError):
                astro.parse_ra(text)
            with self.assertRaises(Exception):
                self._js(f"parseRa({json.dumps(text)})")

    def test_parse_dec(self):
        """parseDec agrees with astro.parse_dec on good and bad strings."""
        for text in DEC_OK:
            got = self._js(f"parseDec({json.dumps(text)})")
            self.assertAlmostEqual(got, astro.parse_dec(text), delta=1e-9)
        for text in DEC_BAD:
            with self.assertRaises(ValueError):
                astro.parse_dec(text)
            with self.assertRaises(Exception):
                self._js(f"parseDec({json.dumps(text)})")

    def test_simbad_parse(self):
        """simbadParseRecord agrees with lookup._parse_ascii_record on a real record."""
        # The page's parser must agree with the desktop app's on a real record.
        fixture = (Path(__file__).parent / "fixtures" / "simbad_vega.txt").read_text(encoding="utf-8")
        expected = lookup._parse_ascii_record(fixture)
        got = self._js(f"simbadParseRecord({json.dumps(fixture)})")
        self.assertIsNotNone(got, "JS parser found no coordinates")
        self.assertEqual(got["name"], expected["name"])
        self.assertAlmostEqual(got["raDeg"], expected["ra"], delta=1e-9)
        self.assertAlmostEqual(got["dec"], expected["dec"], delta=1e-9)
        self.assertAlmostEqual(got["mag"], float(expected["mag"]))
        self.assertEqual(got["spec"], expected["spec"])
        self.assertAlmostEqual(got["plx"], float(expected["plx"]))
        no_coords = "C.D.S.  -  SIMBAD4\n\nSome object\n----\nno coordinates here"
        self.assertIsNone(self._js(f"simbadParseRecord({json.dumps(no_coords)})"))
        minimal = ("C.D.S.  -  SIMBAD4\n\nFaint star\n--------\n"
                   "Coordinates(ICRS,ep=J2000,eq=2000): 01 02 03.00000  +10 11 12.0000\n")
        got2 = self._js(f"simbadParseRecord({json.dumps(minimal)})")
        self.assertIsNone(got2["mag"])
        self.assertIsNone(got2["spec"])
        self.assertIsNone(got2["plx"])

    def test_simbad_coo_parse(self):
        """simbadCooParse extracts the full default table column set."""
        table = (
            "C.D.S.  -  SIMBAD4 rel 1.8  -  2026.11.24CET17:45:23\n\n"
            + "coord 18 36 56.30 38 47 01.45 (ICRS, J2000, 2000.0), "
              "radius: 10 arcmin\n"
            + "----------------------------------------------------------------------\n\n"
            + "Number of objects : 2\n\n"
            + " # |dist(asec)|            identifier             |typ|"
              "       coord1 (ICRS,J2000/2000)        | Mag U  |  Mag B"
              "  |  Mag V  |  Mag R  | Mag I  |  spec. type   |#bib"
              "|#not\n"
            + "--|----------|-----------------------------------|---|"
              "---------------------------------------|--------|------"
              "---|---------|---------|--------|---------------|----"
              "|----\n"
            + "1 |      0.46|* alf Lyr                          |dS*|"
              "18 36 56.33635 +38 47 01.2802          |0.03    |0.03"
              "     |0.03     |0.07     |0.10    |A0V            |2817"
              "|   1\n"
            + "2 |     91.90|2MASS J18370125+3848126            |*  "
              "|18 37 01.2678131808 +38 48 12.673392336|     ~  |    "
              "   ~  |9.5      |     ~  |     ~  |~              |   0"
              "|   0\n"
        )
        got = self._js(f"simbadCooParse({json.dumps(table)})")
        self.assertEqual(got["count"], 2)
        self.assertEqual(len(got["rows"]), 2)
        first = got["rows"][0]
        self.assertEqual(first["name"], "* alf Lyr")
        self.assertAlmostEqual(
            first["raDeg"], 18 * 15 + 36 / 4 + 56.33635 / 240,
            delta=1e-9)
        self.assertAlmostEqual(
            first["dec"], 38 + 47 / 60 + 1.2802 / 3600, delta=1e-9)
        self.assertAlmostEqual(first["u"], 0.03)
        self.assertAlmostEqual(first["b"], 0.03)
        self.assertAlmostEqual(first["v"], 0.03)
        self.assertAlmostEqual(first["r"], 0.07)
        self.assertAlmostEqual(first["i"], 0.10)
        self.assertEqual(first["spt"], "A0V")
        self.assertEqual(first["typ"], "dS*")
        self.assertAlmostEqual(first["distArcsec"], 0.46)
        self.assertEqual(first["bib"], 2817)
        self.assertEqual(first["notn"], 1)
        second = got["rows"][1]
        self.assertIsNone(second["u"])
        self.assertAlmostEqual(second["v"], 9.5)
        self.assertIsNone(second["spt"])
        self.assertEqual(second["bib"], 0)
        self.assertEqual(
            self._js("simbadCooParse('no table here')")["rows"], [])
        self.assertIsNone(
            self._js("simbadCooParse('no table here')")["count"])

    def test_finding_zoom_radius(self):
        """findingRadiusFromZoom log mapping: endpoints, default, monotonic."""
        self.assertEqual(self._js("findingRadiusFromZoom(0)"), 0.5)
        self.assertEqual(self._js("findingRadiusFromZoom(640)"), 30)
        self.assertEqual(self._js("findingRadiusFromZoom(1000)"), 300)
        self.assertEqual(self._js("findingRadiusFromZoom(250)"), 2.5)
        self.assertEqual(self._js("findingRadiusFromZoom(500)"), 12)
        self.assertEqual(self._js("findingRadiusFromZoom(750)"), 61)
        vals = [self._js(f"findingRadiusFromZoom({t})")
                for t in (0, 250, 500, 640, 750, 1000)]
        self.assertEqual(vals, sorted(vals),
                         f"zoom mapping not monotonic: {vals}")
        self.assertTrue(all(v > 0 for v in vals))

    def test_finding_offset_and_detail_table(self):
        """findingOffsetText units and findingDetailTable row selection."""
        self.assertEqual(self._js("findingOffsetText(0.75)"), "0.8\u2033")
        self.assertEqual(self._js("findingOffsetText(59.94)"),
                         "59.9\u2033")
        self.assertEqual(self._js("findingOffsetText(91.9)"), "1.53\u2032")
        self.assertIsNone(self._js("findingOffsetText(null)"))
        row = {"raDeg": 279.186354, "dec": 38.783639,
               "u": 0.03, "b": 0.03, "v": 0.03, "r": 0.07, "i": 0.10,
               "spt": "A0V", "typ": "dS*", "distArcsec": 0.46,
               "bib": 2817, "notn": 1}
        html = self._js(f"findingDetailTable({json.dumps(row)})")
        self.assertIn("<th>RA/Dec</th>", html)
        self.assertIn("<th>Mags</th><td>U 0.03  B 0.03  V 0.03  "
                      "R 0.07  I 0.10</td>", html)
        self.assertIn("<th>SpT</th><td>A0V</td>", html)
        self.assertIn("<th>Type</th><td>dS*</td>", html)
        self.assertIn("<th>From target</th><td>0.5\u2033</td>", html)
        self.assertIn("<th>Bib/Notes</th><td>2817 / 1</td>", html)
        sparse = {"raDeg": 10.0, "dec": -20.0}
        html2 = self._js(f"findingDetailTable({json.dumps(sparse)})")
        self.assertIn("<th>RA/Dec</th>", html2)
        self.assertNotIn("<th>Mags</th>", html2)
        self.assertNotIn("<th>SpT</th>", html2)
        self.assertNotIn("<th>From target</th>", html2)


    def test_share_fragment_round_trip(self):
        """encode/decodeShareFragment round-trip both ways; malformed fragments give null."""
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
        """htmlEscape escapes &, <, >, double and single quotes."""
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
        """trueRuns recovers run boundaries from a boolean sample."""
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
        """transitTimes matches a 60 s LST scan to 1 s."""
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
        """twilightEvents matches threshold crossings to 30 s."""
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

    def test_sun_events(self):
        """sunEvents boundaries match astro.sun_shading_intervals at all four thresholds, with the right direction."""
        start = Time("2026-11-20 00:00:00")
        end = Time("2026-11-21 00:00:00")
        intervals = astro.sun_shading_intervals(LBT, start, end)
        kinds = [k for (_a, _b, k) in intervals]
        bounds = [float(start.to_value("mjd"))] + [float(b) for (_a, b, _k) in intervals]
        start_ms = float(start.to_value("unix")) * 1000.0
        end_ms = float(end.to_value("unix")) * 1000.0
        got = json.loads(
            self._js(f"JSON.stringify(sunEvents({start_ms}, {end_ms}, {LBT.lat}, {LBT.lon}))")
        )
        side = {0: {"day"}, -6: {"day", "civil"}, -12: {"day", "civil", "nautical"},
                -18: {"day", "civil", "nautical", "astro"}}
        for thr in (0, -6, -12, -18):
            ref = []
            for i in range(len(kinds) - 1):
                if (kinds[i] in side[thr]) != (kinds[i + 1] in side[thr]):
                    ref.append((bounds[i + 1] - 40587.0) * 86400000.0)
            evs = [e for e in got if e["alt"] == thr]
            self.assertEqual(len(evs), len(ref), thr)
            for g, r in zip(evs, ref):
                self.assertAlmostEqual(g["ms"], r, delta=120000.0)  # 120 s
        zeros = [e for e in got if e["alt"] == 0]
        # span starts 17:00 local: first crossing is sunset, then sunrise
        self.assertEqual([e["dir"] for e in zeros], [-1, 1])

    def test_moon_position(self):
        """moonRaDec is within 0.03 deg of astropy's full-ELP Moon."""
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
        """moonPhase separation and illumination match astropy (0.05 deg, 0.005)."""
        # Reference in TET (equator of date), the frame the JS core works in.
        t = Time("2026-11-20 22:00:00")
        moon = get_body("moon", t, location=EarthLocation(x=0, y=0, z=0, unit="m")).transform_to(TETE(obstime=t))
        sun = get_sun(t).transform_to(TETE(obstime=t))
        sep_ref = moon.separation(sun).deg
        got = json.loads(self._js(f"JSON.stringify(moonPhase({float(t.jd)}))"))
        self.assertLessEqual(abs(got["sepDeg"] - sep_ref), 0.05)
        self.assertLessEqual(abs(got["illum"] - (1 - math.cos(math.radians(sep_ref))) / 2), 0.005)

    def test_angular_sep(self):
        """angularSep: quadrants, identity, antipodes, and the pole."""
        self.assertAlmostEqual(self._js("angularSep(0, 0, 6, 0)"), 90.0, delta=1e-9)
        self.assertAlmostEqual(self._js("angularSep(12, 45, 12, 45)"), 0.0, delta=1e-9)
        self.assertAlmostEqual(self._js("angularSep(0, 0, 12, 0)"), 180.0, delta=1e-9)
        # Both targets at the north celestial pole: RA is undefined there, sep 0.
        self.assertAlmostEqual(self._js("angularSep(0, 90, 12, 90)"), 0.0, delta=1e-9)

    def test_obj_coords_static(self):
        """objCoords returns the catalog coordinates for a static target."""
        got = self._js("objCoords({ra: '18:36:56.30142', dec: '+38:47:01.4506'}, 1000000)")
        self.assertAlmostEqual(got["raDeg"], 279.2345917, delta=0.001)
        self.assertAlmostEqual(got["dec"], 38.7837361, delta=0.001)

    def test_obj_coords_interpolates_with_wrap(self):
        """objCoords interpolates a moving table and wraps RA across 0/360."""
        ep = [[0, 359.0, 47.0], [3600000, 359.5, 47.01], [7200000, 0.0, 47.02], [10800000, 1.0, 47.03]]
        mid = self._js("objCoords({ephemeris: %s}, 5400000)" % ep)
        self.assertAlmostEqual(mid["raDeg"], 359.75, delta=0.001)
        self.assertAlmostEqual(mid["dec"], 47.015, delta=0.001)
        after = self._js("objCoords({ephemeris: %s}, 8100000)" % ep)
        self.assertAlmostEqual(after["raDeg"], 0.25, delta=0.001)
        before = self._js("objCoords({ephemeris: %s}, -1000)" % ep)
        self.assertAlmostEqual(before["raDeg"], 359.0, delta=1e-9)

    def test_horizons_parse(self):
        """horizonsParse reads date/radec_j2000 pairs and rejects empty replies."""
        text = (
            "header line\n"
            "       date = 2026-02-02 05:00:00\n"
            "     radec_j2000 =   134.92242957  22.11225783\n"
            "       date = 2026-02-02 06:00:00\n"
            "     radec_j2000 =   134.97242957  22.11235783\n"
            "ERROR: trailing note\n"
        )
        rows = json.loads(self._js("JSON.stringify(horizonsParse(%s))" % json.dumps(text)))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0][1], 134.92242957)
        self.assertAlmostEqual(rows[1][2], 22.11235783)
        self.assertEqual(rows[1][0], 1770012000000)
        with self.assertRaises(Exception):
            self._js("horizonsParse('ERROR: no rows here')")


    def test_snr_moon_sep(self):
        """snrMoonSep agrees with sn.snr_moon_sep on a separation grid and the anchors."""
        for sep in (0.0, 5.0, 10.0, 25.0, 45.0, 90.0, 120.0, -30.0):
            self.assertAlmostEqual(
                self._js(f"snrMoonSep({sep})"), sn.snr_moon_sep(sep), places=9, msg=sep)
        self.assertAlmostEqual(self._js("snrMoonSep(0)"), 4.5, places=2)
        self.assertAlmostEqual(self._js("snrMoonSep(90)"), 2.15, places=2)

    def test_snr_moon_darkening(self):
        """snrMoonDarkening scales the full-Moon curve by phase and altitude."""
        for sep, illum, alt in ((0.0, 1.0, 45.0), (30.0, 0.5, 20.0),
                                (90.0, 0.25, 80.0), (10.0, 1.0, -5.0),
                                (45.0, 0.0, 60.0)):
            self.assertAlmostEqual(
                self._js(f"snrMoonDarkening({sep}, {illum}, {alt})"),
                sn.snr_moon_darkening(sep, illum, alt), places=9,
                msg=(sep, illum, alt))
        self.assertEqual(self._js("snrMoonDarkening(10, 1, -5)"), 0.0)

    def test_snr_rates(self):
        """Source and sky rates and the PSF pixel count agree with sn."""
        for aperture in (0.5, 2.0, 8.2):
            self.assertAlmostEqual(
                self._js(f"snrArea({aperture})"), sn.snr_area(aperture), places=6)
        for fwhm, scale in ((1.0, 0.2), (0.8, 0.063), (2.5, 0.5)):
            self.assertAlmostEqual(
                self._js(f"snrPsfPixels({fwhm}, {scale})"),
                sn.snr_psf_pixels(fwhm, scale), places=6)
        args = dict(area_cm2=31000.0, mag=3.5, airmass=1.8, k_ext=0.1, eff=0.55)
        self.assertAlmostEqual(
            self._js("snrSourceRate(31000, 3.5, 1.8, 0.1, 0.55)"),
            sn.snr_source_rate(**args), places=4)
        self.assertAlmostEqual(
            self._js("snrSkyRate(31000, 21.3, 1.8, 0.1, 0.55, 0.2)"),
            sn.snr_sky_rate(31000.0, 21.3, 1.8, 0.1, 0.55, 0.2), places=4)
        self.assertAlmostEqual(
            self._js("snrSkyMag(21.9, 1.25)"),
            sn.snr_sky_mag(21.9, 1.25), places=9)

    def test_snr_value_and_solve_time(self):
        """snrValue and snrSolveTime agree with sn, and the solve inverts the value."""
        rng = random.Random(11)
        for _ in range(20):
            src_rate = 10.0 ** rng.uniform(1.0, 4.0)
            sky = 10.0 ** rng.uniform(-1.0, 1.0)
            n_pix = 10.0 ** rng.uniform(0.0, 2.0)
            dark = 10.0 ** rng.uniform(-3.0, -0.5)
            rn = 10.0 ** rng.uniform(0.0, 1.5)
            t = 10.0 ** rng.uniform(0.0, 3.0)
            self.assertAlmostEqual(
                self._js(f"snrValue({src_rate}, {sky}, {n_pix}, {dark}, {rn}, {t})"),
                sn.snr_value(src_rate, sky, n_pix, dark, rn, t), places=4)
            T = 10.0 ** rng.uniform(0.5, 2.0)
            self.assertAlmostEqual(
                self._js(f"snrSolveTime({src_rate}, {sky}, {n_pix}, {dark}, {rn}, {T})"),
                sn.snr_solve_time(src_rate, sky, n_pix, dark, rn, T), places=4)
            # the solved time must invert back to the target S/N
            tSolved = self._js(
                f"snrSolveTime({src_rate}, {sky}, {n_pix}, {dark}, {rn}, {T})")
            self.assertAlmostEqual(
                self._js(
                    f"snrValue({src_rate}, {sky}, {n_pix}, {dark}, {rn}, {tSolved})"),
                T, places=4)

    def test_snr_pipeline(self):
        """The full V-band pipeline (area, moon, sky, rates, solve) agrees with sn end to end."""
        aperture, fwhm, scale = 2.0, 0.8, 0.2
        mag, x, k, eff = 4.0, 1.5, 0.12, 0.5
        sep, illum, m_alt = 55.0, 0.62, 35.0
        area = sn.snr_area(aperture)
        dark_mag = sn.snr_moon_darkening(sep, illum, m_alt)
        sky_mag = sn.snr_sky_mag(sn.SNR_SKY_DARK, dark_mag)
        src = sn.snr_source_rate(area, mag, x, k, eff)
        sky = sn.snr_sky_rate(area, sky_mag, x, k, eff, scale)
        n_pix = sn.snr_psf_pixels(fwhm, scale)
        T = 15.0
        want = sn.snr_solve_time(src, sky, n_pix, 0.1, 5.0, T)
        js = (
            f"snrSolveTime("
            f"snrSourceRate(snrArea({aperture}), {mag}, {x}, {k}, {eff}), "
            f"snrSkyRate(snrArea({aperture}), "
            f"snrSkyMag({sn.SNR_SKY_DARK}, snrMoonDarkening({sep}, {illum}, {m_alt})), "
            f"{x}, {k}, {eff}, {scale}), "
            f"snrPsfPixels({fwhm}, {scale}), 0.1, 5.0, {T})"
        )
        self.assertAlmostEqual(self._js(js), want, places=4)


if __name__ == "__main__":
    unittest.main()
