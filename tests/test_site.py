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
import urllib.parse
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, unquote
from pathlib import Path

import numpy as np

import html as html_mod
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, TETE, get_body, get_sun
from astropy.io import fits
from astropy.time import Time

from airmass_gui import astro, chart, lookup
from airmass_gui.observatories import PRESETS, RENAME_MAP, to_dict
from airmass_gui.site import THEMES, _blend_hex, build_site
from airmass_gui.workspace import load, new_workspace, save
from test_finding_details_live import VEGA_FIXTURE

try:
    import dukpy
except ImportError:  # pragma: no cover
    dukpy = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

LBT = PRESETS["Mount Graham (LBT)"]
VEGA = SkyCoord("18h36m56.30142s", "+38d47m01.4506s", frame="icrs")

ROOT = Path(__file__).resolve().parent.parent

CORE_BEGIN = "/*JS-CORE-BEGIN*/"
CORE_END = "/*JS-CORE-END*/"

CORE_FUNCTIONS = [
    "wrap360", "d2r", "r2d", "jdFromUtcMs", "gmstMeeus", "lstDeg", "altitude",
    "horizonCoords", "parallacticAngle", "parallacticAngleRate", "precessIcrsToTete", "kastyAir", "airmass", "horizonX", "altFromAirmass",
    "sunRaDec", "sunAltitude", "classifySunAlt", "sunShading", "parseRa",
    "parseDec", "simbadParseRecord", "parallaxDistance", "formatDistance", "pmText",
    "encodeShareFragment", "decodeShareFragment", "htmlEscape",
    "trueRuns", "transitTimes", "twilightEvents", "sunEvents",
    "moonRaDec", "moonTopocentric", "moonRiseSet", "moonPhase", "angularSep", "targetInfoText",
    "objCoords", "fieldRotationDeg",
    "gnomonicProject", "gnomonicUnproject", "formatRaDecHms",
    "simbadCooParse", "findingOffsetText", "findingMagList",
    "findingBasicTable", "findingDetailTable",
    "findingStarSize", "findingQueryRadius",
    "findingRadiusText", "findingRadiusFromZoom",
    "findingScaleArcmin", "hips2fitsImageUrl",
    "sunEclLonDeg", "moonElongationDeg", "moonPhaseInfo",
    "weatherSymbolKind", "wmoText", "wmoIcon",
    "blendHex",
]

RA_OK = ["18:36:56.30142", "18.616h", "279.2346", "0", "23:59:59.9", "7.5",
         "18 36 56.3", "5 55 40.8"]
RA_BAD = ["", "abc", "1:70:0", "2:0:70", "1 2 3 4", "12.5.7"]
DEC_OK = ["+38:47:01.4506", "-38:47:01", "45\u00b0 30\u2032 15\u2033", "-89.99", "0",
          "12d 30m 45s", "-00 17 57", "+38 47 01.28"]
DEC_BAD = ["", "abc", "91", "-95", "30:70", "1 2 3 4"]


def _extract_core(page: str) -> str:
    """Return the JS text between the JS-CORE markers."""
    begin = page.index(CORE_BEGIN) + len(CORE_BEGIN)
    return page[begin : page.index(CORE_END)]


def _extract_fn(page: str, name: str) -> str:
    """Return the text of the top-level function named ``name``.

    Braces inside string literals would break the count, so this is only
    for functions with no braces in their strings.
    """
    start = page.index("function " + name)
    depth = 0
    for i in range(page.index("{", start), len(page)):
        if page[i] == "{":
            depth += 1
        elif page[i] == "}":
            depth -= 1
            if depth == 0:
                return page[start : i + 1]
    raise LookupError(name)


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

    def test_no_horizons_feature_in_page(self):
        """The page carries no Horizons fetch feature: no button, no API
        URL, no parser name, in any letter case."""
        lowered = self.page.lower()
        self.assertNotIn("horizons", lowered)
        self.assertNotIn("jpl.nasa.gov", lowered)

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
                    "panel-targets", "panel-workspace",
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

    def test_target_simbad_refresh(self):
        """Each target row can refresh its stored Simbad record."""
        self.assertIn('class="sb" title="Refresh this target', self.page)
        self.assertIn("Refreshing \\u201c", self.page)
        self.assertIn('30px 1fr 1fr 1fr 44px auto auto 22px', self.page)
        self.assertIn('30px 1fr 1fr 1fr 44px 1fr auto auto 22px', self.page)

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
        # finding chart's sky image is likewise a runtime opt-in: the CDS
        # HiPS2FITS cutout is requested only when the Finding chart
        # checkbox is on, and the page ships without an img src.)
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

    def test_embedded_preset_renames(self):
        """PRESET_RENAMES embeds the old -> new preset rename map."""
        self.assertEqual(_embedded_json(self.page, "PRESET_RENAMES"), RENAME_MAP)

    def test_load_state_migrates_renames(self):
        """loadState runs the preset rename migration before normalizing."""
        self.assertIn("migratePresetRename(s);", self.page)

    def test_embedded_shade_colors(self):
        """SHADE maps each level to a [hex, alpha] blending recipe."""
        shade = _embedded_json(self.page, "SHADE")
        self.assertEqual(set(shade), set(chart.SHADE))
        for kind, (color, alpha) in chart.SHADE.items():
            self.assertEqual(shade[kind], [color, alpha])
            self.assertRegex(color, r"^#[0-9a-f]{6}$")

    def test_embedded_themes(self):
        """THEMES embeds all three color themes (hex colors, shade scale)."""
        themes = _embedded_json(self.page, "THEMES")
        self.assertEqual(themes, THEMES)
        color_keys = ("bg", "bg_alt", "fg", "fg_dim", "accent", "border",
                      "error", "button", "button_hover", "axes_bg",
                      "highlight")
        for name, entry in themes.items():
            for key in color_keys:
                self.assertRegex(entry[key], r"^#[0-9a-f]{6}$", (name, key))
            self.assertIsInstance(entry["shade"], float)

    def test_theme_selector(self):
        """The header offers the three themes with a change handler."""
        self.assertIn('<select id="theme-sel">', self.page)
        for value, label in (("dark", "Dark"), ("light", "Light"),
                             ("red", "Red (night)")):
            self.assertIn(
                '<option value="%s">%s</option>' % (value, label), self.page)
        self.assertIn('$("theme-sel").addEventListener("change"', self.page)
        self.assertIn('html[data-theme="red"] body { filter:', self.page)
        self.assertIn('localStorage.getItem("airmass-gui-theme")', self.page)
        self.assertIn('localStorage.setItem("airmass-gui-theme"', self.page)

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

    def test_finding_chart_ui(self):
        """The Display panel offers a finding chart: checkbox, target,
        zoom slider, a star-chart canvas with the hover's
        nearest-object line and basic-field table below it, the inline
        CDS DSS2 color sky image with its new-tab link and the
        hover's record-expansion table below it, the Simbad coo
        lookup, and a resizable panel."""
        self.assertIn(
            '<input id="finding-chk" type="checkbox">', self.page)
        self.assertIn('<div class="panel" id="finding-panel"', self.page)
        self.assertIn('<select id="finding-target">', self.page)
        self.assertIn(
            '<input type="range" id="finding-zoom" min="0" max="1000"',
            self.page)
        self.assertIn('id="finding-zoom-label"', self.page)
        self.assertIn('<canvas id="finding-chart"></canvas>', self.page)
        self.assertIn('<img id="finding-img" alt="DSS2 color sky image">',
                      self.page)
        self.assertIn('id="finding-img-status"', self.page)
        self.assertIn('<a id="finding-cdslink" href="#" target="_blank"',
                      self.page)
        self.assertIn('id="finding-panes"', self.page)
        self.assertIn('flex: 1 1 180px', self.page)
        self.assertNotIn('<select id="finding-survey">', self.page)
        self.assertNotIn('<select id="finding-radius">', self.page)
        self.assertIn('function findingRadiusFromZoom(', self.page)
        self.assertIn('function updateFindingZoomLabel(', self.page)
        self.assertIn('function liveFindingZoom(', self.page)
        self.assertIn('$("finding-zoom").addEventListener("input"',
                      self.page)
        self.assertIn('function hips2fitsImageUrl(', self.page)
        self.assertIn('function findingScaleArcmin(', self.page)
        self.assertIn('var FINDING_FOV_MAX_DEG = 0.25', self.page)
        self.assertNotIn('finding-imgwrap', self.page)
        self.assertNotIn('findingGridStep', self.page)
        self.assertNotIn('ctx.arc(cx, cy, R, 0, 2 * Math.PI);',
                      self.page)
        self.assertNotIn('(R + 1) * (R + 1)', self.page)
        self.assertIn('Math.abs(pt.x - cx)', self.page)
        self.assertIn('ctx.textAlign = "right"', self.page)
        self.assertIn('<span id="finding-zoom-label">7.5&prime;</span>',
                      self.page)
        self.assertIn('<div id="finding-scale"', self.page)
        self.assertIn('id="finding-scale-bar"', self.page)
        self.assertIn('id="finding-scale-label"', self.page)
        self.assertIn('hips2fitsImageUrl(c.raDeg, c.dec, radiusArcmin / 30.0)',
                      self.page)
        self.assertIn('img.src = imgUrl', self.page)
        self.assertIn('$("finding-img").addEventListener("error"',
                      self.page)
        self.assertIn('function findingRadiusText(', self.page)
        self.assertIn('function updateFindingChart(', self.page)
        self.assertIn('id="finding-nearest"', self.page)
        self.assertIn('id="finding-details"', self.page)
        self.assertNotIn('id="finding-info"', self.page)
        self.assertNotIn('fi-wrap', self.page)
        self.assertIn(
            'id="finding-nearest"><span class="finding-dim">Hover the'
            ' star chart or the sky image for object info</span></div>',
            self.page)
        self.assertIn('function updateFindingInfo(', self.page)
        self.assertIn('function clearFindingInfo(', self.page)
        self.assertIn('$("finding-chart").addEventListener("mousemove"',
                      self.page)
        self.assertIn('$("finding-img").addEventListener("mousemove"',
                      self.page)
        self.assertIn('$("finding-img").addEventListener("mouseleave"',
                      self.page)
        self.assertIn('function updateFindingInfoAt(', self.page)
        self.assertIn('findingQueryRadius(radiusArcmin, size)', self.page)
        self.assertIn('simbadCooLookup(c.raDeg, c.dec, queryRadius)',
                      self.page)
        self.assertIn('"mouseleave"', self.page)
        self.assertIn('justify-content: flex-start', self.page)
        self.assertIn('.finding-pane { position: relative; display: flex;\n'
                      '                flex-direction: column; min-width: 0;'
                      ' flex: 0 0 auto; }', self.page)
        self.assertNotIn('.finding-pane:first-child', self.page)
        self.assertNotIn('.finding-pane:last-child', self.page)
        self.assertIn('<div class="finding-pane" id="finding-chart-pane">',
                      self.page)
        self.assertIn('<div class="finding-pane" id="finding-img-pane">',
                      self.page)
        self.assertIn('#finding-img { display: block; }', self.page)
        self.assertIn('#finding-details { flex: 1 1 auto; min-height: 8.8rem;'
                      ' overflow-y: auto;', self.page)
        self.assertIn('#finding-details td { overflow-wrap: break-word; }',
                      self.page)
        self.assertNotIn('INFO_RESERVE_PX', self.page)
        self.assertIn('#finding-details table { border-collapse: collapse;',
                      self.page)
        self.assertIn('#finding-nearest table { border-collapse: collapse;',
                      self.page)
        # The collapsed line (cursor position, nearest name and offset)
        # sits in the left pane below the star chart; only the detail
        # table sits in the right pane below the sky image.
        chart = self.page.index('id="finding-chart"')
        nearest = self.page.index('id="finding-nearest"')
        img = self.page.index('id="finding-img"')
        details = self.page.index('id="finding-details"')
        self.assertLess(chart, nearest)
        self.assertLess(nearest, img)
        self.assertLess(img, details)
        # Both panes lead with their caption line, so the two
        # squares start on the same row and sit evenly side by side.
        star_caption = self.page.index('>Star chart<')
        sky_caption = self.page.index('Sky image (DSS2 color, CDS)')
        self.assertLess(star_caption, chart)
        self.assertLess(sky_caption, img)
        self.assertIn('function findingOffsetText(', self.page)
        self.assertIn('function findingMagList(', self.page)
        self.assertIn('function findingBasicTable(', self.page)
        self.assertIn('function findingDetailTable(', self.page)
        # The cutout is fetched at runtime: the page ships without an
        # img src, without a SkyView link, and without the survey option.
        self.assertNotIn('<img src', self.page)
        self.assertNotIn('src="https://alasky', self.page)
        self.assertNotIn('src="https://aladin', self.page)
        self.assertNotIn("skyview", self.page.lower())
        self.assertNotIn("findingSurvey", self.page)
        self.assertNotIn('function hpxHash(', self.page)
        self.assertNotIn('function updateFindingImage(', self.page)
        self.assertNotIn('function drawCutoutOverlay(', self.page)
        self.assertIn('function simbadCooLookup(', self.page)
        self.assertIn('function drawFindingChart(', self.page)
        self.assertIn('function findingQueryUrl(', self.page)
        self.assertIn('"finding-panel"', self.page)
        self.assertIn("finding: false", self.page)
        self.assertIn("Finding chart", self.page)
        self.assertIn(
            '<div id="target-details" style="display:none"></div>',
            self.page)
        self.assertIn("#target-details { margin-top: 0.6rem", self.page)
        self.assertNotIn('id="target-tip"', self.page)
        self.assertNotIn("function showTargetTip(", self.page)
        self.assertIn("function renderTargetDetails(", self.page)
        self.assertIn("var findingDetailsCache", self.page)
        self.assertIn("function maybeFetchFindingDetail(", self.page)
        self.assertIn("function targetDetailTable(", self.page)
        self.assertIn("function parallaxDistance(", self.page)

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

    def test_embedded_initial_uses_preset_table(self):
        """The embedded workspace's preset row carries the table's numbers.

        The deployed page embeds whatever workspace.load returns, so a
        saved file with a preset name and stale site numbers must embed
        the reference table's row, not the file's numbers.
        """
        ws = new_workspace()
        ws["observatory"] = {
            "name": "Mount Graham (LBT)",
            "lat": 32.44278,
            "lon": -111.155,
            "elevation_m": 2400.0,
            "timezone": "America/Phoenix",
            "preset": "Mount Graham (LBT)",
        }
        ws["objects"] = []
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "workspace.json"
            save(path, ws)
            data = load(path)
            page = build_site(data, Path(scratch) / "index.html").read_text(
                encoding="utf-8")
        obs = _embedded_json(page, "INITIAL")["observatory"]
        self.assertAlmostEqual(obs["lat"], 32.70131, places=5)
        self.assertAlmostEqual(obs["lon"], -109.88906, places=5)
        self.assertAlmostEqual(obs["elevation_m"], 3221.0, places=4)


class DefaultWorkspaceBuildTests(unittest.TestCase):
    """The deployed page is a build of the repository's default workspace.

    ``--site`` without ``--workspace`` embeds the saved workspace of
    whoever runs it, so the page must be reproducible from the
    repository alone: ``site/default_workspace.json`` is the workspace
    the deployed page embeds, and the deployed page is that file's
    build apart from the build date.
    """

    def test_default_workspace_is_a_workspace(self):
        """site/default_workspace.json loads as a workspace document."""
        data = load(ROOT / "site" / "default_workspace.json")
        self.assertIn("observatory", data)
        self.assertIsInstance(data["objects"], list)
        self.assertIn("time_range", data)

    def test_site_index_matches_default_workspace_build(self):
        """site/index.html equals the build from site/default_workspace.json apart from the build date."""
        data = json.loads(
            (ROOT / "site" / "default_workspace.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            built = build_site(data, Path(tmp) / "page.html").read_text(
                encoding="utf-8")
        deployed = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
        strip = lambda t: re.sub(r"generated by airmass-gui on \S+;",
                                 "generated by airmass-gui on X;", t)
        self.assertEqual(strip(built), strip(deployed))


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class PresetAuthorityPageTests(unittest.TestCase):
    """normalizeState on the page: a preset name is authoritative.

    Every load path (loadState, loadWorkspaceDoc, Reset) passes through
    normalizeState, so it must rename a stale preset and overwrite the
    saved numbers with the embedded table's row. A state without a
    preset keeps its saved numbers.
    """

    @classmethod
    def setUpClass(cls):
        from test_weather_live import STUBS
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      script, re.S).group(1))
        tz_table = {"UTC": 0, "America/Denver": -7}
        for o in obs:
            tz_table.setdefault(o["timezone"], -7)
        om_body = json.loads(json.dumps({
            "latitude": 32.44236, "longitude": -111.16411,
            "timezone": "America/Phoenix", "timezone_abbreviation": "MST",
            "utc_offset_seconds": -25200, "elevation": 645.0,
            "generationtime_ms": 0.1, "hourly_units": {},
            "hourly": {"time": [], "cloud_cover": [],
                       "precipitation_probability": []},
        }))
        stubs = (STUBS
                 .replace("__ALL_IDS__", json.dumps(all_ids))
                 .replace("__TZ_TABLE__", json.dumps(tz_table))
                 .replace("__SIMBAD_BODY__", json.dumps(""))
                 .replace("__OPEN_METEO_BODY__", json.dumps(json.dumps(om_body))))
        cls.interp = dukpy.JSInterpreter()
        cls.interp.evaljs(stubs + ";\n0")
        cls.interp.evaljs(script + ";\n0")

    def _normalize_observatory(self, obs_obj):
        code = ("var s = { observatory: " + json.dumps(obs_obj) +
                ", objects: [], "
                "time_range: { start: '2026-11-25T00:00', days: 1.0 } };"
                "normalizeState(s);JSON.stringify(s.observatory);")
        return json.loads(self.interp.evaljs(code))

    def test_normalize_state_preset_authoritative(self):
        """A saved preset name overwrites the stored site numbers."""
        obs = self._normalize_observatory({
            "name": "Mount Graham (LBT)",
            "lat": 32.44278, "lon": -111.155, "elevation_m": 2400.0,
            "timezone": "America/Phoenix",
            "preset": "Mount Graham (LBT)",
        })
        self.assertAlmostEqual(obs["lat"], 32.70131, places=5)
        self.assertAlmostEqual(obs["lon"], -109.88906, places=5)
        self.assertAlmostEqual(obs["elevation_m"], 3221.0, places=4)
        self.assertEqual(obs["timezone"], "America/Phoenix")

    def test_normalize_state_renamed_preset_authoritative(self):
        """loadWorkspaceDoc renames nothing; normalizeState must do it.

        A state saved under the older name loads with the current
        preset and the table's numbers.
        """
        obs = self._normalize_observatory({
            "name": "LBT (Large Binocular Telescope)",
            "lat": 32.44278, "lon": -111.155, "elevation_m": 2400.0,
            "timezone": "America/Phoenix",
            "preset": "LBT (Large Binocular Telescope)",
        })
        self.assertEqual(obs["preset"], "Mount Graham (LBT)")
        self.assertEqual(obs["name"], "Mount Graham (LBT)")
        self.assertAlmostEqual(obs["lat"], 32.70131, places=5)
        self.assertAlmostEqual(obs["lon"], -109.88906, places=5)
        self.assertAlmostEqual(obs["elevation_m"], 3221.0, places=4)

    def test_normalize_state_no_preset_unchanged(self):
        """An observatory without a preset keeps its saved numbers."""
        obs = self._normalize_observatory({
            "name": "Custom site",
            "lat": 32.44278, "lon": -111.155, "elevation_m": 2400.0,
            "timezone": "America/Phoenix",
        })
        self.assertAlmostEqual(obs["lat"], 32.44278, places=5)
        self.assertAlmostEqual(obs["lon"], -111.155, places=5)
        self.assertAlmostEqual(obs["elevation_m"], 2400.0, places=4)


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class SecurityPageTests(unittest.TestCase):
    """Workspace data reaches the DOM as text, never as markup.

    A workspace can arrive in a #w= link or an imported file, so target
    colors and names are attacker-controllable: normalizeState must
    reduce every color to a palette hex, renderLegend must build its
    swatches with DOM calls instead of string markup, and the summary,
    calendar and target-details panels must escape target names.
    """

    @classmethod
    def setUpClass(cls):
        from test_weather_live import STUBS
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      script, re.S).group(1))
        tz_table = {"UTC": 0, "America/Denver": -7}
        for o in obs:
            tz_table.setdefault(o["timezone"], -7)
        om_body = json.loads(json.dumps({
            "latitude": 32.70131, "longitude": -109.88906,
            "timezone": "America/Phoenix", "timezone_abbreviation": "MST",
            "utc_offset_seconds": -25200, "elevation": 3221.0,
            "generationtime_ms": 0.1, "hourly_units": {},
            "hourly": {"time": [], "cloud_cover": [],
                       "precipitation_probability": []},
        }))
        stubs = (STUBS
                 .replace("__ALL_IDS__", json.dumps(all_ids))
                 .replace("__TZ_TABLE__", json.dumps(tz_table))
                 .replace("__SIMBAD_BODY__", json.dumps(""))
                 .replace("__OPEN_METEO_BODY__", json.dumps(json.dumps(om_body))))
        cls.interp = dukpy.JSInterpreter()
        cls.interp.evaljs(stubs + ";\n0")
        cls.interp.evaljs(script + ";\n0")

    def _set_state(self, objects, options=None):
        """Load a one-observatory state with the given objects."""
        code = (
            "state = normalizeState({ observatory: { name: 'Mount Graham (LBT)',"
            " lat: 32.70131, lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: " + json.dumps(objects) + ","
            " options: " + json.dumps(options or {}) + ","
            " time_range: { start: '2026-11-25T00:00', days: 1.0 } });"
        )
        self.interp.evaljs(code)

    def test_normalize_state_sanitizes_target_colors(self):
        """A color that is not a six-digit hex becomes the row's palette color.

        The color input accepts any string a link or file can carry
        (e.g. '#1f77b4" data-x="1'); only a palette hex may reach the
        chart or the legend.
        """
        self._set_state([
            {"name": "a", "ra": "18:36:56.3", "dec": "38:47:01.5",
             "color": "#1f77b4\" data-x=\"1"},
            {"name": "b", "ra": "18:36:56.3", "dec": "38:47:01.5"},
        ])
        colors = self.interp.evaljs("JSON.stringify(state.objects.map"
                                    "(function (o) { return o.color; }))")
        script = re.findall(r"<script>(.*?)</script>", self.page, re.S)[0]
        palette = json.loads(
            re.search(r"var PALETTE =\s*(\[.*?\]);",
                      re.search(r"<script>(.*?)</script>",
                                self.page, re.S).group(0),
                      re.S).group(1))
        self.assertEqual(json.loads(colors), [palette[0], palette[1]])

    def test_render_legend_builds_dom_not_markup(self):
        """renderLegend sets the swatch color as a style property.

        A color string with an embedded quote must not break out of
        the style attribute or add markup; the name must land as
        text content.
        """
        self._set_state([{"name": "a", "ra": "18:36:56.3",
                          "dec": "38:47:01.5"}])
        # The probe attribute is hyphen-free because the stub DOM's
        # attribute parser does not accept hyphenated names.
        self.interp.evaljs("renderLegend(" + json.dumps([
            {"name": "A", "color": '#1f77b4" x="1'},
            {"name": "B", "color": "blue", "alpha": 0.3},
        ]) + ");")
        out = self.interp.evaljs("""
(function () {
  var box = $("legend");
  var swatches = [], injected = [], texts = [];
  var walk = function (el) {
    var cls = el.className || el["class"] || "";
    if (cls === "sw") {
      swatches.push(el.style && el.style.background);
      if (el.getAttribute("x") !== null) injected.push(cls);
    }
    if (el.textContent) texts.push(el.textContent);
    (el.children || []).forEach(walk);
  };
  walk(box);
  return JSON.stringify({ swatches: swatches, injected: injected, texts: texts });
})()""")
        data = json.loads(out)
        self.assertEqual(data["injected"], [],
                         "legend markup carries an injected attribute")
        self.assertEqual(data["swatches"],
                         ['#1f77b4" x="1', "blue"])
        self.assertIn("A", data["texts"])
        self.assertIn("B", data["texts"])

    def test_target_name_escaped_in_panels(self):
        """A hostile target name stays inert in summary, calendar and details.

        The name is rendered through htmlEscape in all three panels; a
        raw '<b>' in any panel's markup would start a tag.
        """
        self._set_state(
            [{"name": "<b>x</b>\"'&", "ra": "18:36:56.3",
              "dec": "38:47:01.5"}],
            options={"extraInfo": True})
        for fn, box in (("renderSummary", "summary"),
                        ("renderCalendar", "calendar"),
                        ("renderTargetDetails", "target-details")):
            self.interp.evaljs(fn + "();")
            html = self.interp.evaljs(
                "JSON.stringify($(\'%s\').__innerHTML)" % box)
            self.assertIn("&lt;b&gt;x&lt;/b&gt;", html,
                          fn + " did not escape the name")
            self.assertNotIn("<b>x</b>", html,
                             fn + " contains raw markup in the name")


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class CdsSkyImageTests(unittest.TestCase):
    """The finding chart's sky image centers on the J2000 position.

    The CDS HiPS2FITS cutout is requested at J2000, so the inline img
    and its new-tab link must carry the J2000 position that objCoords
    returns, not the precessed equinox-of-date position; precessing
    shifts the image center by the precession (10.7 arcmin for Vega
    on 2026-11-25), which puts the target outside the returned field.
    """

    @classmethod
    def setUpClass(cls):
        from test_weather_live import STUBS
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      script, re.S).group(1))
        tz_table = {"UTC": 0, "America/Phoenix": -7}
        for o in obs:
            tz_table.setdefault(o["timezone"], -7)
        stubs = (STUBS
                 .replace("__ALL_IDS__", json.dumps(all_ids))
                 .replace("__TZ_TABLE__", json.dumps(tz_table))
                 .replace("__SIMBAD_BODY__", json.dumps(""))
                 .replace("__OPEN_METEO_BODY__",
                          json.dumps(json.dumps({}))))
        cls.interp = dukpy.JSInterpreter()
        cls.interp.evaljs(stubs + ";" + chr(10) + "0")
        cls.interp.evaljs(script + ";" + chr(10) + "0")

    def _img_position_deg(self, ra, dec):
        """Return the inline image's center parsed back to degrees."""
        self.interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: " + json.dumps(
                [{"name": "t", "ra": ra, "dec": dec}]) + ","
            " options: { finding: true },"
            " time_range: { start: '2026-11-25T00:00', days: 1.0 } });"
            "$('finding-target').value = '0';"
            "$('finding-zoom').value = '640';"
            "updateFindingChart();"
        )
        src = self.interp.evaljs(
            "JSON.stringify($('finding-img').src)")
        ra_raw = urllib.parse.unquote(
            re.search(r"[?&]ra=([^&]+)", src).group(1))
        dec_raw = urllib.parse.unquote(
            re.search(r"[?&]dec=([^&]+)", src).group(1))
        return float(ra_raw), float(dec_raw)

    def test_img_src_is_cds_cutout(self):
        """The inline image and its new-tab link share one URL, and
        center within 1 arcsec of J2000 for Vega, Sirius and M31."""
        for ra, dec, (want_ra, want_dec) in (
                ("279.2347", "38.7837", (279.2347, 38.7837)),
                ("101.2872", "-16.7161", (101.2872, -16.7161)),
                ("00:42:44.33", "+41:16:07.32", (10.6847, 41.2687))):
            got_ra, got_dec = self._img_position_deg(ra, dec)
            self.assertAlmostEqual(got_ra % 360, want_ra, delta=1.0 / 3600)
            self.assertAlmostEqual(got_dec, want_dec, delta=1.0 / 3600)
            same = self.interp.evaljs(
                "JSON.stringify($('finding-img').src === "
                "$('finding-cdslink').href)")
            self.assertEqual(same, "true")

    def _cds_link_position_deg(self, ra, dec):
        """Return the CDS cutout link's center parsed back to degrees."""
        self.interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: " + json.dumps(
                [{"name": "t", "ra": ra, "dec": dec}]) + ","
            " options: { finding: true },"
            " time_range: { start: '2026-11-25T00:00', days: 1.0 } });"
            "$('finding-target').value = '0';"
            "$('finding-zoom').value = '640';"
            "updateFindingChart();"
        )
        href = self.interp.evaljs(
            "JSON.stringify($('finding-cdslink').href)")
        ra_raw = urllib.parse.unquote(
            re.search(r"[?&]ra=([^&]+)", href).group(1))
        dec_raw = urllib.parse.unquote(
            re.search(r"[?&]dec=([^&]+)", href).group(1))
        return float(ra_raw), float(dec_raw)

    def test_cdslink_center_is_j2000(self):
        """Vega, Sirius and M31 cutouts center within 1 arcsec of J2000."""
        for ra, dec, (want_ra, want_dec) in (
                ("279.2347", "38.7837", (279.2347, 38.7837)),
                ("101.2872", "-16.7161", (101.2872, -16.7161)),
                ("00:42:44.33", "+41:16:07.32", (10.6847, 41.2687))):
            got_ra, got_dec = self._cds_link_position_deg(ra, dec)
            self.assertAlmostEqual(got_ra % 360, want_ra, delta=1.0 / 3600)
            self.assertAlmostEqual(got_dec, want_dec, delta=1.0 / 3600)

    def test_hips2fits_url_matches_recorded_fetch(self):
        """The URL builder reproduces the recorded reply's URL.

        tests/fixtures/cds_hips2fits_m31.jpg was requested from the CDS
        HiPS2FITS service with hips=DSS2%2Fcolor, 500 x 500, TAN, fov
        0.25, M31's J2000 center, format jpg.
        """
        url = self.interp.evaljs(
            "hips2fitsImageUrl(10.6847083, 41.26875)")
        self.assertEqual(
            url,
            "https://alasky.cds.unistra.fr/hips-image-services/"
            "hips2fits?hips=DSS2%2Fcolor&width=500&height=500"
            "&projection=TAN&fov=0.25&ra=10.68471&dec=41.26875"
            "&format=jpg")
        url = self.interp.evaljs(
            "hips2fitsImageUrl(101.2872, -16.7161)")
        self.assertEqual(
            url,
            "https://alasky.cds.unistra.fr/hips-image-services/"
            "hips2fits?hips=DSS2%2Fcolor&width=500&height=500"
            "&projection=TAN&fov=0.25&ra=101.28720&dec=-16.71610"
            "&format=jpg")
        url = self.interp.evaljs(
            "hips2fitsImageUrl(10.6847083, 41.26875, 0.125)")
        self.assertIn("&projection=TAN&fov=0.125&", url)

    def test_zoom_ties_image_field_to_chart(self):
        """The cutout's field of view is twice the chart radius.

        The slider runs the radius from 0.5 to 7.5 arcmin, so the
        requested cutout field runs from 1 arcmin to 0.25 deg (the
        recorded DSS2 field): at both ends the img URL carries that
        fov.
        """
        for zoom, fov in (("0", 1.0 / 60.0), ("1000", 0.25)):
            self.interp.evaljs(
                "state = normalizeState({ observatory: {"
                " name: 'Mount Graham (LBT)', lat: 32.70131,"
                " lon: -109.88906, elevation_m: 3221.0,"
                " timezone: 'America/Phoenix',"
                " preset: 'Mount Graham (LBT)' },"
                " objects: " + json.dumps(
                    [{"name": "t", "ra": "279.2347", "dec": "38.7837"}]) + ","
                " options: { finding: true },"
                " time_range: { start: '2026-11-25T00:00', days: 1.0 } });"
                "$('finding-target').value = '0';"
                "$('finding-zoom').value = '" + zoom + "';"
                "updateFindingChart();"
            )
            src = self.interp.evaljs(
                "JSON.stringify($('finding-img').src)")
            got = float(re.search(r"[?&]fov=([^&]+)", src).group(1))
            self.assertAlmostEqual(got, fov, delta=1e-5)

    def test_panes_share_fixed_square(self):
        """Star chart and sky image are the same fixed square.

        The pane widths follow the shared canvas size, not the info
        tables, so a table that grows pushes the window down instead
        of resizing the squares: after updateFindingChart the canvas,
        the image and both panes all carry findingCanvasSize().
        """
        self.interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: " + json.dumps(
                [{"name": "t", "ra": "279.2347", "dec": "38.7837"}]) + ","
            " options: { finding: true },"
            " time_range: { start: '2026-11-25T00:00', days: 1.0 } });"
            "$('finding-target').value = '0';"
            "$('finding-zoom').value = '1000';"
            "var fc = $('finding-chart');"
            "fc.getContext = function () { return __ctx; };"
            "var c = objCoords(state.objects[0], Date.now());"
            "var q = findingQueryRadius(7.5, findingCanvasSize());"
            "findingCache[c.raDeg.toFixed(4) + ',' + c.dec.toFixed(4) + "
            " ',' + q.toFixed(1)] = { rows: [], count: 0 };"
            "updateFindingChart();"
        )
        sizes = json.loads(self.interp.evaljs(
            "JSON.stringify({ canvas: $('finding-chart').style.width,"
            " img: $('finding-img').style.width,"
            " imgH: $('finding-img').style.height,"
            " left: $('finding-chart-pane').style.width,"
            " right: $('finding-img-pane').style.width,"
            " size: findingCanvasSize() + 'px' })"))
        for key in ("canvas", "img", "imgH", "left", "right"):
            self.assertEqual(sizes[key], sizes["size"], key)

    def test_hover_mark_encircled_crosshair(self):
        """Hovering the star chart marks the nearest object with an
        encircled crosshair in the theme's highlight color.

        The mark is drawn on top of the chart at the pixel position of
        the object whose details the hover panel shows, and follows it:
        hovering a different object moves the mark, and its color is
        THEME.highlight (red in the dark theme, white in the red/night
        theme, which the theme-sel change handler re-applies through
        updateFindingChart).
        """
        self.interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: " + json.dumps(
                [{"name": "Vega", "ra": "279.2347", "dec": "38.7837"}]) + ","
            " options: { finding: true },"
            " time_range: { start: '2026-11-25T00:00', days: 1.0 } });"
            "$('finding-target').value = '0';"
            "$('finding-zoom').value = '1000';"
            "var fc = $('finding-chart');"
            "fc.getContext = function () { return __ctx; };"
            "var fcsize = findingCanvasSize();"
            "var c = objCoords(state.objects[0], Date.now());"
            "var q = findingQueryRadius(7.5, fcsize);"
            "findingCache[c.raDeg.toFixed(4) + ',' + c.dec.toFixed(4) + "
            " ',' + q.toFixed(1)] = { rows: ["
            "  { name: 'Vega', raDeg: c.raDeg, dec: c.dec, v: 0.03 },"
            "  { name: 'faint', raDeg: 279.3, dec: 38.7, v: 9.0 } ],"
            " count: 2 };"
            "__arcCalls = [];"
            "__strokeSet = [];"
            "(function () {"
            "  var orig = __ctx.arc;"
            "  __ctx.arc = function (x, y, r) {"
            "    __arcCalls.push([x, y, r]); return orig(x, y, r); };"
            "  var style = null;"
            "  Object.defineProperty(__ctx, 'strokeStyle', {"
            "    get: function () { return style; },"
            "    set: function (v) { __strokeSet.push(v); style = v; }"
            "  });"
            "})(); "
            "updateFindingChart();"
        )
        center = self.interp.evaljs("fcsize / 2")
        self.interp.evaljs(
            "updateFindingInfo(" + repr(center) + ", " + repr(center)
            + "); 0")
        arcs = json.loads(self.interp.evaljs("JSON.stringify(__arcCalls)"))
        strokes = json.loads(self.interp.evaljs("JSON.stringify(__strokeSet)"))
        # The mark sits at the gnomonic projection of the hovered row,
        # which equals the canvas center up to float noise for a row at
        # the center, so allow a sub-nanopixel tolerance.
        hits = [a for a in arcs
                if abs(a[2] - 9) < 1e-9 and abs(a[0] - center) < 1e-6
                and abs(a[1] - center) < 1e-6]
        self.assertTrue(hits,
                        "no encircled crosshair on the hovered object")
        self.assertEqual(strokes[-1], "#ff5555",
                         "crosshair not stroked in the theme highlight")
        # The mark follows the object: hover the faint row instead.
        self.interp.evaljs(
            "var p2 = gnomonicProject(279.3, 38.7,"
            " findingHover.center.raDeg, findingHover.center.dec);"
            "var scale2 = (fcsize / 2 - 8) / d2r(findingHover.radiusArcmin"
            " / 60);"
            "updateFindingInfo(fcsize / 2 - p2.x * scale2,"
            " fcsize / 2 - p2.y * scale2); 0")
        ok = json.loads(self.interp.evaljs(
            "JSON.stringify((function () {"
            "  var last = __arcCalls[__arcCalls.length - 1];"
            "  return { ok: Math.abs(last[0] - (fcsize / 2 - p2.x * scale2))"
            "           < 1e-9"
            "          && Math.abs(last[1] - (fcsize / 2 - p2.y * scale2))"
            "           < 1e-9"
            "          && Math.abs(last[2] - 9) < 1e-9 };"
            "})())"))
        self.assertTrue(ok["ok"], "crosshair did not follow the hover")
        # Red/night theme: the theme-sel handler re-draws through
        # updateFindingChart; the mark comes back white.
        self.interp.evaljs(
            "applyTheme('red'); updateFindingChart(); 0")
        ok2 = json.loads(self.interp.evaljs(
            "JSON.stringify({ stroke: __strokeSet[__strokeSet.length - 1] })"))
        self.assertEqual(ok2["stroke"], "#ffffff",
                         "red/night theme crosshair is not white")

    def test_img_error_shows_unavailable(self):
        """A failed image load says the image is unavailable, hides the
        img, and leaves the new-tab link in place."""
        self.interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: " + json.dumps(
                [{"name": "t", "ra": "279.2347", "dec": "38.7837"}]) + ","
            " options: { finding: true },"
            " time_range: { start: '2026-11-25T00:00', days: 1.0 } });"
            "$('finding-target').value = '0';"
            "$('finding-zoom').value = '640';"
            "updateFindingChart();"
            "$('finding-img').dispatchEvent({ type: 'error' });"
        )
        status = json.loads(self.interp.evaljs(
            "JSON.stringify($('finding-img-status').textContent)"))
        self.assertIn("unavailable", status)
        display = json.loads(self.interp.evaljs(
            "JSON.stringify($('finding-img').style.display)"))
        self.assertEqual(display, "none")

    def test_scale_bar_matches_recorded_wcs(self):
        """The scale bar spans one third of the cutout's field.

        At the default maximum zoom-out (7.5 arcmin radius, 0.25 deg
        cutout) findingScaleArcmin picks 5 arcmin, so after
        updateFindingChart the bar is 33.33 percent of the image
        width with a 5' label; the fov the URL builder requests
        reproduces the recorded FITS WCS's center-row field.
        """
        self.assertEqual(self.interp.evaljs("findingScaleArcmin(15)"), 5)
        self.interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: " + json.dumps(
                [{"name": "t", "ra": "00:42:44.33", "dec": "+41:16:07.32"}]) + ","
            " options: { finding: true },"
            " time_range: { start: '2026-11-25T00:00', days: 1.0 } });"
            "$('finding-target').value = '0';"
            "$('finding-zoom').value = '1000';"
            "updateFindingChart();"
        )
        width = float(self.interp.evaljs(
            "parseFloat($('finding-scale-bar').style.width)"))
        self.assertAlmostEqual(width, 100.0 * 5.0 / 15.0, delta=0.01)
        label = json.loads(self.interp.evaljs(
            "JSON.stringify($('finding-scale-label').textContent)"))
        self.assertEqual(label, "5\u2032")
        src = self.interp.evaljs("JSON.stringify($('finding-img').src)")
        fov = float(re.search(r"[?&]fov=([^&]+)", src).group(1))
        header = fits.getheader(CDS_FITS_FIXTURE)
        self.assertAlmostEqual(fov, 500.0 * abs(header["CDELT1"]),
                               delta=1e-5)


CDS_FIXTURE = (Path(__file__).resolve().parent / "fixtures" /
               "cds_hips2fits_m31.jpg")
CDS_FITS_FIXTURE = (Path(__file__).resolve().parent / "fixtures" /
                    "cds_hips2fits_m31_fits.fits")


def _jpeg_sof0(data):
    """Return (width, height, n_components) from a JPEG's first SOF0.

    Walks the marker segments of a JPEG byte stream: each marker is FF
    followed by a type; most carry a big-endian length, SOF0 (FF C0)
    the precision, height, width and the component count.
    """
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            raise ValueError("no JPEG marker at byte %d" % i)
        while data[i + 1] == 0xFF:
            i += 1
        marker = data[i + 1]
        if marker == 0xC0:
            # precision byte, then height, width, component count
            height = int.from_bytes(data[i + 5:i + 7], "big")
            width = int.from_bytes(data[i + 7:i + 9], "big")
            return width, height, data[i + 9]
        if marker == 0xD9 or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i += 2
            continue
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    raise ValueError("SOF0 not found")


class TestCdsHiPS2FITSFixture(unittest.TestCase):
    """The archived CDS HiPS2FITS reply is the recorded M31 cutout.

    Recorded 2026-11-28 00:07:56 UTC (HTTP 200, content-type
    image/jpeg, content-disposition cutout-CDS_P_DSS2_color.jpg) from
    https://alasky.cds.unistra.fr/hips-image-services/hips2fits
    ?hips=DSS2%2Fcolor&width=500&height=500&projection=TAN&fov=0.25
    &ra=10.6847083&dec=41.26875&format=jpg, saved unedited.
    """

    def test_reply_is_a_500x500_jpeg(self):
        """The reply is a 3-component 500 x 500 JPEG, as recorded."""
        data = CDS_FIXTURE.read_bytes()
        self.assertEqual(data[:3], b"\xff\xd8\xff")
        self.assertEqual(_jpeg_sof0(data), (500, 500, 3))

    def test_m31_disk_is_in_the_cutout(self):
        """The field is not blank sky: the M31 disk is present.

        The center 60 x 60 median exceeds the mean of the four corner
        40 x 40 box medians by more than 1.3 (measured 1.7 on the
        recorded reply); the same field from SkyView's DSS copy, which
        lacks M31, was flat (center-to-sky ratio 1.25).
        """
        if Image is None:
            self.skipTest("Pillow not installed")
        with Image.open(CDS_FIXTURE) as im:
            gray = np.asarray(im.convert("L")).astype(float)
        cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
        central = np.median(gray[cy - 30:cy + 30, cx - 30:cx + 30])
        h, w = gray.shape
        corners = [
            np.median(gray[10:50, 10:50]),
            np.median(gray[10:50, w - 50:w - 10]),
            np.median(gray[h - 50:h - 10, 10:50]),
            np.median(gray[h - 50:h - 10, w - 50:w - 10]),
        ]
        self.assertGreater(central / float(np.mean(corners)), 1.3)


class TestCdsHiPS2FITSFitsFixture(unittest.TestCase):
    """The archived CDS HiPS2FITS FITS reply fixes the cutout's WCS.

    Recorded 2026-11-28 01:02 UTC (HTTP 200, content-type
    application/fits) from the same service and parameters as
    tests/fixtures/cds_hips2fits_m31.jpg with format=fits:
    https://alasky.cds.unistra.fr/hips-image-services/hips2fits
    ?hips=DSS2%2Fcolor&width=500&height=500&projection=TAN&fov=0.25
    &ra=10.6847083&dec=41.26875&format=fits, saved unedited. The
    JPEG reply carries no WCS, so this reply fixes the cutout's field
    of view and plate scale.
    """

    @classmethod
    def setUpClass(cls):
        cls.header = fits.getheader(CDS_FITS_FIXTURE)

    def test_reply_is_a_500x500_tan_cutout(self):
        """Single HDU, 500 x 500 x 4 uint8, TAN in ICRS centered on
        M31's J2000 position."""
        header = self.header
        self.assertEqual(header["NAXIS"], 3)
        self.assertEqual((header["NAXIS1"], header["NAXIS2"],
                          header["NAXIS3"]), (500, 500, 4))
        self.assertEqual(header["BITPIX"], 8)
        self.assertEqual(header["CTYPE1"], "RA---TAN")
        self.assertEqual(header["CTYPE2"], "DEC--TAN")
        self.assertEqual(header["RADESYS"], "ICRS")
        self.assertAlmostEqual(header["CRVAL1"], 10.6847083, delta=1e-6)
        self.assertAlmostEqual(header["CRVAL2"], 41.26875, delta=1e-6)

    def test_field_and_plate_scale(self):
        """The fov parameter is the center-row width: 0.25 deg,
        about 1.8 arcsec per pixel."""
        header = self.header
        self.assertAlmostEqual(500.0 * abs(header["CDELT1"]), 0.25,
                               delta=0.25 * 0.001)
        self.assertAlmostEqual(abs(header["CDELT1"]) * 3600.0, 1.8,
                               delta=1.8 * 0.001)


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class MoonTopocentricTests(unittest.TestCase):
    """The displayed Moon is topocentric, not geocentric.

    The geocentric Moon and the Moon an observer sees differ by the
    diurnal parallax: at the LBT over 30 days the altitude differs by
    0.74 deg on mean and 1.03 deg at most (against astropy's
    topocentric Moon). moonTopocentric adds the correction of Meeus,
    Astronomical Algorithms (2nd ed.), chapters 11 and 40.
    """

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.page = build_site(new_workspace(),
                              Path(cls.dir.name) / "index.html").read_text(
            encoding="utf-8")
        cls.js = dukpy.JSInterpreter()
        cls.js.evaljs(_extract_core(cls.page))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_moon_altitude_is_topocentric(self):
        """Hourly for 30 days at the LBT, within 0.05 deg of astropy.

        astropy's get_body('moon', t, location) is the topocentric
        apparent position; the AltAz frame's refraction is zeroed,
        as the page applies none.
        """
        loc = EarthLocation.from_geodetic(
            LBT.lon * u.deg, LBT.lat * u.deg, LBT.elevation_m)
        t0 = Time("2026-11-25 00:00:00")
        max_diff = 0.0
        n = 0
        for i in range(720):
            t = t0 + i * 3600 * u.s
            aa = AltAz(obstime=t, location=loc)
            aa.refraction = 0 * u.deg
            alt_astro = float(
                SkyCoord(get_body("moon", t, loc)).transform_to(aa).alt.deg)
            if alt_astro < -2.0:
                continue
            jd = float(t.jd)
            moon = json.loads(self.js.evaljs(
                "JSON.stringify(moonTopocentric(%r, %r, %r, %r))"
                % (jd, LBT.lat, LBT.lon, LBT.elevation_m)))
            alt_js = self.js.evaljs(
                "horizonCoords(%r, %r, %r, %r, %r).alt"
                % (moon["raH"], moon["dec"], jd, LBT.lat, LBT.lon))
            n += 1
            max_diff = max(max_diff, abs(alt_js - alt_astro))
        self.assertGreater(n, 300)
        self.assertLess(max_diff, 0.05)

    def test_page_draw_and_tooltip_use_topocentric_moon(self):
        """draw() and the tooltip call moonTopocentric, not moonRaDec."""
        self.assertGreaterEqual(
            self.page.count("moonTopocentric(jd, obs.lat, obs.lon, obs.elevation_m)"),
            2)


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class TransitTimesRaOfDateTests(unittest.TestCase):
    """Transit times use the RA of date, not J2000.

    The sidereal time is in the equinox of date, so transitTimes
    must be given the RA precessed to TETE. Fed the J2000 RA, the
    first transits of 2026-11-25 at the LBT come out early by
    52 s (Vega), 73 s (Sirius), 89 s (M31) and 186 s (Dec +80).
    """

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.page = build_site(new_workspace(),
                              Path(cls.dir.name) / "index.html").read_text(
            encoding="utf-8")
        cls.js = dukpy.JSInterpreter()
        cls.js.evaljs(_extract_core(cls.page))
        cls.loc = EarthLocation.from_geodetic(
            LBT.lon * u.deg, LBT.lat * u.deg, LBT.elevation_m)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def _astropy_first_transit(self, ra_deg, dec_deg, start, span_s):
        """First instant after start at which the apparent hour angle
        is zero, from astropy (apparent sidereal time minus the TETE
        RA), by a 60 s scan and bisection.
        """
        coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
        prev = None
        for i in range(int(span_s / 60.0)):
            t = start + i * 60 * u.s
            st_deg = t.sidereal_time(
                "apparent", self.loc.lon.deg * u.deg).to_value("deg") % 360
            d = (st_deg - coord.transform_to(TETE(obstime=t)).ra.deg) % 360
            if prev is not None and prev > 180.0 and d <= 180.0:
                lo = t - 60 * u.s
                hi = t
                for _ in range(40):
                    mid = lo + 0.5 * (hi - lo)
                    st_m = mid.sidereal_time(
                        "apparent", self.loc.lon.deg * u.deg).to_value(
                        "deg") % 360
                    dm = (st_m - coord.transform_to(TETE(obstime=mid)).ra.deg) % 360
                    if dm > 180.0:
                        lo = mid
                    else:
                        hi = mid
                return lo + 0.5 * (hi - lo)
            prev = d
        self.fail("no apparent transit within the span")

    def test_first_transit_matches_astropy(self):
        """First transit after 2026-11-25 00:00 UTC, within 10 s.

        The RA of date comes from precessIcrsToTete at the start of
        the scan, the way both callers now pass it.
        """
        start = Time("2026-11-25 00:00:00")
        start_ms = int((start - Time("1970-01-01 00:00:00")).sec * 1000)
        end_ms = start_ms + 48 * 3600 * 1000
        for ra, dec in ((279.2347, 38.7837), (101.2872, -16.7161),
                        (10.6847, 41.2687), (150.0, 80.0)):
            got_ms = self.js.evaljs(
                "JSON.stringify(transitTimes(precessIcrsToTete(%r, %r, "
                "jdFromUtcMs(%r)).raDeg / 15, %r, %r, %r))"
                % (ra, dec, start_ms, start_ms, end_ms, LBT.lon))
            got = json.loads(got_ms)[0]
            want = self._astropy_first_transit(ra, dec, start, 48 * 3600)
            diff_s = abs((Time(got / 1000, format="unix") - want).sec)
            self.assertLess(diff_s, 10.0,
                            "transit for (%s, %s) off by %.1f s" % (ra, dec, diff_s))

    def test_callers_precess_before_transit_times(self):
        """Both callers pass the RA of date, precessed at the scan start."""
        self.assertIn(
            "precessIcrsToTete(mid.raDeg, mid.dec, jdFromUtcMs(day[0])).raDeg / 15",
            self.page)
        self.assertIn(
            "precessIcrsToTete(mid.raDeg, mid.dec, jdFromUtcMs(v0)).raDeg / 15",
            self.page)


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class SummaryMoonMidnightTests(unittest.TestCase):
    """The per-night list gives the Moon's phase at local midnight.

    The illumination changes by up to 6 points in 12 hours, so the
    listed phase must be evaluated at a local midnight, not at the
    middle of each day's part of the chart (local noon for a full
    day).
    """

    @classmethod
    def setUpClass(cls):
        from test_weather_live import STUBS
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      script, re.S).group(1))
        tz_table = {"UTC": 0, "America/Phoenix": -7}
        for o in obs:
            tz_table.setdefault(o["timezone"], -7)
        stubs = (STUBS
                 .replace("__ALL_IDS__", json.dumps(all_ids))
                 .replace("__TZ_TABLE__", json.dumps(tz_table))
                 .replace("__SIMBAD_BODY__", json.dumps(""))
                 .replace("__OPEN_METEO_BODY__",
                          json.dumps(json.dumps({}))))
        cls.interp = dukpy.JSInterpreter()
        cls.interp.evaljs(stubs + ";" + chr(10) + "0")
        cls.interp.evaljs(script + ";" + chr(10) + "0")

    def test_moon_phase_evaluated_at_local_midnight(self):
        """Chart 2026-11-20 12:00 to 2026-11-22 12:00, America/Phoenix.

        The three listed days must use 2026-11-21 00:00,
        2026-11-22 00:00 and 2026-11-22 00:00 local: the first day's
        part runs to a midnight, the full day to its end midnight,
        and the last day starts at a midnight.
        """
        self.interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: [{\"name\": \"t\", \"ra\": \"18:36:56.3\","
            " \"dec\": \"+38:47:01.5\"}],"
            " options: { windows: true, threshold: 2.0 },"
            " time_range: { start: '2026-11-20T12:00', days: 2.0 } });"
            "var __moonJd = [];"
            "var __origMpi = moonPhaseInfo;"
            "moonPhaseInfo = function (jd) {"
            " __moonJd.push(jd); return __origMpi(jd); }; 0")
        self.interp.evaljs("renderSummary();")
        got = json.loads(self.interp.evaljs("JSON.stringify(__moonJd)"))
        want_ms = [int(datetime(y, m, d, 7, 0, 0,
                                tzinfo=timezone.utc).timestamp() * 1000)
                   for (y, m, d) in ((2026, 11, 21), (2026, 11, 22),
                                     (2026, 11, 22))]
        self.assertEqual(len(got), 3)
        for jd, want in zip(got, want_ms):
            self.assertAlmostEqual(jd, want / 86400000 + 2440587.5,
                                   delta=1.0 / 86400)


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class CenterNowTests(unittest.TestCase):
    """The Center button expands the time range so the current time is its middle.

    The clock is pinned. The range is 2026-11-25 00:00 to
    2026-11-27 00:00 America/Phoenix, i.e. 2026-11-25T07:00Z to
    2026-11-27T07:00Z. Clicking the chart's Center button expands
    the range symmetrically about the current time: the longer of
    the before-now and after-now portions is mirrored onto the
    shorter side, so the current time sits at the exact middle of
    the range. When the current time is outside the range the
    range expands until it is contained and mirrored the same way.
    The zoomed view is discarded and the range is saved.
    """

    RANGE_START = int(datetime(2026, 11, 25, 7, 0,
                               tzinfo=timezone.utc).timestamp() * 1000)
    RANGE_END = int(datetime(2026, 11, 27, 7, 0,
                             tzinfo=timezone.utc).timestamp() * 1000)
    HOUR = 3600000

    @classmethod
    def setUpClass(cls):
        from test_weather_live import STUBS
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      script, re.S).group(1))
        tz_table = {"UTC": 0, "America/Phoenix": -7}
        for o in obs:
            tz_table.setdefault(o["timezone"], -7)
        cls.stubs = (STUBS
                     .replace("__ALL_IDS__", json.dumps(all_ids))
                     .replace("__TZ_TABLE__", json.dumps(tz_table))
                     .replace("__SIMBAD_BODY__", json.dumps(""))
                     .replace("__OPEN_METEO_BODY__",
                              json.dumps(json.dumps({}))))
        cls.script = script

    def _interp(self, clock_ms):
        """Page interpreter with the clock pinned at the given epoch ms."""
        interp = dukpy.JSInterpreter()
        interp.evaljs(self.stubs + ";" + chr(10) + "0")
        interp.evaljs("Date.now = function () { return %d; }; 0"
                      % clock_ms)
        interp.evaljs(self.script + ";" + chr(10) + "0")
        return interp

    def _after_center(self, interp, now):
        """Click Center on the pinned range and read back the result.

        now is the pinned clock in epoch ms. Returns a dict with the
        range start string, the duration in days, the range's
        middle in epoch ms, and the view window (None when the full
        range is shown).
        """
        interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: [{\"name\": \"t\", \"ra\": \"18:36:56.3\","
            " \"dec\": \"+38:47:01.5\"}],"
            " options: { windows: true, threshold: 2.0 },"
            " time_range: { start: '2026-11-25T00:00', days: 2.0 } });"
            "$('center-now').click(); 0")
        return json.loads(interp.evaljs(
            "JSON.stringify({ start: state.time_range.start,"
            " days: state.time_range.days,"
            " center_ms: startUtcMs() + state.time_range.days * 43200000,"
            " view: view })"))

    def test_center_mirrors_after_side_when_it_is_larger(self):
        """Now inside the range with a longer after side: the range extends before."""
        now = int(datetime(2026, 11, 25, 19, 0,
                           tzinfo=timezone.utc).timestamp() * 1000)
        got = self._after_center(self._interp(now), now)
        self.assertEqual(got["start"], "2026-11-24 00:00")
        self.assertEqual(got["days"], 3.0)
        self.assertIsNone(got["view"])
        self.assertAlmostEqual(got["center_ms"], now, delta=5000)

    def test_center_mirrors_before_side_when_it_is_larger(self):
        """Now inside the range with a longer before side: the range extends after."""
        now = int(datetime(2026, 11, 26, 19, 0,
                           tzinfo=timezone.utc).timestamp() * 1000)
        got = self._after_center(self._interp(now), now)
        self.assertEqual(got["start"], "2026-11-25 00:00")
        self.assertEqual(got["days"], 3.0)
        self.assertIsNone(got["view"])
        self.assertAlmostEqual(got["center_ms"], now, delta=5000)

    def test_center_expands_when_now_is_before_range(self):
        """Now before the range: the range grows back until it is centered."""
        now = int(datetime(2026, 11, 24, 12, 0,
                           tzinfo=timezone.utc).timestamp() * 1000)
        got = self._after_center(self._interp(now), now)
        self.assertEqual(got["start"], "2026-11-21 10:00")
        self.assertAlmostEqual(got["days"], 134 / 24, delta=1e-4)
        self.assertIsNone(got["view"])
        self.assertAlmostEqual(got["center_ms"], now, delta=5000)

    def test_center_expands_when_now_is_after_range(self):
        """Now after the range: the range grows forward until it is centered."""
        now = int(datetime(2026, 11, 27, 18, 0,
                           tzinfo=timezone.utc).timestamp() * 1000)
        got = self._after_center(self._interp(now), now)
        self.assertEqual(got["start"], "2026-11-25 00:00")
        self.assertAlmostEqual(got["days"], 118 / 24, delta=1e-4)
        self.assertIsNone(got["view"])
        self.assertAlmostEqual(got["center_ms"], now, delta=5000)

    def test_center_leaves_range_unchanged_when_now_is_mid_range(self):
        """Now at the range's middle: nothing changes."""
        now = int(datetime(2026, 11, 26, 7, 0,
                           tzinfo=timezone.utc).timestamp() * 1000)
        got = self._after_center(self._interp(now), now)
        self.assertEqual(got["start"], "2026-11-25 00:00")
        self.assertEqual(got["days"], 2.0)
        self.assertIsNone(got["view"])


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class PanelOrderTests(unittest.TestCase):
    """The up/down arrows in a panel header reorder the panel window.

    Each panel window's header carries a small up arrow and a small
    down arrow at its right end; clicking one moves that panel one
    visible slot in that direction within its column, the sidebar
    panels among themselves and the panels of the main column among
    themselves. Hidden panels are skipped, and the non-panel
    siblings (the chart box, the sidebar strip) bound the columns,
    so the arrows are disabled at the column ends. Showing or
    hiding an optional panel refreshes the arrows on the next
    redraw. The panel order is per-page only: it is not part of the
    saved state. The clock is pinned to 2026-11-25T19:00Z.
    """

    NOW = int(datetime(2026, 11, 25, 19, 0,
                       tzinfo=timezone.utc).timestamp() * 1000)

    BUILD_TREE = """
    (function () {
      function prepPanel(id) {
        var p = $(id);
        p.className = "panel";
        var h2 = document.createElement("h2");
        h2.textContent = id;
        p.appendChild(h2);
        return p;
      }
      var sidebar = $("sidebar");
      var ids = ["panel-observatory", "panel-time", "panel-display",
                 "panel-targets", "panel-workspace"], i;
      for (i = 0; i < ids.length; i++) sidebar.appendChild(prepPanel(ids[i]));
      sidebar.appendChild($("sidebar-strip"));
      var main = document.createElement("main");
      main.className = "chart-col";
      var chart = document.createElement("div");
      chart.className = "chart-box";
      main.appendChild(chart);
      main.appendChild(prepPanel("finding-panel"));
      main.appendChild(prepPanel("summary-panel"));
      main.appendChild(prepPanel("wxreport-panel"));
      main.appendChild(prepPanel("calendar-panel"));
      document.body.appendChild(sidebar);
      document.body.appendChild(main);
      __mainCol = main;
    })(); 0"""

    @classmethod
    def setUpClass(cls):
        from test_weather_live import STUBS
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      script, re.S).group(1))
        tz_table = {"UTC": 0, "America/Phoenix": -7}
        for o in obs:
            tz_table.setdefault(o["timezone"], -7)
        cls.stubs = (STUBS
                     .replace("__ALL_IDS__", json.dumps(all_ids))
                     .replace("__TZ_TABLE__", json.dumps(tz_table))
                     .replace("__SIMBAD_BODY__", json.dumps(""))
                     .replace("__OPEN_METEO_BODY__",
                              json.dumps(json.dumps({}))))
        cls.script = script

    def _interp(self):
        """Page interpreter with the clock pinned at self.NOW."""
        interp = dukpy.JSInterpreter()
        interp.evaljs(self.stubs + ";" + chr(10) + "0")
        interp.evaljs("Date.now = function () { return %d; }; 0"
                      % self.NOW)
        interp.evaljs(self.script + ";" + chr(10) + "0")
        return interp

    def _setup(self, interp):
        """Pinned LBT state, the two column trees, and the arrows.

        The stub body carries the page's two panel columns: the five
        sidebar panels plus the sidebar strip, and the main column's
        chart box plus its four panels, with the finding chart and
        the weather report hidden as by default.
        """
        interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: [{\"name\": \"t\", \"ra\": \"18:36:56.3\","
            " \"dec\": \"+38:47:01.5\"}],"
            " options: { windows: true, calendar: true, threshold: 2.0 },"
            " time_range: { start: '2026-11-25T00:00', days: 1 } });"
            "syncAllControls(); 0")
        interp.evaljs(self.BUILD_TREE)
        interp.evaljs(
            '$("summary-panel").style.display = "";'
            '$("wxreport-panel").style.display = "none";'
            '$("finding-panel").style.display = "none";'
            '$("calendar-panel").style.display = ""; 0')
        interp.evaljs("addPanelOrderButtons(); 0")

    @staticmethod
    def _orders(interp):
        """The id (or class) of every child of both columns, in order."""
        return json.loads(interp.evaljs(
            "(function () {"
            " return JSON.stringify({"
            "  sidebar: $(\"sidebar\").children.map(function (c) {"
            "    return c.id; }),"
            "  main: __mainCol.children.map(function (c) {"
            "    return c.id || c.className; }) }); })()"))

    @staticmethod
    def _arrows(interp, panel_id):
        """The panel's header buttons: class, glyph, disabled flag."""
        return json.loads(interp.evaljs(
            "(function () {"
            " var h2 = $(\"%s\").children[0];"
            " var out = [];"
            " for (var i = 0; i < h2.children.length; i++)"
            "   out.push({ cls: h2.children[i].className || \"\","
            "              glyph: h2.children[i].textContent || \"\","
            "              disabled: !!h2.children[i].disabled });"
            " return JSON.stringify(out); })()" % panel_id))

    @classmethod
    def _click_arrow(cls, interp, panel_id, which):
        """Click the panel's up/down arrow; assert it exists."""
        ok = interp.evaljs(
            "(function () {"
            " var h2 = $(\"%s\").children[0];"
            " for (var i = 0; i < h2.children.length; i++)"
            "   if (h2.children[i].className === \"%s\") {"
            "     h2.children[i].click(); return true; }"
            " return false; })()" % (panel_id, which))
        assert ok is True, "no %s arrow on %s" % (which, panel_id)

    @classmethod
    def _disabled(cls, interp, panel_id, which):
        """Whether the panel's up/down arrow is disabled."""
        for arrow in cls._arrows(interp, panel_id):
            if arrow["cls"] == which:
                return arrow["disabled"]
        raise AssertionError("no %s arrow on %s" % (which, panel_id))

    def test_every_panel_header_gets_arrows(self):
        """All nine panels get up/down arrows, disabled at the ends.

        The sidebar column ends at the sidebar strip, the main
        column starts at the chart box, and the finding chart and
        the weather report are hidden, so the arrows with no
        visible neighbor in that direction are disabled.
        """
        interp = self._interp()
        self._setup(interp)
        panels = ["panel-observatory", "panel-time", "panel-display",
                  "panel-targets", "panel-workspace", "finding-panel",
                  "summary-panel", "wxreport-panel", "calendar-panel"]
        for panel in panels:
            arrows = self._arrows(interp, panel)
            self.assertEqual([a["cls"] for a in arrows], ["pu", "pd"])
            self.assertEqual(arrows[0]["glyph"], "\u25b2")
            self.assertEqual(arrows[1]["glyph"], "\u25bc")
        self.assertTrue(self._disabled(interp, "panel-observatory", "pu"))
        self.assertFalse(self._disabled(interp, "panel-observatory", "pd"))
        self.assertFalse(self._disabled(interp, "panel-workspace", "pu"))
        self.assertTrue(self._disabled(interp, "panel-workspace", "pd"))
        self.assertTrue(self._disabled(interp, "summary-panel", "pu"))
        self.assertTrue(self._disabled(interp, "calendar-panel", "pd"))
        orders = self._orders(interp)
        self.assertEqual(orders["sidebar"][-1], "sidebar-strip")
        self.assertEqual(orders["main"][0], "chart-box")

    def test_down_arrow_swaps_with_next_visible_panel(self):
        """The sidebar's down arrow moves a panel below the next one.

        Moving the observatory panel down swaps it with the time
        panel; the arrows are refreshed at the new ends, and the
        up arrow moves the panel back to its original slot.
        """
        interp = self._interp()
        self._setup(interp)
        self._click_arrow(interp, "panel-observatory", "pd")
        orders = self._orders(interp)
        self.assertEqual(orders["sidebar"][:2],
                         ["panel-time", "panel-observatory"])
        self.assertTrue(self._disabled(interp, "panel-time", "pu"))
        self.assertFalse(self._disabled(interp, "panel-observatory", "pd"))
        self._click_arrow(interp, "panel-observatory", "pu")
        orders = self._orders(interp)
        self.assertEqual(orders["sidebar"][:3],
                         ["panel-observatory", "panel-time",
                          "panel-display"])
        self.assertTrue(self._disabled(interp, "panel-observatory", "pu"))

    def test_up_arrow_skips_hidden_panels(self):
        """The main column's up arrow skips the hidden panels.

        The calendar's up arrow jumps over the hidden weather
        report to swap with the summary; above the calendar there
        is only the hidden finding chart and the chart box, so its
        up arrow is disabled, and so is the summary's down arrow
        over the hidden weather report.
        """
        interp = self._interp()
        self._setup(interp)
        self._click_arrow(interp, "calendar-panel", "pu")
        orders = self._orders(interp)
        self.assertEqual(orders["main"],
                         ["chart-box", "finding-panel", "calendar-panel",
                          "summary-panel", "wxreport-panel"])
        self.assertTrue(self._disabled(interp, "calendar-panel", "pu"))
        self.assertFalse(self._disabled(interp, "summary-panel", "pu"))
        self.assertTrue(self._disabled(interp, "summary-panel", "pd"))

    def test_showing_a_panel_refreshes_the_arrows(self):
        """Turning the weather report on re-enables the arrows above it.

        The checkbox shows the panel on the next redraw, which
        refreshes the arrows; the summary's up arrow then swaps it
        past the calendar back to its original slot.
        """
        interp = self._interp()
        self._setup(interp)
        self._click_arrow(interp, "calendar-panel", "pu")
        interp.evaljs(
            '$("wxreport-chk").checked = true;'
            '$("wxreport-chk").dispatchEvent({ type: "change" }); 0')
        interp.evaljs("flushTimers(); 0")
        self.assertEqual(interp.evaljs(
            '$("wxreport-panel").style.display'), "")
        self.assertFalse(self._disabled(interp, "summary-panel", "pu"))
        self._click_arrow(interp, "summary-panel", "pu")
        orders = self._orders(interp)
        self.assertEqual(orders["main"],
                         ["chart-box", "finding-panel", "summary-panel",
                          "calendar-panel", "wxreport-panel"])


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class FindingHoverLayoutTests(unittest.TestCase):
    """The finding chart's hover info is split across the two panes.

    The left pane below the star chart (#finding-nearest) shows the
    cursor position and, when a row is available, the nearest
    object's name, angular offset from the cursor, and the basic
    field table (findingBasicTable: RA/Dec, Mags, SpT, Type, From
    target, Bib/Notes, a "-" cell where a field is blank); the right
    pane below the sky image (#finding-details) shows only the record
    expansion (findingDetailTable: Parallax, Distance, Proper motion,
    Radial velocity, Identifiers, a "-" cell where the record lacks
    the field, a single "querying" row while the lookup is in
    flight). clearFindingInfo restores the placeholder in the left
    element and empties the right one. The clock is pinned to
    2026-11-25T19:00Z.
    """

    NOW = int(datetime(2026, 11, 25, 19, 0,
                       tzinfo=timezone.utc).timestamp() * 1000)

    @classmethod
    def setUpClass(cls):
        from test_weather_live import STUBS
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        tz_table = {"UTC": 0, "America/Phoenix": -7}
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      script, re.S).group(1))
        for o in obs:
            tz_table.setdefault(o["timezone"], -7)
        cls.stubs = (STUBS
                     .replace("__ALL_IDS__", json.dumps(all_ids))
                     .replace("__TZ_TABLE__", json.dumps(tz_table))
                     .replace("__SIMBAD_BODY__", json.dumps(""))
                     .replace("__OPEN_METEO_BODY__",
                              json.dumps(json.dumps({}))))
        cls.script = script

    def _interp(self):
        """Page interpreter with the clock pinned at self.NOW."""
        interp = dukpy.JSInterpreter()
        interp.evaljs(self.stubs + ";" + chr(10) + "0")
        interp.evaljs("Date.now = function () { return %d; }; 0"
                      % self.NOW)
        interp.evaljs(self.script + ";" + chr(10) + "0")
        return interp

    def _setup_hover(self, interp):
        """Pinned LBT state with a one-row findingHover for Vega."""
        interp.evaljs(
            "state = normalizeState({"
            " observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: [{\"name\": \"Vega\", \"ra\": \"18:36:56.3\","
            " \"dec\": \"+38:47:01.5\"}],"
            " options: { windows: true, calendar: true, threshold: 2.0,"
            " finding: true },"
            " time_range: { start: '2026-11-25T00:00', days: 1 } });"
            "syncAllControls();"
            "var fc = $('finding-chart');"
            "fc.getContext = function () { return __ctx; }; 0")
        interp.evaljs(
            "findingHover = { center: { raDeg: 279.2347, dec: 38.7837 },"
            " rows: [{ name: 'Vega', raDeg: 279.2347, dec: 38.7837,"
            " v: 0.03, distArcsec: 12.5 }],"
            " radiusArcmin: 7.5, label: 'Vega', centerIdx: 0 }; 0")

    def test_hover_line_under_chart_table_under_image(self):
        """updateFindingInfoAt puts the cursor + nearest line and the
        basic table in #finding-nearest and the expansion table in
        #finding-details.

        The Vega record is the archived sim-id reply
        (fixtures/simbad_vega.txt) parsed by the page's own
        simbadParseRecord.
        """
        interp = self._interp()
        self._setup_hover(interp)
        vega = VEGA_FIXTURE.read_text(encoding="utf-8")
        interp.evaljs(
            "findingDetailsCache['Vega'] = { ok: true, record:"
            " simbadParseRecord(" + json.dumps(vega) + ") }; 0")
        out = interp.evaljs(
            "updateFindingInfoAt({ raDeg: 279.235, dec: 38.784 });"
            ' JSON.stringify({ left: $("finding-nearest").innerHTML,'
            ' right: $("finding-details").innerHTML })')
        info = json.loads(out)
        self.assertIn("Cursor:", info["left"])
        self.assertIn("Nearest: Vega", info["left"])
        self.assertIn("from cursor", info["left"])
        self.assertNotIn("from cursor", info["right"])
        self.assertIn("<table>", info["left"])
        self.assertIn("<table>", info["right"])
        for label in ("RA/Dec", "Mags", "SpT", "Type", "From target",
                      "Bib/Notes"):
            self.assertIn(label, info["left"])
            self.assertNotIn(label, info["right"])
        for label in ("Parallax", "Distance", "Proper motion",
                      "Radial velocity", "Identifiers"):
            self.assertNotIn(label, info["left"])
            self.assertIn(label, info["right"])

    def test_hover_sparse_row_fixed_rows(self):
        """A row with no optional fields still renders every basic
        row in the left pane with a "-" cell, and the right pane shows
        a single "querying" row while the record lookup is in flight,
        then the five expansion rows, all "-", once the record is
        (nonexistent) and cached.
        """
        interp = self._interp()
        self._setup_hover(interp)
        interp.evaljs(
            "findingHover.rows.push({ name: 'faint object',"
            " raDeg: 279.5, dec: 38.7 }); 0")
        out = interp.evaljs(
            "updateFindingInfoAt({ raDeg: 279.5, dec: 38.7 });"
            ' JSON.stringify({ left: $("finding-nearest").innerHTML,'
            ' right: $("finding-details").innerHTML })')
        info = json.loads(out)
        self.assertIn("Nearest: faint object", info["left"])
        for label in ("RA/Dec", "Mags", "SpT", "Type", "From target",
                      "Bib/Notes"):
            self.assertIn(label, info["left"])
        self.assertEqual(info["left"].count("<td>-</td>"), 5,
                         info["left"])
        self.assertNotIn("querying", info["left"])
        self.assertIn("querying Simbad", info["right"])
        self.assertEqual(info["right"].count("<tr>"), 1, info["right"])
        interp.evaljs(
            'findingDetailsCache["faint object"] = { ok: true,'
            " record: null }; 0")
        right2 = interp.evaljs(
            "updateFindingInfoAt({ raDeg: 279.5, dec: 38.7 });"
            ' JSON.stringify($("finding-details").innerHTML)')
        for label in ("Parallax", "Distance", "Proper motion",
                      "Radial velocity", "Identifiers"):
            self.assertIn("<th>%s</th><td>-</td>" % label, right2)
        self.assertNotIn("querying", right2)
        self.assertNotIn("<th>RA/Dec</th>", right2)

    def test_clear_finding_info_split(self):
        """clearFindingInfo restores the placeholder under the chart
        and empties the detail element under the image."""
        interp = self._interp()
        self._setup_hover(interp)
        interp.evaljs(
            "updateFindingInfoAt({ raDeg: 279.235, dec: 38.784 }); 0")
        interp.evaljs("clearFindingInfo(); 0")
        left = interp.evaljs('JSON.stringify($("finding-nearest").innerHTML)')
        right = interp.evaljs('JSON.stringify($("finding-details").innerHTML)')
        self.assertIn(
            "Hover the star chart or the sky image for object info", left)
        self.assertEqual(right, '""')


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class NightOfTests(unittest.TestCase):
    """The Night-of dropdown lists local nights and sets the range to one.

    The clock is pinned to 2026-11-25T19:00Z, local noon of Nov 25 in
    America/Phoenix. The dropdown offers the local day before the
    current local day (tagged "yesterday") through 14 days ahead
    (tagged with the year): 16 nights, each a one-day range starting
    at that local day's midnight. Each label carries the Moon's
    illumination at the night's closing local midnight and the max
    altitude of up to two targets over the night. Choosing a night
    sets the range to that local day; a range whose start is not on
    an offered night leaves "(custom range)" selected, and choosing
    it changes nothing.
    """

    NOW = int(datetime(2026, 11, 25, 19, 0,
                       tzinfo=timezone.utc).timestamp() * 1000)

    @classmethod
    def setUpClass(cls):
        from test_weather_live import STUBS
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      script, re.S).group(1))
        tz_table = {"UTC": 0, "America/Phoenix": -7}
        for o in obs:
            tz_table.setdefault(o["timezone"], -7)
        cls.stubs = (STUBS
                     .replace("__ALL_IDS__", json.dumps(all_ids))
                     .replace("__TZ_TABLE__", json.dumps(tz_table))
                     .replace("__SIMBAD_BODY__", json.dumps(""))
                     .replace("__OPEN_METEO_BODY__",
                              json.dumps(json.dumps({}))))
        cls.script = script

    def _interp(self):
        """Page interpreter with the clock pinned at self.NOW."""
        interp = dukpy.JSInterpreter()
        interp.evaljs(self.stubs + ";" + chr(10) + "0")
        interp.evaljs("Date.now = function () { return %d; }; 0"
                      % self.NOW)
        interp.evaljs(self.script + ";" + chr(10) + "0")
        return interp

    def _set_state(self, interp, start, days):
        """Normalize an LBT state with one target (Vega) into the page."""
        interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: [{\"name\": \"Vega\", \"ra\": \"18:36:56.3\","
            " \"dec\": \"+38:47:01.5\"}],"
            " options: { windows: true, threshold: 2.0 },"
            " time_range: { start: %r, days: %r } }); 0" % (start, days))

    def _options(self, interp):
        """The [value, label] pairs currently in the #night-of select."""
        return json.loads(interp.evaljs(
            "JSON.stringify($('night-of').children.map("
            "function (o) { return [o.value, o.textContent]; }))"))

    @staticmethod
    def _moon_illum(t):
        """Illuminated fraction of the Moon at t from astropy positions.

        The fraction is (1 + cos phi)/2 where phi is the
        Sun-Moon-Earth phase angle; the Sun and Moon geocentric
        positions come from get_sun and get_body.
        """
        geoc = EarthLocation(0 * u.m, 0 * u.m, 0 * u.m)
        moon = get_body("moon", t, location=geoc).cartesian
        sun = get_sun(t).cartesian
        to_sun = sun - moon
        to_earth = -moon
        cos_phi = float(np.dot(to_sun.xyz.value, to_earth.xyz.value)
                        / (np.linalg.norm(to_sun.xyz.value)
                           * np.linalg.norm(to_earth.xyz.value)))
        return (1.0 + cos_phi) / 2.0

    def test_night_of_lists_yesterday_through_fourteen_ahead(self):
        """Sixteen nights are offered, tagged, with moon percent in each."""
        interp = self._interp()
        self._set_state(interp, "2026-11-25T00:00", 2.0)
        interp.evaljs("refreshNightOf(); 0")
        opts = self._options(interp)
        self.assertEqual(opts[0], ["", "(custom range)"])
        vals = [o[0] for o in opts[1:]]
        self.assertEqual(
            vals, ["2026-11-%02d" % d for d in range(24, 31)]
            + ["2026-12-%02d" % d for d in range(1, 10)])
        labels = [o[1] for o in opts[1:]]
        self.assertIn("yesterday", labels[0])
        self.assertIn("tonight", labels[1])
        self.assertIn("tomorrow", labels[2])
        for lab in labels[3:]:
            self.assertIn("2026", lab)
        for lab in labels:
            self.assertRegex(lab, r" moon \d+%$")
            self.assertRegex(lab, "Vega \\d+°")

    def test_night_of_change_sets_range_to_that_night(self):
        """Choosing a night sets the range to that local day, one day long."""
        interp = self._interp()
        self._set_state(interp, "2026-11-25T00:00", 2.0)
        interp.evaljs("refreshNightOf(); 0")
        interp.evaljs(
            "$('night-of').value = '2026-11-26';"
            "$('night-of').dispatchEvent({ type: 'change' }); 0")
        got = json.loads(interp.evaljs(
            "JSON.stringify({ start: state.time_range.start,"
            " days: state.time_range.days,"
            " ctl: $('start').value })"))
        self.assertEqual(got["start"], "2026-11-26 00:00")
        self.assertEqual(got["days"], 1)
        self.assertEqual(got["ctl"], "2026-11-26 00:00")

    def test_night_of_change_ignores_custom_range(self):
        """Choosing "(custom range)" leaves the range unchanged."""
        interp = self._interp()
        self._set_state(interp, "2026-01-15T00:00", 2.0)
        interp.evaljs("refreshNightOf(); 0")
        interp.evaljs(
            "$('night-of').value = '';"
            "$('night-of').dispatchEvent({ type: 'change' }); 0")
        got = json.loads(interp.evaljs(
            "JSON.stringify({ start: state.time_range.start,"
            " days: state.time_range.days })"))
        self.assertEqual(got["start"], "2026-01-15T00:00")
        self.assertEqual(got["days"], 2.0)

    def test_night_of_selects_the_shown_night(self):
        """The night the range starts on is selected; none for custom."""
        interp = self._interp()
        self._set_state(interp, "2026-11-25T00:00", 1.0)
        interp.evaljs("refreshNightOf(); 0")
        self.assertEqual(
            json.loads(interp.evaljs("JSON.stringify($('night-of').value)")),
            "2026-11-25")
        self._set_state(interp, "2026-01-15T00:00", 2.0)
        interp.evaljs("refreshNightOf(); 0")
        self.assertEqual(
            json.loads(interp.evaljs("JSON.stringify($('night-of').value)")),
            "")

    def test_night_of_labels_match_astropy(self):
        """Tonight's label: moon percent and Vega max alt agree with astropy.

        The moon percent is at the night's closing local midnight
        (2026-11-26 07:00 UTC at America/Phoenix); the Vega altitude is
        the max over the night sampled at the same half-hour grid the
        page uses, against astropy's refraction-free apparent altitude.
        """
        interp = self._interp()
        self._set_state(interp, "2026-11-25T00:00", 2.0)
        interp.evaljs("refreshNightOf(); 0")
        opts = self._options(interp)
        tonight = [o[1] for o in opts if o[0] == "2026-11-25"][0]
        m = re.search(r"moon (\d+)%$", tonight)
        self.assertIsNotNone(m)
        illum = self._moon_illum(Time("2026-11-26 07:00:00"))
        self.assertAlmostEqual(int(m.group(1)), illum * 100, delta=1.0)
        am = re.search(r"Vega (\d+)°", tonight)
        self.assertIsNotNone(am)
        loc = EarthLocation.from_geodetic(
            LBT.lon * u.deg, LBT.lat * u.deg, LBT.elevation_m)
        best = -90.0
        for i in range(49):
            ts = Time("2026-11-25 07:00:00") + i * 1800 * u.s
            aa = AltAz(obstime=ts, location=loc)
            aa.refraction = 0 * u.deg
            alt = float(VEGA.transform_to(aa).alt.deg)
            if alt > best:
                best = alt
        self.assertAlmostEqual(int(am.group(1)), best, delta=0.5)


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class WorkspacePresetTests(unittest.TestCase):
    """Named workspace presets carry a save stamp and rich labels.

    Saved entries are name -> {state, saved}, where saved is the UTC
    wall time the workspace was saved. The dropdown label of such an
    entry is its name, the saved state's observatory, the target
    count, and the save time; an entry saved by an older version as a
    bare state keeps its name only as the label. Load accepts both
    shapes; Delete removes the entry.
    """

    WS = "airmass-gui:workspaces"

    @classmethod
    def setUpClass(cls):
        from test_weather_live import STUBS
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      script, re.S).group(1))
        tz_table = {"UTC": 0, "America/Phoenix": -7}
        for o in obs:
            tz_table.setdefault(o["timezone"], -7)
        cls.stubs = (STUBS
                     .replace("__ALL_IDS__", json.dumps(all_ids))
                     .replace("__TZ_TABLE__", json.dumps(tz_table))
                     .replace("__SIMBAD_BODY__", json.dumps(""))
                     .replace("__OPEN_METEO_BODY__",
                              json.dumps(json.dumps({}))))
        cls.script = script

    def _interp(self):
        """Page interpreter with the clock pinned at 2026-11-25T19:00Z."""
        interp = dukpy.JSInterpreter()
        interp.evaljs(self.stubs + ";" + chr(10) + "0")
        interp.evaljs(
            "Date.now = function () {"
            " return " + str(int(datetime(2026, 11, 25, 19, 0,
                                          tzinfo=timezone.utc)
                                     .timestamp() * 1000)) + "; }; 0")
        interp.evaljs(self.script + ";" + chr(10) + "0")
        return interp

    def _set_state(self, interp):
        """Normalize an LBT state with one target (Vega) into the page."""
        interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: [{\"name\": \"Vega\", \"ra\": \"18:36:56.3\","
            " \"dec\": \"+38:47:01.5\"}],"
            " options: { windows: true, threshold: 2.0 },"
            " time_range: { start: '2026-11-25T00:00', days: 1.0 } }); 0")

    def _ws(self, interp):
        """The parsed name -> entry map currently in browser storage."""
        raw = interp.evaljs("localStorage.getItem('%s')" % self.WS)
        return json.loads(raw) if raw else {}

    def _ws_options(self, interp):
        """The [value, label] pairs currently in the #ws-select select."""
        return json.loads(interp.evaljs(
            "JSON.stringify($('ws-select').children.map("
            "function (o) { return [o.value, o.textContent]; }))"))

    def _seed(self, interp):
        """Seed storage with a v2 entry (Alpha) and a legacy entry (Beta)."""
        interp.evaljs(
            "localStorage.setItem('%s', JSON.stringify({"
            " Alpha: { state: { observatory: {"
            " name: 'Kitt Peak (Mayall 4m)', lat: 31.96333,"
            " lon: -111.6, elevation_m: 2120,"
            " timezone: 'America/Phoenix' },"
            " objects: [{ name: 'a', ra: '5:55:40.8',"
            " dec: '+38:47:01.28' },"
            " { name: 'b', ra: '18:36:56.3', dec: '+38:47:01.5' }],"
            " time_range: { start: '2026-11-25T00:00', days: 1.0 } },"
            " saved: '2026-11-01 05:00 UTC' },"
            " Beta: { observatory: { name: 'X', lat: 10.0,"
            " lon: 20.0, elevation_m: 100.0, timezone: 'UTC' },"
            " objects: [], time_range: {"
            " start: '2026-11-25T00:00', days: 1.0 } }"
            "})); 0" % self.WS)

    def test_save_stores_state_with_saved_stamp(self):
        """Saving stores {state, saved} and the label shows the details."""
        interp = self._interp()
        self._set_state(interp)
        interp.evaljs("$('ws-save-btn').click(); 0")
        ws = self._ws(interp)
        self.assertIn("Smoke WS", ws)
        entry = ws["Smoke WS"]
        self.assertIn("state", entry)
        self.assertEqual(entry["state"]["objects"][0]["name"], "Vega")
        self.assertRegex(entry["saved"],
                         r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC$")
        opts = self._ws_options(interp)
        self.assertEqual(opts[0][0], "Smoke WS")
        self.assertRegex(
            opts[0][1],
            r"^Smoke WS · Mount Graham \(LBT\) ·"
            r" 1 target · saved \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC$")

    def test_labels_describe_v2_and_legacy_entries(self):
        """v2 entries are described; legacy bare-state entries keep the name."""
        interp = self._interp()
        self._seed(interp)
        interp.evaljs("refreshWsSelect(); 0")
        by_val = dict(self._ws_options(interp))
        self.assertEqual(
            by_val["Alpha"],
            "Alpha · Kitt Peak (Mayall 4m) · 2 targets"
            " · saved 2026-11-01 05:00 UTC")
        self.assertEqual(by_val["Beta"], "Beta")

    def test_load_unwraps_v2_entry(self):
        """Loading a v2 entry loads its inner state."""
        interp = self._interp()
        self._seed(interp)
        interp.evaljs(
            "refreshWsSelect();"
            "$('ws-select').value = 'Alpha';"
            "$('ws-load-btn').click(); 0")
        got = json.loads(interp.evaljs(
            "JSON.stringify({ obs: state.observatory.name,"
            " names: state.objects.map(function (o) { return o.name; }),"
            " status: $('status').textContent })"))
        self.assertEqual(got["obs"], "Kitt Peak (Mayall 4m)")
        self.assertEqual(got["names"], ["a", "b"])
        self.assertIn("Alpha", got["status"])

    def test_load_accepts_legacy_bare_state(self):
        """Loading an entry saved as a bare state still works."""
        interp = self._interp()
        interp.evaljs(
            "localStorage.setItem('%s', JSON.stringify({"
            " Beta: { observatory: { name: 'X', lat: 10.0,"
            " lon: 20.0, elevation_m: 100.0, timezone: 'UTC' },"
            " objects: [{ name: 'only', ra: '7.5', dec: '-45' }],"
            " time_range: { start: '2026-11-25T00:00', days: 1.0 } }"
            "})); 0" % self.WS)
        interp.evaljs(
            "refreshWsSelect();"
            "$('ws-select').value = 'Beta';"
            "$('ws-load-btn').click(); 0")
        got = json.loads(interp.evaljs(
            "JSON.stringify({ obs: state.observatory.name,"
            " names: state.objects.map(function (o) { return o.name; }) })"))
        self.assertEqual(got["obs"], "X")
        self.assertEqual(got["names"], ["only"])

    def test_delete_removes_the_entry(self):
        """Delete drops the entry and refreshes the dropdown."""
        interp = self._interp()
        self._seed(interp)
        interp.evaljs(
            "refreshWsSelect();"
            "$('ws-select').value = 'Alpha';"
            "$('ws-delete-btn').click(); 0")
        ws = self._ws(interp)
        self.assertNotIn("Alpha", ws)
        self.assertIn("Beta", ws)
        self.assertEqual([o[0] for o in self._ws_options(interp)],
                         ["Beta"])


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class MoonRiseSetTests(unittest.TestCase):
    """The per-night list shows the Moon's rise and set and the day's
    maximum hourly rain chance.

    moonRiseSet scans the topocentric Moon's altitude at five-minute
    steps over a local day's part of the chart and bisects each
    horizon crossing; summaryData adds one entry per day, and
    renderSummary lists the events in local time (or notes that the
    Moon stays up or down for the whole window) and, with weather on,
    appends the day's maximum hourly rain chance and the average
    clear-sky fraction.
    """

    NOW = int(datetime(2026, 11, 25, 19, 0,
                       tzinfo=timezone.utc).timestamp() * 1000)

    @classmethod
    def setUpClass(cls):
        from test_weather_live import STUBS
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      script, re.S).group(1))
        tz_table = {"UTC": 0, "America/Phoenix": -7}
        for o in obs:
            tz_table.setdefault(o["timezone"], -7)
        cls.stubs = (STUBS
                     .replace("__ALL_IDS__", json.dumps(all_ids))
                     .replace("__TZ_TABLE__", json.dumps(tz_table))
                     .replace("__SIMBAD_BODY__", json.dumps(""))
                     .replace("__OPEN_METEO_BODY__",
                              json.dumps(json.dumps({}))))
        cls.script = script
        cls.js = dukpy.JSInterpreter()
        cls.js.evaljs(_extract_core(cls.page))

    def _interp(self):
        """Page interpreter with the clock pinned at self.NOW."""
        interp = dukpy.JSInterpreter()
        interp.evaljs(self.stubs + ";" + chr(10) + "0")
        interp.evaljs("Date.now = function () { return %d; }; 0"
                      % self.NOW)
        interp.evaljs(self.script + ";" + chr(10) + "0")
        return interp

    def _set_state(self, interp, start, days, weather):
        """Normalize an LBT state with one target (Vega) into the page."""
        interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: [{\"name\": \"Vega\", \"ra\": \"18:36:56.3\","
            " \"dec\": \"+38:47:01.5\"}],"
            " options: { windows: true, threshold: 2.0,"
            " weather: %s },"
            " time_range: { start: %r, days: %r } }); 0"
            % ("true" if weather else "false", start, days))

    @staticmethod
    def _astro_moon_crossings_s(day_start_ms, n_steps=288, step_s=300.0):
        """First Moon horizon crossings within a day, from astropy.

        The topocentric apparent Moon (get_body at the observer's
        location, refraction zeroed because the page applies none)
        is sampled at step_s seconds from the day's start, the
        page's five-minute grid; each horizon sign change is refined
        by linear interpolation between its bracketing samples.

        Parameters
        ----------
        day_start_ms : int
            Day start in UTC epoch ms.
        n_steps : int, default 288
            Number of step_s steps spanning the day.
        step_s : float, default 300
            Grid spacing in seconds.

        Returns
        -------
        (rise_s, set_s, alts) : (float or None, float or None,
        list of float)
            First rise and first set as seconds from the day start
            (None when absent), and the altitude samples in degrees.
        """
        loc = EarthLocation.from_geodetic(
            LBT.lon * u.deg, LBT.lat * u.deg, LBT.elevation_m)
        t0 = Time(day_start_ms / 1000.0, format="unix")
        alts = []
        for i in range(n_steps + 1):
            t = t0 + i * step_s * u.s
            aa = AltAz(obstime=t, location=loc)
            aa.refraction = 0 * u.deg
            alts.append(float(
                SkyCoord(get_body("moon", t, loc)).transform_to(aa).alt.deg))
        rise_s = None
        set_s = None
        for i in range(1, n_steps + 1):
            a = alts[i - 1]
            b = alts[i]
            if a < 0.0 and b >= 0.0 and rise_s is None:
                rise_s = (i - 1 + a / (a - b)) * step_s
            elif a > 0.0 and b <= 0.0 and set_s is None:
                set_s = (i - 1 + a / (a - b)) * step_s
        return rise_s, set_s, alts

    def test_moon_rise_set_matches_astropy(self):
        """LBT local days 2026-11-25 through 28: page crossings agree
        with astropy within five minutes.

        Days are the local (America/Phoenix) days at 07:00 UTC
        boundaries. Interior crossings (more than five minutes from
        the day boundaries) are compared directly; with no interior
        crossing and both endpoint altitudes more than 0.5 deg from
        the horizon, the page's events and upAll/downAll flags must
        match astropy's.
        """
        for day in range(25, 29):
            day_start_ms = int(
                datetime(2026, 11, day, 7, 0, tzinfo=timezone.utc)
                .timestamp() * 1000)
            rise_s, set_s, alts = self._astro_moon_crossings_s(day_start_ms)
            n = len(alts) - 1
            page = json.loads(self.js.evaljs(
                "JSON.stringify(moonRiseSet(%r, %r, %r, %r, %r))"
                % (day_start_ms, day_start_ms + n * 300 * 1000,
                   LBT.lat, LBT.lon, LBT.elevation_m)))
            edge_margin = min(abs(alts[0]), abs(alts[-1])) > 0.5
            for kind, astro in (("rise", rise_s), ("set", set_s)):
                if astro is not None and 300 < astro < n * 300 - 300:
                    self.assertIsNotNone(
                        page[kind], "day %d: page missed %s" % (day, kind))
                    self.assertLessEqual(
                        abs(page[kind] - (day_start_ms + astro * 1000)),
                        5 * 60 * 1000, "day %d %s" % (day, kind))
                elif astro is None and edge_margin:
                    self.assertIsNone(
                        page[kind], "day %d: phantom %s" % (day, kind))
            if rise_s is None and set_s is None and edge_margin:
                self.assertEqual(page["upAll"], alts[0] > 0.0)
                self.assertEqual(page["downAll"], alts[0] < 0.0)

    def test_summary_lists_moon_events_and_rain(self):
        """The per-night list carries moonset/moonrise in local time
        and the day's maximum hourly rain chance.

        Local day 2026-11-26 at the LBT: near the 2026-11-24/25 full
        Moon, the Moon's only crossing that day is a set in the
        early morning (astropy cross-check). The weather cache is
        seeded with a 30-minute grid at 50% cloud cover and
        pp = (7 i) % 101, so the day's maximum rain chance is 99%.
        """
        interp = self._interp()
        self._set_state(interp, "2026-11-26T00:00", 1, True)
        day_start_ms = int(
            datetime(2026, 11, 26, 7, 0, tzinfo=timezone.utc)
            .timestamp() * 1000)
        hours = [{"ms": day_start_ms + i * 1800000, "cc": 50.0,
                  "pp": float((i * 7) % 101), "prec": None, "wc": None}
                 for i in range(48)]
        interp.evaljs("__wxCache = { key: 'seed', fetchedMs: 0,"
                      " hours: %s }; 0" % json.dumps(hours))
        data = json.loads(interp.evaljs("JSON.stringify(summaryData())"))
        self.assertEqual(len(data["days"]), 1)
        self.assertEqual(data["weather"][0]["clear"], 50.0)
        self.assertEqual(data["weather"][0]["rain"], 99.0)
        mo = data["moon"][0]
        rise_s, set_s, alts = self._astro_moon_crossings_s(day_start_ms)
        n = len(alts) - 1
        edge_margin = min(abs(alts[0]), abs(alts[-1])) > 0.5
        for kind, astro in (("rise", rise_s), ("set", set_s)):
            if astro is not None and 300 < astro < n * 300 - 300:
                self.assertIsNotNone(mo[kind])
                self.assertLessEqual(
                    abs(mo[kind] - (day_start_ms + astro * 1000)),
                    5 * 60 * 1000)
            elif astro is None and edge_margin:
                self.assertIsNone(mo[kind])
        interp.evaljs("renderSummary(); 0")
        html = json.loads(interp.evaljs(
            'JSON.stringify($("summary").innerHTML)'))
        self.assertIn("rain 99%", html)
        self.assertIn("50% clear", html)
        if mo["set"] is not None:
            dt = datetime.fromtimestamp(
                mo["set"] / 1000.0, tz=timezone(timedelta(hours=-7)))
            self.assertIn("moonset %02d:%02d" % (dt.hour, dt.minute), html)
        if mo["rise"] is None:
            self.assertNotIn("moonrise", html)

    def test_summary_moon_without_weather(self):
        """The per-night Moon events show with the Weather off too."""
        interp = self._interp()
        self._set_state(interp, "2026-11-26T00:00", 1, False)
        data = json.loads(interp.evaljs("JSON.stringify(summaryData())"))
        self.assertIsNone(data["weather"])
        self.assertEqual(len(data["moon"]), 1)
        self.assertIn("rise", data["moon"][0])
        self.assertIn("set", data["moon"][0])
        self.assertIn("upAll", data["moon"][0])
        self.assertIn("downAll", data["moon"][0])


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class WxReportTests(unittest.TestCase):
    """The Weather report panel shows the current conditions and
    eight-day outlook for the observatory.

    The checkbox in the Display panel is off by default. When on,
    the page asks open-meteo's forecast API for the current
    conditions and a daily forecast covering the observatory's
    local today through seven days out, and renders a
    current-conditions line, one row per day (condition, min and
    max temperature, maximum rain chance, mean cloud cover, maximum
    wind, and the Moon's illumination at the day's closing local
    midnight), and links to the underlying weather services, including the
Meteoblue meteogram for the site's coordinates. The
    reply is cached per observatory and reused for one hour.

    The render test uses the reply recorded 2026-11-25 from
    https://api.open-meteo.com/v1/forecast?latitude=32.70131
    &longitude=-109.88906&current=temperature_2m,wind_speed_10m,
    cloud_cover,precipitation_probability,weather_code&daily=
    temperature_2m_max,temperature_2m_min,precipitation_
    probability_max,cloudcover_mean,wind_speed_10m_max,weather_code
    &timezone=America/Phoenix&start_date=2026-11-25&end_date=
    2026-12-02, saved unedited in
    tests/fixtures/open_meteo_daily_20261125.json and seeded
    directly into the report cache, so no network is needed.
    """

    NOW = int(datetime(2026, 11, 25, 19, 0,
                       tzinfo=timezone.utc).timestamp() * 1000)
    FIXTURE = ROOT / "tests" / "fixtures" / "open_meteo_daily_20261125.json"
    EXPECTED_URL = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=32.70131&longitude=-109.88906"
        "&current=temperature_2m,wind_speed_10m,cloud_cover,"
        "precipitation_probability,weather_code"
        "&daily=temperature_2m_max,temperature_2m_min,"
        "precipitation_probability_max,cloudcover_mean,"
        "wind_speed_10m_max,weather_code"
        "&timezone=America%2FPhoenix"
        "&start_date=2026-11-25&end_date=2026-12-02")
    WMO_ICON = {0: "\u2600", 1: "\U0001F324", 2: "\u26c5",
               3: "\u2601", 45: "\U0001F32b", 48: "\U0001F32b",
               51: "\U0001F326", 53: "\U0001F326", 55: "\U0001F326",
               56: "\U0001F327", 57: "\U0001F327", 61: "\U0001F327",
               63: "\U0001F327", 65: "\U0001F327", 66: "\U0001F327",
               67: "\U0001F327", 71: "\U0001F328", 73: "\U0001F328",
               75: "\U0001F328", 77: "\U0001F328", 80: "\U0001F327",
               81: "\U0001F327", 82: "\U0001F327", 85: "\U0001F328",
               86: "\U0001F328", 95: "\u26c8", 96: "\u26c8",
               99: "\u26c8"}
    WMO = {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
           3: "Overcast", 45: "Fog", 48: "Fog", 51: "Drizzle",
           53: "Drizzle", 55: "Drizzle", 56: "Freezing drizzle",
           57: "Freezing drizzle", 61: "Rain", 63: "Rain", 65: "Rain",
           66: "Freezing rain", 67: "Freezing rain", 71: "Snow",
           73: "Snow", 75: "Snow", 77: "Snow", 80: "Rain showers",
           81: "Rain showers", 82: "Rain showers", 85: "Snow showers",
           86: "Snow showers", 95: "Thunderstorm",
           96: "Thunderstorm with hail", 99: "Thunderstorm with hail"}
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    @classmethod
    def setUpClass(cls):
        from test_weather_live import STUBS
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      script, re.S).group(1))
        tz_table = {"UTC": 0, "America/Phoenix": -7}
        for o in obs:
            tz_table.setdefault(o["timezone"], -7)
        cls.stubs = (STUBS
                     .replace("__ALL_IDS__", json.dumps(all_ids))
                     .replace("__TZ_TABLE__", json.dumps(tz_table))
                     .replace("__SIMBAD_BODY__", json.dumps(""))
                     .replace("__OPEN_METEO_BODY__",
                              json.dumps(json.dumps({}))))
        cls.script = script
        cls.body = json.loads(cls.FIXTURE.read_text(encoding="utf-8"))

    def _interp(self):
        """Page interpreter with the clock pinned at self.NOW."""
        interp = dukpy.JSInterpreter()
        interp.evaljs(self.stubs + ";" + chr(10) + "0")
        interp.evaljs("Date.now = function () { return %d; }; 0"
                      % self.NOW)
        interp.evaljs(self.script + ";" + chr(10) + "0")
        return interp

    def _set_state(self, interp, wx_report):
        """Normalize an LBT state with the report option set."""
        interp.evaljs(
            "state = normalizeState({ observatory: {"
            " name: 'Mount Graham (LBT)', lat: 32.70131,"
            " lon: -109.88906, elevation_m: 3221.0,"
            " timezone: 'America/Phoenix',"
            " preset: 'Mount Graham (LBT)' },"
            " objects: [],"
            " options: { wxReport: %s },"
            " time_range: { start: '2026-11-25T00:00', days: 1 } }); 0"
            % ("true" if wx_report else "false"))

    @staticmethod
    def _moon_illum(t):
        """Illuminated fraction of the Moon at t from astropy.

        The fraction is (1 + cos phi)/2 where phi is the
        Sun-Moon-Earth phase angle; the Sun and Moon geocentric
        positions come from get_sun and get_body.
        """
        t = Time(t)
        geoc = EarthLocation(0 * u.m, 0 * u.m, 0 * u.m)
        moon = get_body("moon", t, location=geoc).cartesian
        sun = get_sun(t).cartesian
        to_sun = sun - moon
        to_earth = -moon
        cos_phi = float(np.dot(to_sun.xyz.value, to_earth.xyz.value)
                        / (np.linalg.norm(to_sun.xyz.value)
                           * np.linalg.norm(to_earth.xyz.value)))
        return (1.0 + cos_phi) / 2.0

    def test_daily_request_url(self):
        """The request targets the daily API for eight local days.

        With the clock pinned at 2026-11-25T19:00Z (local noon of
        Nov 25 at the LBT), the range is 2026-11-25 through
        2026-12-02 and the time zone is percent-encoded, as in
        weatherRequest.
        """
        interp = self._interp()
        self._set_state(interp, True)
        url = interp.evaljs("wxDailyRequest(state.observatory).url")
        self.assertEqual(url, self.EXPECTED_URL)

    def test_wmo_icon(self):
        """wmoIcon maps WMO codes to their display glyphs.

        Codes without a glyph, and null, give an empty string so the
        condition cell degrades to text only.
        """
        interp = self._interp()
        checks = {0: "\u2600", 1: "\U0001F324", 2: "\u26c5",
                  3: "\u2601", 45: "\U0001F32b", 51: "\U0001F326",
                  61: "\U0001F327", 71: "\U0001F328", 95: "\u26c8",
                  99: "\u26c8"}
        for code, icon in checks.items():
            self.assertEqual(interp.evaljs("wmoIcon(%d)" % code), icon)
        self.assertEqual(interp.evaljs("wmoIcon(null)"), "")
        self.assertEqual(interp.evaljs("wmoIcon(123)"), "")

    def test_report_hidden_by_default(self):
        """The panel is off by default and hidden after a sync."""
        interp = self._interp()
        self._set_state(interp, False)
        self.assertEqual(interp.evaljs("state.options.wxReport"), False)
        interp.evaljs("syncAllControls(); 0")
        self.assertEqual(interp.evaljs(
            '$("wxreport-panel").style.display'), "none")
        self.assertFalse(interp.evaljs('$("wxreport-chk").checked'))

    def test_report_renders_recorded_reply(self):
        """The panel shows the recorded reply's values, DOM-built.

        The report cache is seeded with the unedited 2026-11-25
        reply; the current line and every day's row are checked
        against the fixture, the Moon column against astropy's
        phase, and the footer links are checked one by one.
        """
        interp = self._interp()
        self._set_state(interp, True)
        interp.evaljs("__wxReportCache = { key: 'seed', fetchedMs: %d,"
                      " body: %s }; 0" % (self.NOW, json.dumps(self.body)))
        interp.evaljs("renderWxReport(); 0")
        self.assertEqual(interp.evaljs(
            '$("wxreport-panel").style.display'), "")
        report = json.loads(interp.evaljs("""(function () {
          var out = { rows: [], links: [], text: "" };
          var walk = function (el) {
            if (el.textContent) out.text += el.textContent + " ";
            if (el.tagName === "TR")
              out.rows.push((el.children || []).map(function (c) {
                return c.textContent; }));
            if (el.tagName === "A")
              out.links.push(String(el.href || ""));
            (el.children || []).forEach(walk);
          };
          walk($("wxreport"));
          return JSON.stringify(out);
        })()"""))
        rows = report["rows"]
        self.assertEqual(rows[0], ["Date", "Condition", "Tmin", "Tmax",
                                   "Rain %", "Cloud %", "Wind max",
                                   "Moon %"])
        self.assertEqual(len(rows), 9)
        cur = self.body["current"]
        self.assertIn("Now:", report["text"])
        c = cur["temperature_2m"]
        self.assertIn("%s \u00b0C (%s \u00b0F)" % ("%.1f" % c,
                                                  "%.1f" % (c * 9 / 5 + 32)),
                      report["text"])
        self.assertIn("wind %s km/h" % ("%.1f" % cur["wind_speed_10m"]),
                      report["text"])
        self.assertIn("cloud %d%%" % cur["cloud_cover"], report["text"])
        self.assertIn("rain %d%%" % cur["precipitation_probability"],
                      report["text"])
        self.assertIn(self.WMO[cur["weather_code"]], report["text"])
        d = self.body["daily"]
        for i, day in enumerate(d["time"]):
            y, mo, dd = day.split("-")
            row = rows[1 + i]
            self.assertEqual(row[0], self.MONTHS[int(mo) - 1] + " " + dd)
            self.assertEqual(row[1],
                             self.WMO_ICON[d["weather_code"][i]] + " " +
                             self.WMO[d["weather_code"][i]])
            self.assertEqual(row[2],
                             "%.1f \u00b0C" % d["temperature_2m_min"][i])
            self.assertEqual(row[3],
                             "%.1f \u00b0C" % d["temperature_2m_max"][i])
            self.assertEqual(row[4],
                             "%d%%" % d["precipitation_probability_max"][i])
            self.assertEqual(row[5], "%d%%" % d["cloudcover_mean"][i])
            self.assertEqual(row[6],
                             "%.1f km/h" % d["wind_speed_10m_max"][i])
            closing = datetime(int(y), int(mo), int(dd), 7, 0,
                               tzinfo=timezone.utc) + timedelta(days=1)
            self.assertEqual(int(row[7].rstrip("%")),
                             int(round(self._moon_illum(closing) * 100)))
        self.assertEqual(report["links"], [
            "https://open-meteo.com/", "https://www.windy.com/",
            "https://www.weather.gov/", "https://www.wunderground.com/",
            "https://www.meteoblue.com/en/weather/forecast"
            "?latitude=32.70131&longitude=-109.88906"])

    def test_refresh_fetches_once_then_reuses(self):
        """wxReportRefresh fetches once and serves later calls from cache.

        The stub fetch is replaced with a counter that returns the
        recorded reply in the live shape; two refreshes in a row
        issue one request for the exact daily URL, and the panel is
        rendered from the cache.
        """
        interp = self._interp()
        self._set_state(interp, True)
        interp.evaljs("var __wxrepBody = " + json.dumps(self.body) + "; 0")
        interp.evaljs("""
          var __wxrepFetches = [];
          fetch = function (url, opts) {
            __wxrepFetches.push(String(url));
            return Promise.resolve({
              ok: true, status: 200,
              text: function () {
                return Promise.resolve(JSON.stringify(__wxrepBody)); }
            });
          };
          0""")
        interp.evaljs("wxReportRefresh(); 0")
        interp.evaljs("Promise.resolve(); 0")
        self.assertEqual(json.loads(
            interp.evaljs("JSON.stringify(__wxrepFetches)")),
            [self.EXPECTED_URL])
        interp.evaljs("wxReportRefresh(); 0")
        interp.evaljs("Promise.resolve(); 0")
        self.assertEqual(json.loads(
            interp.evaljs("JSON.stringify(__wxrepFetches)")),
            [self.EXPECTED_URL])
        text = json.loads(interp.evaljs("""(function () {
          var out = "";
          var walk = function (el) {
            out += el.textContent || "";
            (el.children || []).forEach(walk);
          };
          walk($("wxreport"));
          return JSON.stringify(out);
        })()"""))
        self.assertIn("Now:", text)



@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class JsCoreNumericTests(unittest.TestCase):
    """Evaluate the page's JS core in Duktape and compare against astropy."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.page = build_site(new_workspace(), Path(cls.dir.name) / "index.html").read_text(encoding="utf-8")
        cls.js = dukpy.JSInterpreter()
        cls.js.evaljs(_extract_core(cls.page))
        cls.js.evaljs(_extract_fn(cls.page, "sexagesimalString"))

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

    def test_blend_hex(self):
        """blendHex agrees with the Python _blend_hex for random colors."""
        self.assertEqual(self._js("blendHex('#123456', 0, '#fedcba')"),
                         "#fedcba")
        self.assertEqual(self._js("blendHex('#123456', 1, '#fedcba')"),
                         "#123456")
        rng = random.Random(42)
        digits = "0123456789abcdef"
        for _ in range(25):
            fg = "#" + "".join(rng.choice(digits) for _ in range(6))
            bg = "#" + "".join(rng.choice(digits) for _ in range(6))
            alpha = round(rng.random(), 2)
            expected = _blend_hex(fg, alpha, bg)
            got = self._js("blendHex('%s', %r, '%s')" % (fg, alpha, bg))
            for i in (1, 3, 5):
                self.assertAlmostEqual(
                    int(got[i : i + 2], 16), int(expected[i : i + 2], 16),
                    delta=1.0)
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

    def test_parse_dec_negative_zero(self):
        """parseDec takes the sign from the leading '-', not from -0.

        Sexagesimal decs between 0 and -1 deg carry a degree part of
        -0; a "deg < 0" sign check reads that as positive. Simbad
        prints delta Ori as -00 17 56.74.
        """
        cases = [
            ("-00:17:57", -(17 / 60 + 57 / 3600)),
            ("-00:30:00", -0.5),
            ("-0:00:48", -48 / 3600),
            ("-00d30m00s", -0.5),
            ("+00:30:00", 0.5),
            ("-01:00:00", -1.0),
            ("-0.5", -0.5),
        ]
        for entry, expected in cases:
            self.assertAlmostEqual(
                self._js("parseDec(%s)" % json.dumps(entry)), expected, delta=1e-9)

    def test_parse_dec_simbad_round_trip(self):
        """Simbad-formatted decs (sign + sexagesimalString) parse back.

        A lookup fills the dec field as sign plus sexagesimalString of
        the record's dec; parseDec must return the same value to 1e-5
        deg, including decs between 0 and -1 deg.
        """
        for dec in (-0.2990944, -0.5, -0.0125, -1.0):
            formatted = self._js(
                "(function (d) { return (d < 0 ? '-' : '+') + "
                "sexagesimalString(Math.abs(d), 2); })("
                + repr(dec) + ")")
            got = self._js("parseDec(%s)" % json.dumps(formatted))
            self.assertAlmostEqual(got, dec, delta=1e-5)

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
        self.assertAlmostEqual(got["plxErr"], float(expected["plxErr"]))
        self.assertAlmostEqual(got["pmRa"], float(expected["pmRa"]))
        self.assertAlmostEqual(got["pmDec"], float(expected["pmDec"]))
        self.assertAlmostEqual(got["pmRaErr"], float(expected["pmRaErr"]))
        self.assertAlmostEqual(got["pmDecErr"], float(expected["pmDecErr"]))
        self.assertAlmostEqual(got["rv"], float(expected["rv"]))
        self.assertEqual(got["idents"], expected["idents"])
        self.assertEqual(got["idTotal"], expected["idTotal"])
        self.assertEqual(set(got["fluxes"]), set(expected["fluxes"]))
        for band, ref in expected["fluxes"].items():
            self.assertAlmostEqual(got["fluxes"][band]["value"], ref["value"])
            if ref["err"] is None:
                self.assertIsNone(got["fluxes"][band]["err"])
            else:
                self.assertAlmostEqual(got["fluxes"][band]["err"], ref["err"])
        no_coords = "C.D.S.  -  SIMBAD4\n\nSome object\n----\nno coordinates here"
        self.assertIsNone(self._js(f"simbadParseRecord({json.dumps(no_coords)})"))
        minimal = ("C.D.S.  -  SIMBAD4\n\nFaint star\n--------\n"
                   "Coordinates(ICRS,ep=J2000,eq=2000): 01 02 03.00000  +10 11 12.0000\n")
        got2 = self._js(f"simbadParseRecord({json.dumps(minimal)})")
        self.assertIsNone(got2["mag"])
        self.assertIsNone(got2["spec"])
        self.assertIsNone(got2["plx"])
        self.assertIsNone(got2["plxErr"])
        self.assertIsNone(got2["pmRa"])
        self.assertIsNone(got2["pmDec"])
        self.assertIsNone(got2["rv"])
        self.assertEqual(got2["fluxes"], {})
        self.assertEqual(got2["idents"], [])
        self.assertIsNone(got2["idTotal"])

    def test_parallax_distance(self):
        """parallaxDistance inverts mas to pc with error propagation; formatDistance picks units."""
        d = self._js("parallaxDistance(130.23, 0.36)")
        self.assertAlmostEqual(d["pc"], 1000.0 / 130.23, places=9)
        self.assertAlmostEqual(d["errPc"], (1000.0 / 130.23) * 0.36 / 130.23,
                               places=9)
        self.assertEqual(self._js("formatDistance(7.678654, 0.021175)"),
                         "7.68 pc \u00b1 0.02 pc")
        self.assertEqual(self._js("formatDistance(1500, 30)"),
                         "1.500 kpc \u00b1 0.030 kpc")
        self.assertEqual(self._js("formatDistance(1234.5, 1.5)"),
                         "1.234 kpc \u00b1 0.002 kpc")
        self.assertEqual(self._js("formatDistance(3296700, 45123)"),
                         "3296.7 kpc \u00b1 45.1 kpc")
        self.assertEqual(self._js("formatDistance(12.5)"), "12.5 pc")
        self.assertIsNone(self._js("formatDistance(null, 0.02)"))
        self.assertIsNone(self._js("parallaxDistance(null, 0.36)")["pc"])
        self.assertIsNone(self._js("parallaxDistance(-1, 0.3)")["pc"])
        self.assertIsNone(self._js("parallaxDistance(10, null)")["errPc"])

    def test_target_info_text(self):
        """targetInfoText shows only the ephemeris point count and the clear cross."""
        self.assertEqual(self._js("targetInfoText({})"), "")
        self.assertEqual(
            self._js("targetInfoText({ephemeris: [[1], [2]]})"),
            "eph 2 pts \u00d7")
        self.assertEqual(self._js("targetInfoText({mag: 0.03})"),
                         "\u00d7")
        self.assertEqual(
            self._js('targetInfoText({ephemeris: [[1]], idents: ["a"]})'),
            "eph 1 pts \u00d7")

    def test_target_detail_table(self):
        """targetDetailTable renders every stored Simbad field, counting identifiers from idTotal."""
        obj = {"name": "Vega", "mag": 0.03, "spec": "A0V",
               "plx": 130.23, "plxErr": 0.36,
               "pmRa": 200.94, "pmDec": 286.23, "pmRaErr": 0.32,
               "pmDecErr": 0.4, "rv": -13.5,
               "fluxes": {"U": {"value": 0.03, "err": None},
                          "J": {"value": -0.177, "err": 0.206}},
               "idents": ["TIC 157587146", "GJ 721"], "idTotal": 65}
        html = self._js(f"targetDetailTable({json.dumps(obj)})")
        self.assertIn("<th>Name</th><td>Vega</td>", html)
        self.assertIn("<th>Mags</th><td>U 0.03  J -0.18 \u00b10.21</td>",
                      html)
        self.assertIn("<th>SpT</th><td>A0V</td>", html)
        self.assertIn("<th>Parallax</th><td>130.23 \u00b1 0.36 mas</td>",
                      html)
        self.assertIn("<th>Distance</th><td>7.68 pc \u00b1 0.02 pc</td>",
                      html)
        self.assertIn("<th>Proper motion</th><td>RA +200.94\u00b10.32 / "
                      "Dec +286.23\u00b10.40 mas/yr</td>", html)
        self.assertIn("<th>Radial velocity</th><td>-13.5 km/s</td>", html)
        self.assertIn("<th>Identifiers</th><td>TIC 157587146, GJ 721 "
                      "(+ 63 more)</td>", html)
        self.assertEqual(self._js("targetDetailTable({})"), "")

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
        """findingRadiusFromZoom log mapping: endpoints, default, monotonic.

        The range is 0.5 to 7.5 arcmin, the DSS2 cutout's half field
        (0.25 deg / 2), so the maximum zoom-out matches the sky image.
        """
        self.assertEqual(self._js("findingRadiusFromZoom(0)"), 0.5)
        self.assertEqual(self._js("findingRadiusFromZoom(1000)"), 7.5)
        self.assertEqual(self._js("findingRadiusFromZoom(250)"), 1.0)
        self.assertEqual(self._js("findingRadiusFromZoom(500)"), 1.9)
        self.assertEqual(self._js("findingRadiusFromZoom(750)"), 3.8)
        vals = [self._js(f"findingRadiusFromZoom({t})")
                for t in (0, 250, 500, 750, 1000)]
        self.assertEqual(vals, sorted(vals),
                         f"zoom mapping not monotonic: {vals}")
        self.assertTrue(all(v > 0 for v in vals))

    def test_finding_query_radius(self):
        """findingQueryRadius pads out to the square canvas's half-diagonal.

        The star chart canvas is a square with the drawing circle
        inscribed in it (8 px margin), so the Simbad query radius is
        the display radius times sqrt(2) times the half-side over the
        circle's radius.
        """
        for size in (220, 320, 400, 480):
            for radius in (0.5, 7.5):
                want = radius * math.sqrt(2) * size / 2 / (size / 2 - 8)
                got = self._js(f"findingQueryRadius({radius}, {size})")
                self.assertAlmostEqual(got, want, delta=1e-9)
        # Bigger canvas, smaller relative margin: the padding factor
        # s / (s - 8) decreases toward 1, so the query radius
        # decreases toward radius * sqrt(2) as the size grows.
        vals = [self._js(f"findingQueryRadius(7.5, {s})")
                for s in (220, 320, 400, 480)]
        self.assertEqual(vals, sorted(vals, reverse=True),
                         f"query radius not monotonic in size: {vals}")
        self.assertTrue(all(v > 7.5 * math.sqrt(2) for v in vals))

    def test_finding_offset_and_basic_table(self):
        """findingOffsetText units and findingBasicTable's fixed rows.

        The basic table always renders every row - RA/Dec, Mags, SpT,
        Type, From target, Bib/Notes - with a "-" cell where the field
        is blank for this object, and carries no record expansion.
        """
        self.assertEqual(self._js("findingOffsetText(0.75)"), "0.8\u2033")
        self.assertEqual(self._js("findingOffsetText(59.94)"),
                         "59.9\u2033")
        self.assertEqual(self._js("findingOffsetText(91.9)"), "1.53\u2032")
        self.assertIsNone(self._js("findingOffsetText(null)"))
        row = {"raDeg": 279.186354, "dec": 38.783639,
               "u": 0.03, "b": 0.03, "v": 0.03, "r": 0.07, "i": 0.10,
               "spt": "A0V", "typ": "dS*", "distArcsec": 0.46,
               "bib": 2817, "notn": 1}
        html = self._js(f"findingBasicTable({json.dumps(row)})")
        self.assertIn("<th>RA/Dec</th>", html)
        self.assertIn("<th>Mags</th><td>U 0.03  B 0.03  V 0.03  "
                      "R 0.07  I 0.10</td>", html)
        self.assertIn("<th>SpT</th><td>A0V</td>", html)
        self.assertIn("<th>Type</th><td>dS*</td>", html)
        self.assertIn("<th>From target</th><td>0.5\u2033</td>", html)
        self.assertIn("<th>Bib/Notes</th><td>2817 / 1</td>", html)
        for label in ("Parallax", "Distance", "Proper motion",
                      "Radial velocity", "Identifiers"):
            self.assertNotIn("<th>%s</th>" % label, html)
        sparse = {"raDeg": 10.0, "dec": -20.0}
        html2 = self._js(f"findingBasicTable({json.dumps(sparse)})")
        self.assertIn("<th>RA/Dec</th>", html2)
        self.assertEqual(html2.count("<td>-</td>"), 5, html2)
        # The sim-id record's fluxes fill the Mags row.
        extra = {"fluxes": {"J": {"value": -0.177, "err": 0.206}}}
        html3 = self._js("findingBasicTable(" + json.dumps(sparse) +
                         ", " + json.dumps(extra) + ")")
        self.assertIn("<th>Mags</th><td>J -0.18 \u00b10.21</td>", html3)


    def test_finding_detail_table_extra(self):
        """findingDetailTable's fixed expansion rows and the pending row.

        The record's fields - Parallax, Distance, Proper motion,
        Radial velocity, Identifiers - always render, with a "-" cell
        where the record lacks the field; while the lookup is in
        flight the table is a single "querying" row.
        """
        idents = ["ID %d" % k for k in range(13)]
        extra = {"spec": "A0V", "plx": 130.23, "plxErr": 0.36,
                 "pmRa": 200.94, "pmDec": 286.23, "pmRaErr": 0.32,
                 "pmDecErr": 0.4, "rv": -13.5,
                 "fluxes": {"J": {"value": -0.177, "err": 0.206},
                            "H": {"value": -0.029, "err": 0.146},
                            "K": {"value": 0.129, "err": 0.186}},
                 "idents": idents}
        html = self._js("findingDetailTable(" + json.dumps(extra) +
                        ", true)")
        self.assertEqual(html.count("<tr>"), 1, html)
        self.assertIn("<th>Details</th><td>querying Simbad\u2026</td>",
                      html)
        html2 = self._js("findingDetailTable(" + json.dumps(extra) + ")")
        self.assertIn("<th>Parallax</th><td>130.23 \u00b1 0.36 mas</td>",
                      html2)
        self.assertIn("<th>Distance</th><td>7.68 pc \u00b1 0.02 pc</td>",
                      html2)
        self.assertIn("<th>Proper motion</th><td>RA +200.94\u00b10.32 / "
                      "Dec +286.23\u00b10.40 mas/yr</td>", html2)
        self.assertIn("<th>Radial velocity</th><td>-13.5 km/s</td>", html2)
        self.assertIn("<th>Identifiers</th>", html2)
        self.assertIn("(+ 1 more)", html2)
        for label in ("RA/Dec", "Mags", "SpT", "Type", "From target",
                      "Bib/Notes"):
            self.assertNotIn("<th>%s</th>" % label, html2)
        # Without a record, the five expansion rows still render, all
        # "-".
        html3 = self._js("findingDetailTable(null)")
        for label in ("Parallax", "Distance", "Proper motion",
                      "Radial velocity", "Identifiers"):
            self.assertIn("<th>%s</th><td>-</td>" % label, html3)
        # A partial record leaves the missing fields as "-".
        html4 = self._js("findingDetailTable(" + json.dumps({"plx": 23.45})
                         + ")")
        self.assertIn("<th>Parallax</th><td>23.45 mas</td>", html4)
        self.assertIn("<th>Distance</th><td>42.6 pc</td>", html4)
        for label in ("Proper motion", "Radial velocity", "Identifiers"):
            self.assertIn("<th>%s</th><td>-</td>" % label, html4)

    def test_share_fragment_round_trip(self):
        """encode/decodeShareFragment round-trip both ways; malformed fragments give null."""
        doc = {
            "observatory": {"name": "Mount Graham (LBT)", "lat": 31.76,
                            "lon": -111.37, "elevation_m": 2283.0,
                            "timezone": "America/Phoenix",
                            "preset": "Mount Graham (LBT)"},
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


    def test_preset_select_grouping(self):
        """buildPresetSelect groups shared sites into optgroups with facility labels."""
        fn = _extract_fn(self.page, "buildPresetSelect")
        stub = (
            "function __mk(tag) {"
            " var e = {tag: tag, children: [], value: '', textContent: '', label: '',"
            " innerHTML: '', appendChild: function (c) { this.children.push(c); }};"
            " return e; }"
            "var __root = __mk('select');"
            "var document = {createElement: __mk};"
            "function $(id) { return __root; }"
            "var OBSERVATORIES = " + json.dumps([to_dict(p) for p in PRESETS.values()]) + ";"
            + fn + ";"
            "buildPresetSelect();"
            "JSON.stringify(__root.children)"
        )
        got = json.loads(self.js.evaljs(stub))
        groups = [c for c in got if c["tag"] == "optgroup"]
        flat = [c for c in got if c["tag"] == "option"]
        mauna = [g for g in groups if g["label"] == "Mauna Kea"]
        self.assertEqual(len(mauna), 1)
        self.assertEqual(
            [(o["value"], o["textContent"]) for o in mauna[0]["children"]],
            [("Mauna Kea (Gemini North)", "Gemini North"),
             ("Mauna Kea (Keck I/II)", "Keck I/II"),
             ("Mauna Kea (Subaru)", "Subaru")])
        flat_pairs = {(o["value"], o["textContent"]) for o in flat}
        self.assertIn(("Calar Alto", "Calar Alto"), flat_pairs)
        self.assertEqual(got[-1]["value"], "Custom")
        values = [o["value"] for c in got
                  for o in ([c] if c["tag"] == "option" else c["children"])]
        self.assertEqual(sorted(values),
                         sorted([p.name for p in PRESETS.values()] + ["Custom"]))

    def test_preset_rename_migration(self):
        """migratePresetRename rewrites pre-rename preset names to the current names."""
        start = self.page.index("function migratePresetRename")
        end = self.page.index("\n}", start) + 2
        self.js.evaljs("var PRESET_RENAMES = " + json.dumps(RENAME_MAP) + ";")
        self.js.evaljs(self.page[start:end])
        doc = {
            "observatory": {"name": "LBT (Large Binocular Telescope)",
                            "preset": "LBT (Large Binocular Telescope)",
                            "lat": 32.44278, "lon": -111.155,
                            "elevation_m": 2400.0,
                            "timezone": "America/Phoenix"},
            "objects": [],
            "time_range": {"start": "2026-11-20 00:00", "days": 1.0},
        }
        got = json.loads(self.js.evaljs(
            "JSON.stringify(migratePresetRename(JSON.parse(" + json.dumps(json.dumps(doc)) + ")))"))
        self.assertEqual(got["observatory"]["preset"], "Mount Graham (LBT)")
        self.assertEqual(got["observatory"]["name"], "Mount Graham (LBT)")
        doc2 = {"observatory": {"name": "My custom site", "preset": "VLT (Paranal)"}}
        got2 = json.loads(self.js.evaljs(
            "JSON.stringify(migratePresetRename(JSON.parse(" + json.dumps(json.dumps(doc2)) + ")))"))
        self.assertEqual(got2["observatory"]["preset"], "Cerro Paranal (VLT)")
        self.assertEqual(got2["observatory"]["name"], "My custom site")
        self.assertIsNone(self.js.evaljs("migratePresetRename(null)"))
        self.assertEqual(json.loads(self.js.evaljs("JSON.stringify(migratePresetRename({}))")), {})

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



    def test_instructions_dropdown(self):
        """The footer carries an Instructions dropdown; the header keeps only the title."""
        self.assertIn('<details id="instructions">', self.page)
        self.assertIn('<summary>Instructions</summary>', self.page)
        for lead in ("<b>Observatory</b>", "<b>Time range</b>", "<b>Display</b>",
                     "<b>Targets</b>", "<b>Main chart</b>", "<b>Observing windows</b>",
                     "<b>Annual calendar</b>", "<b>Workspace</b>", "<b>Theme</b>"):
            self.assertIn(lead, self.page)
        self.assertNotIn("<p>Airmass (X) for the targets below", self.page)

    def test_no_snr_leftovers(self):
        """The S/N estimate component is gone from the page."""
        self.assertNotIn("S/N", self.page)
        self.assertNotIn("snr-", self.page)
        self.assertNotIn("snrOn", self.page)
        self.assertNotIn("normalizeSnr", self.page)

    def test_duration_preset_buttons(self):
        """The Duration row offers 1/2/3 day presets that re-fire the input handler."""
        for n in (1, 2, 3):
            self.assertIn('<button id="days-%d" title=' % n, self.page)
        self.assertIn("function setDurationDays", self.page)
        self.assertIn('dispatchEvent(new Event("input"))', self.page)

    def test_moon_glyph_illum_label(self):
        """The midnight moon glyph is radius 8 and carries its % illumination."""
        self.assertIn("drawMoonGlyph(mnx, plot.y + 26, 8, mni.elongDeg)", self.page)
        self.assertIn('Math.round(mni.illum * 100) + "%"', self.page)
        self.assertIn("mnSpPx > 46", self.page)

    def test_moon_phase_info(self):
        """moonPhaseInfo agrees with astropy: elongation, illumination, name, age."""
        from astropy.coordinates.builtin_frames.ecliptic import GeocentricMeanEcliptic
        for jd in (2458000.5, 2460000.5, 2461130.5, 2463000.5):
            t = Time(jd, format="jd")
            moon = get_body("moon", t, location=EarthLocation(x=0, y=0, z=0, unit="m"))
            sun = get_sun(t)
            want_e = (moon.transform_to(GeocentricMeanEcliptic(equinox=t)).lon.deg -
                      sun.transform_to(GeocentricMeanEcliptic(equinox=t)).lon.deg) % 360
            sep = moon.separation(sun)
            want_illum = (1 - math.cos(math.radians(sep.deg))) / 2
            got = json.loads(self._js("JSON.stringify(moonPhaseInfo(%f))" % jd))
            self.assertAlmostEqual(got["elongDeg"], want_e, delta=0.7, msg=jd)
            self.assertAlmostEqual(got["illum"], want_illum, delta=0.005, msg=jd)
            names = ["new moon", "waxing crescent", "first quarter",
                     "waxing gibbous", "full moon", "waning gibbous",
                     "last quarter", "waning crescent"]
            self.assertEqual(got["name"],
                             names[int((got["elongDeg"] + 22.5) % 360 // 45) % 8],
                             msg=jd)
            self.assertAlmostEqual(got["ageDays"],
                                   got["elongDeg"] / 360 * 29.530589, places=12)
            mp = json.loads(self._js("JSON.stringify(moonPhase(%f))" % jd))
            self.assertEqual(got["illum"], mp["illum"])

    def test_weather_symbols(self):
        """weatherSymbolKind classifies storm, rain, or nothing."""
        self.assertEqual(
            self._js("weatherSymbolKind({prec: 0.5, wc: 2, pp: 10})"), "rain")
        self.assertEqual(
            self._js("weatherSymbolKind({wc: 95, prec: 0, pp: 0})"), "storm")
        self.assertTrue(
            self._js("weatherSymbolKind({prec: 0, wc: 2, pp: 30}) === null"))
        self.assertEqual(
            self._js("weatherSymbolKind({prec: 0, wc: null, pp: 40})"), "rain")


if __name__ == "__main__":
    unittest.main()
