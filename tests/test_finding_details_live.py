"""Live test for the finding-chart secondary Simbad detail lookup.

Evaluates the built page's full script in a real JS engine (dukpy /
Duktape) with a fetch stub whose Simbad answers are the archived Vega
sim-id record (tests/fixtures/simbad_vega.txt). It drives the real
hover path (findingHover -> updateFindingInfo -> maybeFetchFindingDetail
-> simbadLookup -> findingDetailsCache -> re-render) and checks that
the first hover shows a "querying" row, the second renders the record's
parallax, distance, proper motion, radial velocity, and identifiers,
and that a repeat hover is served from the cache without a second
fetch. It also drives renderTargets with a stored Vega record plus a
bare target and checks the Target details window (renderTargetDetails)
renders each record's table under its name heading, a dim placeholder
for the bare target, a cross-only row status, and hides when the Extra
info option is off.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

try:
    import dukpy
except ImportError:  # pragma: no cover
    dukpy = None

from airmass_gui.site import build_site
from airmass_gui.workspace import new_workspace
from test_weather_live import STUBS

VEGA_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "simbad_vega.txt"

DRIVER = r"""
var __driverDone = false;
var __driverError = null;
var __probe = { pendingShown: false, html: "", html2: "",
                simbadUrls: [], simbadUrls2: [] };

function __simbadUrls() {
  return __fetchLog.filter(function (u) {
    return String(u).indexOf("simbad") !== -1;
  });
}

async function __main() {
  var __fc = $("finding-chart");
  if (!__fc.getContext)
    __fc.getContext = function () { return __ctx; };
  findingHover = {
    center: { raDeg: 279.23473479, dec: 38.78368889 },
    radiusArcmin: 30,
    rows: [{ raDeg: 279.23473479, dec: 38.78368889, name: "Vega",
             u: 0.03, b: 0.03, v: 0.03, r: 0.07, i: 0.10,
             spt: "A0V", typ: "dS*", distArcsec: 0.5,
             bib: 2817, notn: 1 }],
    label: "Vega",
    centerIdx: 0
  };
  updateFindingInfo(240, 240);
  __probe.pendingShown = $("finding-details").innerHTML
      .indexOf("querying Simbad") !== -1;
  for (var q = 0; q < 200
       && Object.keys(findingDetailsPending).length > 0; q++)
    await Promise.resolve();
  __probe.html = $("finding-details").innerHTML;
  __probe.simbadUrls = __simbadUrls();
  updateFindingInfo(240, 240);
  __probe.html2 = $("finding-details").innerHTML;
  __probe.simbadUrls2 = __simbadUrls();
}

__main().then(function () { __driverDone = true; },
              function (err) {
                __driverError = String((err && err.stack) || err);
                __driverDone = true;
              });
"""

DRIVER_DETAILS = r"""
var __driverDone = false;
var __driverError = null;
var __probe = { display: "", html: "", rowText: "", displayOff: "" };

async function __main() {
  state.objects.length = 0;
  state.objects.push({
    name: "Vega", ra: "18 36 56.34", dec: "+38 47 01.28",
    color: "#4d90fe", mag: 0.03, spec: "A0V",
    plx: 130.23, plxErr: 0.36,
    pmRa: 200.94, pmDec: 286.23, pmRaErr: 0.32, pmDecErr: 0.4,
    rv: -13.5,
    fluxes: { J: { value: -0.177, err: 0.206 } },
    idents: ["TIC 157587146"], idTotal: 65
  });
  state.objects.push({
    name: "Other", ra: "10 00 00", dec: "+20 00 00", color: "#ff4d4d"
  });
  state.options.extraInfo = true;
  renderTargets();
  __probe.display = $("target-details").style.display;
  __probe.html = $("target-details").innerHTML;
  __probe.rowText = $("targets").children[0].children[5].textContent;
  state.options.extraInfo = false;
  renderTargetDetails();
  __probe.displayOff = $("target-details").style.display;
}

__main().then(function () { __driverDone = true; },
              function (err) {
                __driverError = String((err && err.stack) || err);
                __driverDone = true;
              });
"""


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class TestFindingDetailsLive(unittest.TestCase):
    """Hover detail lookup against a stubbed Simbad sim-id record."""

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        cls.script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        cls.all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        cls.fixture = VEGA_FIXTURE.read_text(encoding="utf-8")
        cls.interp = dukpy.JSInterpreter()
        stubs = (
            STUBS
            .replace("__ALL_IDS__", json.dumps(cls.all_ids))
            .replace("__TZ_TABLE__", json.dumps(
                {"UTC": 0, "America/Phoenix": -7}))
            # Raw sim-id record text (no JSON layer on the Simbad path).
            .replace("__SIMBAD_BODY__", json.dumps(cls.fixture))
            # Double encoding: the page JSON.parses the stub text result.
            .replace("__OPEN_METEO_BODY__", json.dumps(json.dumps({})))
        )
        cls.interp.evaljs(stubs + ";\n0")
        cls.interp.evaljs(cls.script + ";\n0")

    def _driver_result(self):
        """Run the hover driver; return the probe report as a dict."""
        self.interp.evaljs(DRIVER + ";\n0")
        return self._collect_result()

    def _details_result(self):
        """Run the Target details window driver; return the probe report."""
        self.interp.evaljs(DRIVER_DETAILS + ";\n0")
        return self._collect_result()

    def _collect_result(self):
        """Wait for the driver, check for errors, return the probe dict."""
        self.assertTrue(self.interp.evaljs("__driverDone === true"),
                        "driver did not finish")
        err = self.interp.evaljs("__driverError")
        self.assertIsNone(err, "driver error: %s" % err)
        return json.loads(self.interp.evaljs("JSON.stringify(__probe)"))

    def test_hover_fetch_and_detail_rows(self):
        """First hover queries once and re-renders with the record's astrometry."""
        info = self._driver_result()
        self.assertTrue(info["pendingShown"],
                        "no 'querying' row while the lookup is in flight")
        self.assertEqual(len(info["simbadUrls"]), 1,
                         "expected exactly one Simbad fetch, got %r"
                         % info["simbadUrls"])
        url = info["simbadUrls"][0]
        self.assertIn("sim-id", url)
        self.assertIn("Ident=Vega", url)
        self.assertIn("output.format=ASCII", url)
        for token in ("Parallax", "130.23", "mas", "Distance", "7.68 pc",
                      "Proper motion", "200.94", "286.23",
                      "Radial velocity", "-13.5 km/s",
                      "Identifiers", "(+ 53 more)"):
            self.assertIn(token, info["html"])

    def test_repeat_hover_served_from_cache(self):
        """A second hover at the same position does not refetch."""
        info = self._driver_result()
        self.assertEqual(len(info["simbadUrls2"]), 1,
                         "repeat hover issued a new Simbad fetch")
        self.assertNotIn("querying Simbad", info["html2"])
        self.assertIn("Parallax", info["html2"])
        self.assertIn("Identifiers", info["html2"])

    def test_target_details_window(self):
        """renderTargets builds the toggleable Target details window per target."""
        info = self._details_result()
        self.assertEqual(info["display"], "")
        self.assertIn("<h3>Vega</h3>", info["html"])
        self.assertIn("130.23 \u00b1 0.36 mas", info["html"])
        self.assertIn("7.68 pc \u00b1 0.02 pc", info["html"])
        self.assertIn("RA +200.94\u00b10.32 / Dec +286.23\u00b10.40 mas/yr",
                      info["html"])
        self.assertIn("TIC 157587146 (+ 64 more)", info["html"])
        self.assertIn("Other: no Simbad record", info["html"])
        self.assertEqual(info["rowText"], "\u00d7")
        self.assertEqual(info["displayOff"], "none")
