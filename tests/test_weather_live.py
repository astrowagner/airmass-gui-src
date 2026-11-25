"""Regression tests for the open-meteo weather pipeline of the HTML site.

Two layers:

* ``TestLiveFixtureContract`` checks the archived live API response
  (``tests/fixtures/open_meteo_live_20261121.json``, captured 2026-11-21
  from api.open-meteo.com for the LBT) against the contract the site
  consumes: keys, 240 hourly local times, and percentage ranges.
* ``TestWeatherPipelineLive`` evaluates the built page's full script in
  a real JS engine (dukpy / Duktape) with a validating open-meteo fetch
  stub that returns HTTP 400 plus the API's error reason when the
  requested date range exceeds the live horizon (~15 days forward of
  the UTC clock). It drives the real UI path (checkbox -> fetch ->
  cache -> draw -> summary) with a body generated at run time in the
  live response shape, so the test stays valid on any date. It exists
  to catch request windows the live API rejects: the 2026-11-21
  regression, where end_date = today + 16 d was one day past the
  horizon and produced HTTP 400 in real browsers while a permissive
  fetch stub masked it.
"""
import json
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    import dukpy
except ImportError:  # pragma: no cover
    dukpy = None

from airmass_gui.observatories import PRESETS
from airmass_gui.site import build_site
from airmass_gui.workspace import new_workspace

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "open_meteo_live_20261121.json"


class TestLiveFixtureContract(unittest.TestCase):
    """The archived live open-meteo body matches the API contract."""

    @classmethod
    def setUpClass(cls):
        cls.body = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_top_level_keys(self):
        """Top level carries the live API's full key set."""
        expected = {
            "latitude", "longitude", "timezone", "timezone_abbreviation",
            "utc_offset_seconds", "elevation", "generationtime_ms",
            "hourly", "hourly_units",
        }
        missing = expected - set(self.body)
        self.assertFalse(missing, "fixture missing keys: %s" % sorted(missing))

    def test_location_is_lbt(self):
        """The fixture was requested for the LBT in America/Phoenix.

        open-meteo echoes the model grid point it used rather than the
        requested coordinates; both agree within a grid cell (~1 km).
        """
        lbt = PRESETS["Mount Graham (LBT)"]
        self.assertLessEqual(abs(self.body["latitude"] - lbt.lat), 0.01)
        self.assertLessEqual(abs(self.body["longitude"] - lbt.lon), 0.01)
        self.assertEqual(self.body["timezone"], "America/Phoenix")
        self.assertEqual(self.body["utc_offset_seconds"], -25200)

    def test_hourly_shape(self):
        """Hourly block: 240 local hours, 2026-11-21 00:00 .. 2026-11-30 23:00."""
        hourly = self.body["hourly"]
        self.assertEqual(sorted(hourly),
                         ["cloud_cover", "precipitation_probability", "time"])
        self.assertEqual(len(hourly["time"]), 240)
        self.assertEqual(len(hourly["cloud_cover"]), 240)
        self.assertEqual(len(hourly["precipitation_probability"]), 240)
        self.assertEqual(hourly["time"][0], "2026-11-21T00:00")
        self.assertEqual(hourly["time"][-1], "2026-11-30T23:00")
        for i, t in enumerate(hourly["time"]):
            self.assertRegex(t, r"^\d{4}-\d{2}-\d{2}T\d{2}:00$",
                             "bad hour %d: %s" % (i, t))

    def test_value_ranges(self):
        """Cloud cover and rain chance are percentages, never null."""
        for key in ("cloud_cover", "precipitation_probability"):
            values = self.body["hourly"][key]
            self.assertTrue(all(v is not None for v in values),
                            key + " has nulls")
            self.assertTrue(all(0.0 <= v <= 100.0 for v in values),
                            key + " out of [0, 100]")


@unittest.skipUnless(dukpy is not None, "dukpy not installed")
class TestWeatherPipelineLive(unittest.TestCase):
    """Full-page weather pipeline against a live-shaped, validating stub."""

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            cls.page = build_site(new_workspace(),
                                  Path(tmp) / "index.html").read_text(
                encoding="utf-8")
        cls.script = re.findall(r"<script>(.*?)</script>", cls.page, re.S)[0]
        cls.all_ids = sorted(set(re.findall(r'id="([^"]+)"', cls.page)))
        obs = json.loads(
            re.search(r"var OBSERVATORIES =\s*(\[.*?\]);",
                      cls.script, re.S).group(1))
        cls.tz_table = {"UTC": 0, "America/Denver": -7}
        for o in obs:
            cls.tz_table.setdefault(o["timezone"], -7)
        cls.om_body = cls._live_shaped_body()
        cls.interp = dukpy.JSInterpreter()
        stubs = (
            STUBS
            .replace("__ALL_IDS__", json.dumps(cls.all_ids))
            .replace("__TZ_TABLE__", json.dumps(cls.tz_table))
            .replace("__SIMBAD_BODY__", json.dumps(""))
            # Double encoding: the page JSON.parses the stub text result.
            .replace("__OPEN_METEO_BODY__", json.dumps(json.dumps(cls.om_body)))
            .replace("__HORIZONS_BODY__", json.dumps(""))
        )
        cls.interp.evaljs(stubs + ";\n0")
        cls.interp.evaljs(cls.script + ";\n0")

    @staticmethod
    def _live_shaped_body():
        """Hourly body in the live API shape, centered on the current date.

        Local times are generated in America/Phoenix (fixed UTC-7, no
        DST), covering today-17 d .. today+16 d so any request window
        the page forms around today is fully inside the data.
        """
        phx = ZoneInfo("America/Phoenix")
        now = datetime.now(timezone.utc).astimezone(phx)
        start = (now - timedelta(days=17)).replace(minute=0, second=0,
                                                   microsecond=0)
        times, cc, pp = [], [], []
        for i in range(17 * 2 * 24):
            t = start + timedelta(hours=i)
            times.append(t.strftime("%Y-%m-%dT%H:00"))
            cc.append((i * 37) % 101)
            pp.append((i * 53) % 101)
        return {
            "latitude": 32.44278, "longitude": -111.155,
            "timezone": "America/Phoenix", "timezone_abbreviation": "MST",
            "utc_offset_seconds": -25200, "elevation": 2400.0,
            "generationtime_ms": 0.123,
            "hourly_units": {"time": "iso8601", "cloud_cover": "%",
                             "precipitation_probability": "%"},
            "hourly": {"time": times, "cloud_cover": cc,
                       "precipitation_probability": pp},
        }

    def _driver_result(self):
        """Run the pipeline driver; return the probe report as a dict."""
        self.interp.evaljs(DRIVER + ";\n0")
        self.assertTrue(self.interp.evaljs("__driverDone === true"),
                        "driver did not finish")
        err = self.interp.evaljs("__driverError")
        self.assertIsNone(err, "driver error: %s" % err)
        return json.loads(self.interp.evaljs("JSON.stringify(__probe)"))

    def test_request_window_within_safe_horizon(self):
        """weatherRequest never asks past open-meteo's forecast horizon."""
        self.interp.evaljs(
            "state.time_range.start = ''; state.time_range.days = 30;")
        req = json.loads(self.interp.evaljs(
            "JSON.stringify(weatherRequest(state.observatory))"))
        self.assertIsNotNone(req, "weatherRequest returned null")
        m = re.search(r"start_date=(\d{4}-\d{2}-\d{2})"
                      r"&end_date=(\d{4}-\d{2}-\d{2})", req["url"])
        self.assertIsNotNone(m, req["url"])
        phx = ZoneInfo("America/Phoenix")
        now = datetime.now(timezone.utc).astimezone(phx)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0,
                            tzinfo=None)
        end = datetime.strptime(m.group(2), "%Y-%m-%d")
        start = datetime.strptime(m.group(1), "%Y-%m-%d")
        self.assertLessEqual(end - today, timedelta(days=14),
                             "end_date %s exceeds the safe horizon" % m.group(2))
        self.assertGreaterEqual(start - today, timedelta(days=-16),
                                "start_date %s before coverage" % m.group(1))

    def test_rejected_range_reports_api_reason(self):
        """A 400 from the API surfaces its reason in the status bar."""
        self.interp.evaljs(r"""
          var __rejDone = false;
          var __rejStatus = null;
          var __savedFetch = fetch;
          fetch = function (url, opts) {
            return Promise.resolve({
              ok: false, status: 400,
              text: function () {
                return Promise.resolve(JSON.stringify({
                  error: true,
                  reason: "Parameter 'end_date' is out of allowed range"
                }));
              }
            });
          };
          (async function () {
            try {
              var flight = weatherRefresh();
              if (flight) await flight;
              for (var q = 0; q < 5; q++) await Promise.resolve();
              __rejStatus = String($("status").textContent);
            } finally {
              fetch = __savedFetch;
              __rejDone = true;
            }
          })();
        """)
        for _ in range(10):
            self.interp.evaljs("0")
        self.assertTrue(self.interp.evaljs("__rejDone === true"),
                        "reject driver did not finish")
        status = self.interp.evaljs("String(__rejStatus)")
        self.assertIn("out of allowed range", status, status)

    def test_weather_fetch_and_summary(self):
        """Checkbox toggle fetches within the safe horizon and renders the band with its legend entry."""
        info = self._driver_result()
        self.assertEqual(info["fetches"], 1)
        self.assertIn("latitude=32.44278", info["url"])
        self.assertIn("longitude=-111.155", info["url"])
        self.assertIn("timezone=America%2FPhoenix", info["url"])
        self.assertIn("hourly=cloud_cover,precipitation_probability",
                      info["url"])
        self.assertEqual(info["hours"], len(self.om_body["hourly"]["time"]))
        self.assertGreaterEqual(info["inWindow"], 48)
        self.assertEqual(info["status"], "")
        self.assertGreaterEqual(len(info["pcts"]), 1)
        for pct in info["pcts"]:
            self.assertTrue(0 <= pct <= 100)

STUBS = r"""
// ================= results & timers =================
var __results = [];
function step(name, fn) {
  try { fn(); __results.push({ name: name, ok: true }); }
  catch (err) { __results.push({ name: name, ok: false, error: String((err && err.stack) || err) }); }
}
var __timers = {}, __nextTid = 1;
function setTimeout(fn, ms) { var id = __nextTid++; __timers[id] = fn; return id; }
function clearTimeout(id) { delete __timers[id]; }
function flushTimers() {
  for (var round = 0; round < 20; round++) {
    var ids = Object.keys(__timers);
    if (ids.length === 0) break;
    ids.forEach(function (id) { var fn = __timers[id]; delete __timers[id]; fn(); });
  }
}

// ================= Intl (fixed-offset stand-in; tz accuracy is unit-tested in Python) =================
var __TZ_OFFSETS = __TZ_TABLE__;
function __tzOffsetHours(tz) {
  if (tz == null || tz === "") return 0;
  if (Object.prototype.hasOwnProperty.call(__TZ_OFFSETS, tz)) return __TZ_OFFSETS[tz];
  var m = /^UTC([+-]\d{1,2})(?::([0-5]\d))?$/.exec(tz);
  if (m) return parseInt(m[1], 10) + (m[2] ? parseInt(m[2], 10) / 60 : 0);
  throw new RangeError("unsupported time zone in smoke stub: " + tz);
}
Intl = {
  DateTimeFormat: function (locale, opts) {
    var off = __tzOffsetHours(opts && opts.timeZone);
    function partsAt(ms) {
      var d = new Date(ms + off * 3600000);
      return { year: d.getUTCFullYear(), month: d.getUTCMonth() + 1, day: d.getUTCDate(),
               hour: d.getUTCHours(), minute: d.getUTCMinutes(), second: d.getUTCSeconds() };
    }
    function pad(n) { return (n < 10 ? "0" : "") + n; }
    var MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    this.format = function (date) {
      var p = partsAt(date.getTime());
      if (opts && opts.month && opts.month === "short") {
        return MONTHS[p.month - 1] + " " + pad(p.day) + " " + pad(p.hour) + ":" + pad(p.minute);
      }
      return p.year + "-" + pad(p.month) + "-" + pad(p.day) + " " +
             pad(p.hour) + ":" + pad(p.minute) + ":" + pad(p.second);
    };
    this.formatToParts = function (date) {
      var p = partsAt(date.getTime());
      return [
        { type: "year", value: String(p.year) },
        { type: "month", value: pad(p.month) },
        { type: "day", value: pad(p.day) },
        { type: "hour", value: pad(p.hour) },
        { type: "minute", value: pad(p.minute) },
        { type: "second", value: pad(p.second) }
      ];
    };
  }
};

// ================= DOM =================
var __els = {}, __winHandlers = {};
var __clips = [], __confirms = [], __prompts = [], __execCmds = [];
var __blobs = [], __dataUrls = [], __lastReplaceUrl = null;

function __parseFlatHtml(html) {
  var els = [];
  var re = /<([a-zA-Z][a-zA-Z0-9]*)((?:"[^"]*"|[^">])*)>/g;
  var m;
  while ((m = re.exec(html)) !== null) {
    var el = __makeEl(m[1]);
    var attrRe = /([a-zA-Z_:]+)\s*=\s*"([^"]*)"/g, am;
    while ((am = attrRe.exec(m[2])) !== null) el[am[1]] = am[2];
    els.push(el);
  }
  return els;
}

function __makeEl(tag, id) {
  var el = {
    tagName: (tag || "div").toUpperCase(),
    id: id || "",
    value: "", checked: false, disabled: false,
    textContent: "",
    className: "",
    style: {},
    children: [],
    parentElement: null,
    clientWidth: 0, clientHeight: 0,
    offsetWidth: 200, offsetHeight: 100,
    files: null,
    dataset: {},
    __innerHTML: "",
    __handlers: {}
  };
  el.style.setProperty = function (k, v) { this[k] = v; };
  el.style.getPropertyValue = function (k) {
    return (k in this) ? this[k] : "";
  };
  el.setAttribute = function (k, v) { this[k] = v; };
  el.getAttribute = function (k) { return (k in this) ? this[k] : null; };
  el.hasAttribute = function (k) { return k in this; };
  el.addEventListener = function (type, fn) { this.__handlers[type] = fn; };
  el.dispatchEvent = function (ev) {
    var h = this.__handlers && this.__handlers[ev.type];
    if (h) h.call(this, ev);
    return true;
  };
  el.removeEventListener = function () {};
  el.appendChild = function (child) { this.children.push(child); child.parentElement = this; return child; };
  el.removeChild = function (child) {
    var i = this.children.indexOf(child);
    if (i >= 0) { this.children.splice(i, 1); child.parentElement = null; }
    return child;
  };
  el.remove = function () { if (this.parentElement) this.parentElement.removeChild(this); };
  el.focus = function () {};
  el.select = function () {};
  el.click = function () {
    if (this.__handlers["click"]) {
      var ev = { type: "click", target: this };
      ev.preventDefault = function () {};
      this.__handlers["click"].call(this, ev);
    }
  };
  el.getBoundingClientRect = function () {
    var w = this.clientWidth || 800, h = this.clientHeight || 600;
    return { left: 0, top: 0, right: w, bottom: h, width: w, height: h };
  };
  Object.defineProperty(el, "innerHTML", {
    get: function () { return this.__innerHTML; },
    set: function (html) {
      this.__innerHTML = String(html);
      this.children = __parseFlatHtml(this.__innerHTML);
      var self = this;
      this.children.forEach(function (c) { c.parentElement = self; });
    }
  });
  return el;
}

var __ctx = {
  measureText: function (s) { return { width: String(s).length * 6.5 }; }
};
["arc","beginPath","clearRect","clip","closePath","fill","fillRect","fillText","lineTo","moveTo",
 "ellipse","quadraticCurveTo","rect","restore","rotate","save","setLineDash",
 "setTransform","stroke","strokeRect","translate"].forEach(function (m) {
    __ctx[m] = function () {};
  });

__ALL_IDS__.forEach(function (id) { __els[id] = __makeEl("div", id); });
if (!__els["chart"]) __els["chart"] = __makeEl("div", "chart");
var __canvas = __els["chart"];
__canvas.tagName = "CANVAS";
__canvas.clientWidth = 800;
__canvas.clientHeight = 396;
__canvas.getContext = function () { return __ctx; };
__canvas.toDataURL = function (type) { __dataUrls.push(type || ""); return "data:image/png;base64,AAAA"; };
__canvas.parentElement = __makeEl("div");
__canvas.parentElement.getBoundingClientRect = function () {
  return { left: 0, top: 0, width: 820, height: 700 };
};

document = {
  documentElement: __makeEl("html"),
  body: __makeEl("body"),
  createElement: function (tag) { return __makeEl(tag); },
  getElementById: function (id) {
    if (!__els[id]) __els[id] = __makeEl("div", id);
    return __els[id];
  },
  execCommand: function (cmd) { __execCmds.push(cmd); return true; }
};

window = {
  devicePixelRatio: 2,
  location: { href: "https://astrowagner.github.io/airmass-gui/", hash: "", search: "" },
  history: {
    replaceState: function (a, b, url) {
      __lastReplaceUrl = String(url || "");
      var i = __lastReplaceUrl.indexOf("#");
      window.location.hash = i === 0 ? __lastReplaceUrl : (i > 0 ? __lastReplaceUrl.slice(i) : "");
    }
  },
  addEventListener: function (type, fn) { __winHandlers[type] = fn; },
  confirm: function (msg) { __confirms.push(msg); return true; },
  prompt: function (msg) { __prompts.push(msg); return "Smoke WS"; }
};

localStorage = {
  __map: {},
  getItem: function (k) { return (k in localStorage.__map) ? localStorage.__map[k] : null; },
  setItem: function (k, v) { localStorage.__map[k] = String(v); },
  removeItem: function (k) { delete localStorage.__map[k]; }
};

navigator = {
  clipboard: {
    writeText: function (text) { __clips.push(text); return Promise.resolve(); }
  }
};

var __fetchLog = [];
// Mimic the live open-meteo API's range validation: 400 plus an error
// reason when start_date/end_date fall outside the allowed window
// (~90 days back, ~15 days forward of the server's UTC clock).
function __omRangeError(url) {
  var m = /start_date=(\d{4}-\d{2}-\d{2})&end_date=(\d{4}-\d{2}-\d{2})/.exec(String(url));
  if (!m) return null;
  var now = new Date();
  var utcToday = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  var parse = function (d) {
    var p = /^(\d{4})-(\d{2})-(\d{2})$/.exec(d);
    return Date.UTC(+p[1], +p[2] - 1, +p[3]);
  };
  var s = parse(m[1]), e = parse(m[2]);
  if (s < utcToday - 90 * 86400000)
    return "Parameter 'start_date' is out of allowed range";
  if (e > utcToday + 15 * 86400000)
    return "Parameter 'end_date' is out of allowed range";
  return null;
}
fetch = function (url, opts) {
  __fetchLog.push(String(url));
  var body;
  if (String(url).indexOf("open-meteo") !== -1) {
    var err = __omRangeError(url);
    if (err) {
      return Promise.resolve({
        ok: false, status: 400,
        text: function () {
          return Promise.resolve(JSON.stringify({ error: true, reason: err }));
        }
      });
    }
    body = __OPEN_METEO_BODY__;
  } else if (String(url).indexOf("horizons") !== -1) {
    body = __HORIZONS_BODY__;
  } else {
    body = __SIMBAD_BODY__;
  }
  var resp = {
    ok: true, status: 200,
    text: function () { return Promise.resolve(body); }
  };
  return Promise.resolve(resp);
};

URL = {
  __count: 0,
  createObjectURL: function (blob) { return "blob:smoke-" + (++URL.__count); },
  revokeObjectURL: function () {}
};
Blob = function (parts, opts) {
  this.text = (parts || []).map(function (p) { return String(p); }).join("");
  this.type = (opts && opts.type) || "";
  __blobs.push(this);
};
AbortController = function () {
  this.signal = { aborted: false };
  var self = this;
  this.abort = function () { self.signal.aborted = true; };
};
Event = function (type) { this.type = type; };
FileReader = function () {
  var self = this;
  this.readAsText = function (file) {
    self.result = (file && file.text) || "";
    if (self.onload) self.onload({ target: self });
  };
};
"""

DRIVER = r"""

var __driverDone = false;
var __driverError = null;

function fire(el, type, ev) {
  var h = el.__handlers && el.__handlers[type];
  if (!h) throw new Error("no '" + type + "' handler on <" + (el.id || el.tagName) + ">");
  ev = ev || {};
  ev.type = type;
  if (!ev.target) ev.target = el;
  if (!ev.preventDefault) ev.preventDefault = function () { ev.__pd = true; };
  h.call(el, ev);
}

async function __main() {
  if (state.observatory.timezone !== "America/Phoenix")
    throw new Error("unexpected default tz: " + state.observatory.timezone);

  var p = tzParts("America/Phoenix", Date.now());
  state.time_range.start = p.year + "-" + pad2(p.month) + "-" + pad2(p.day) + " 00:00";
  state.time_range.days = 2;
  view = null;
  $("weather-chk").checked = true;
  fire($("weather-chk"), "change");
  flushTimers();
  for (var q = 0; q < 40; q++) await Promise.resolve();

  var wxUrls = __fetchLog.filter(function (u) { return u.indexOf("open-meteo") !== -1; });
  if (wxUrls.length !== 1)
    throw new Error("expected one weather fetch, got " + wxUrls.length);
  var u = wxUrls[0];
  var m = /start_date=(\d{4}-\d{2}-\d{2})&end_date=(\d{4}-\d{2}-\d{2})/.exec(u);
  if (!m) throw new Error("no date range in url: " + u);
  var DAY = 86400000;
  var t0 = wallToUtc("America/Phoenix", p.year, p.month, p.day, 0, 0, 0);
  var maxEnd = tzParts("America/Phoenix", t0 + 14 * DAY);
  var minStart = tzParts("America/Phoenix", t0 - 16 * DAY);
  var ymd = function (s) {
    var a = s.split("-");
    return +a[0] * 10000 + +a[1] * 100 + +a[2];
  };
  var num = function (q) { return q.year * 10000 + q.month * 100 + q.day; };
  if (ymd(m[2]) > num(maxEnd))
    throw new Error("end_date " + m[2] + " beyond safe horizon");
  if (ymd(m[1]) < num(minStart))
    throw new Error("start_date " + m[1] + " before coverage");
  if (u.indexOf("latitude=32.44278") === -1 || u.indexOf("longitude=-111.155") === -1)
    throw new Error("weather url missing LBT coords: " + u);
  if (u.indexOf("timezone=America%2FPhoenix") === -1)
    throw new Error("weather url missing resolved tz: " + u);
  if (u.indexOf("hourly=cloud_cover,precipitation_probability") === -1)
    throw new Error("weather url missing hourly fields: " + u);

  var hours = weatherHours();
  if (!hours) throw new Error("weather hours not cached");
  if (hours.length !== 816)
    throw new Error("expected " + 816 + " hours, got " + hours.length);
  for (var i = 0; i < hours.length; i++) {
    var h = hours[i];
    if (!isFinite(h.ms) || h.cc < 0 || h.cc > 100 || h.pp < 0 || h.pp > 100)
      throw new Error("hour entry out of range at " + i);
  }
  var s = startUtcMs(), e = endUtcMs();
  var inRange = 0;
  hours.forEach(function (h) { if (h.ms >= s && h.ms <= e) inRange++; });
  if (inRange < 48) throw new Error("too few in-window hours: " + inRange);

  draw();
  var lg = $("legend").__innerHTML;
  if (lg.indexOf("Cloud cover (fill height = %)") === -1)
    throw new Error("legend missing the cloud cover entry: " + lg);
  if (lg.indexOf('class="sw"') === -1)
    throw new Error("legend swatch missing: " + lg);
  if (lg.indexOf("opacity:0.45") === -1)
    throw new Error("legend swatch missing cloud-cover alpha: " + lg);
  var status = String($("status").textContent);
  if (status !== "") throw new Error("weather fetch set status: " + status);
  var sum = $("summary").innerHTML;
  var pcts = (sum.match(/\d+% clear/g) || []).map(function (x) {
    return parseInt(x, 10);
  });
  if (pcts.length < 1)
    throw new Error("summary missing clear-sky fraction: " + sum.slice(0, 300));
  var n = __fetchLog.length;
  draw();
  if (__fetchLog.length !== n) throw new Error("cache not reused on second draw");

  __probe = {
    fetches: wxUrls.length, url: u, hours: hours.length,
    inWindow: inRange, pcts: pcts, status: status
  };
}
__main().then(
  function () { __driverDone = true; },
  function (err) { __driverError = String((err && err.message) || err) + "\n" + String((err && err.stack) || ""); __driverDone = true; }
);

"""


def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
