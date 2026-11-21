"""Interactive single-file HTML site for airmass charts (headless, github.io-ready).

The generated page is fully self-contained (no external files or network access):
the chart is drawn on a <canvas>, all astronomy math runs in embedded JavaScript,
and every control on the page is live. The workspace passed in becomes the
initial state; user edits persist in the browser via localStorage, and the page
can export the workspace back to a JSON file the desktop app can open.

The JS astronomy helpers live between the JS-CORE markers and are pure
(no DOM / no Intl) so tests can evaluate them in a JS engine and compare
against astropy.
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from .chart import (
    AXES_BG,
    BORDER,
    ERROR_FG,
    FIG_BG,
    FG,
    FG_DIM,
    PALETTE,
    SHADE,
)
from .observatories import PRESETS, from_dict, to_dict


def _blend_hex(fg_hex: str, alpha: float, bg_hex: str) -> str:
    """Blend a foreground color over a background color.

    Parameters
    ----------
    fg_hex : str
        Foreground color as '#rrggbb'.
    alpha : float
        Foreground opacity, 0 (transparent) to 1 (opaque).
    bg_hex : str
        Background color as '#rrggbb'.

    Returns
    -------
    str
        Resulting color as '#rrggbb'.
    """
    fg = tuple(int(fg_hex[i : i + 2], 16) for i in (1, 3, 5))
    bg = tuple(int(bg_hex[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(f * alpha + b * (1.0 - alpha)) for f, b in zip(fg, bg))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _js_json(obj) -> str:
    """Serialize a value to JSON that can be embedded in a <script> block.

    Parameters
    ----------
    obj
        Any JSON-serializable Python value.

    Returns
    -------
    str
        JSON text with every '</' escaped as '<\\/' so an embedded
        script cannot break out of its block.
    """
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Airmass chart &mdash; __OBS_NAME__</title>
<style>
:root {
  --bg: #1e1e2e;
  --bg-alt: #313244;
  --fg: #cdd6f4;
  --fg-dim: #9aa5c4;
  --accent: #89b4fa;
  --border: #45475a;
  --error: #f38ba8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.45;
}
.wrap { max-width: 1400px; margin: 0 auto; padding: 1.5rem 1.25rem 2rem; }
header h1 { margin: 0 0 0.25rem; font-size: 1.5rem; }
header p { margin: 0; color: var(--fg-dim); }
.layout { display: grid; grid-template-columns: 340px 1fr; gap: 1rem; margin-top: 1rem; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
.panel { background: var(--bg-alt); border: 1px solid var(--border);
         border-radius: 10px; padding: 0.75rem 0.9rem; margin-bottom: 0.9rem; }
.panel h2 { font-size: 0.75rem; color: var(--accent); margin: 0 0 0.4rem;
            text-transform: uppercase; letter-spacing: 0.05em; }
.panel h2:first-child { margin-top: 0; }
label.field { display: flex; align-items: center; gap: 0.5rem; margin: 0.22rem 0;
              font-size: 0.83rem; color: var(--fg-dim); }
label.field span { flex: 0 0 96px; }
input, select {
  background: var(--bg); color: var(--fg); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.24rem 0.4rem; font: inherit; font-size: 0.83rem;
  flex: 1; min-width: 0;
}
input:focus, select:focus { outline: none; border-color: var(--accent); }
input[type="color"] { padding: 0; width: 30px; height: 26px; flex: none; }
button {
  background: #45475a; color: var(--fg); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.28rem 0.7rem; font: inherit; font-size: 0.83rem;
  cursor: pointer;
}
button:hover { background: #585b70; }
.target-row { display: grid; grid-template-columns: 30px 1fr 1fr 1fr 22px;
              gap: 0.3rem; margin: 0.25rem 0; align-items: center; }
.target-row .rm { border: none; background: transparent; color: var(--fg-dim);
                  font-size: 1.05rem; padding: 0; }
.target-row .rm:hover { color: var(--error); }
.add-row { display: grid; grid-template-columns: 1fr 1fr 1fr auto auto;
           gap: 0.3rem; margin-top: 0.35rem; }
.actions { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.ws-row { display: grid; grid-template-columns: 1fr auto auto auto;
          gap: 0.3rem; margin-top: 0.45rem; }
input[type="checkbox"] { flex: none; }
.chart-col { min-width: 0; display: flex; flex-direction: column; }
.chart-box { background: var(--bg); border: 1px solid var(--border);
             border-radius: 10px; padding: 0.5rem; min-width: 0;
             position: relative; }
#summary-panel .sum-total { margin-bottom: 0.5rem; font-size: 0.82rem; color: var(--fg-dim); }
#summary-panel .sum-total b { color: var(--fg); }
#summary-panel .sum-target { margin: 0.35rem 0; font-size: 0.82rem; }
#summary-panel .sum-target b { color: var(--fg); }
#summary-panel .sum-day { display: block; margin-left: 1rem; color: var(--fg-dim);
                          font: 0.78rem/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
#summary-panel .sum-empty { color: var(--fg-dim); font-size: 0.8rem; }
canvas { display: block; width: 100%; }
#tooltip { position: absolute; display: none; pointer-events: none; z-index: 10;
           background: var(--bg-alt); border: 1px solid var(--border);
           border-radius: 6px; padding: 0.3rem 0.5rem;
           font: 0.72rem/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
           white-space: pre; color: var(--fg); }
#xhair { position: absolute; display: none; width: 1px; background: var(--border);
         pointer-events: none; z-index: 9; }
.status { min-height: 1.15em; font-size: 0.8rem; color: var(--error);
          margin-top: 0.4rem; word-break: break-word; }
footer { margin-top: 1.25rem; color: var(--fg-dim); font-size: 0.8rem; }
footer li { margin-bottom: 0.2rem; }
</style>
</head>
<body>
<div class="wrap">
<header>
<h1>Airmass chart &mdash; <span id="obs-title">__OBS_NAME__</span></h1>
<p>Airmass (X) for the targets below, as seen from this observatory.
   Every control is live; your edits are saved in this browser and encoded in
   the page URL, so the address bar is a shareable link.</p>
</header>
<div class="layout">
<aside>
  <div class="panel">
    <h2>Observatory</h2>
    <label class="field"><span>Preset</span><select id="preset"></select></label>
    <label class="field"><span>Latitude (&deg;)</span><input id="lat" type="number" step="0.00001"></label>
    <label class="field"><span>Longitude (&deg;)</span><input id="lon" type="number" step="0.00001"></label>
    <label class="field"><span>Elevation (m)</span><input id="elev" type="number" step="1"></label>
    <label class="field"><span>Time zone</span><input id="tz" type="text" placeholder="IANA, e.g. America/Phoenix"></label>
  </div>
  <div class="panel">
    <h2>Time range (local)</h2>
    <label class="field"><span>Start</span><input id="start" type="datetime-local"></label>
    <label class="field"><span>Duration (days)</span><input id="days" type="number" step="0.1" min="0.25" max="30"></label>
  </div>
  <div class="panel">
    <h2>Display</h2>
    <label class="field"><span>Windows</span><input id="windows-chk" type="checkbox" checked></label>
    <label class="field"><span>Max airmass</span><input id="threshold" type="number" step="0.1" min="1.05"></label>
    <label class="field"><span>Moon</span><input id="moon-chk" type="checkbox"></label>
    <label class="field"><span>Civil twilight</span><input id="civil-chk" type="checkbox"></label>
    <label class="field"><span>Nautical twilight</span><input id="nautical-chk" type="checkbox"></label>
    <label class="field"><span>Dawn/dusk ticks</span><input id="dusk-chk" type="checkbox" checked></label>
  </div>
  <div class="panel">
    <h2>Targets</h2>
    <div id="targets"></div>
    <div class="add-row">
      <input id="new-name" placeholder="Name">
      <input id="new-ra" placeholder="RA  h:m:s / deg">
      <input id="new-dec" placeholder="Dec d:m:s / deg">
      <button id="simbad-btn" title="Look up this name in CDS Simbad">Simbad</button>
      <button id="add-btn">Add</button>
    </div>
  </div>
  <div class="panel">
    <h2>Workspace</h2>
    <div class="actions">
      <button id="copylink-btn" title="Copy a link that opens this exact setup">Copy link</button>
      <button id="import-btn" title="Load a workspace from an exported JSON file">Import JSON</button>
      <button id="png-btn" title="Download the chart as a PNG image">PNG</button>
      <button id="export-btn">Export JSON</button>
      <button id="reset-btn">Reset</button>
    </div>
    <div class="ws-row">
      <select id="ws-select" title="Workspaces saved in this browser"></select>
      <button id="ws-save-btn" title="Save the current workspace under a name">Save</button>
      <button id="ws-load-btn" title="Load the selected workspace">Load</button>
      <button id="ws-delete-btn" title="Delete the selected workspace">Del</button>
    </div>
    <input type="file" id="import-file" accept=".json,application/json" hidden>
    <div class="status" id="status"></div>
  </div>
</aside>
<main class="chart-col">
  <div class="chart-box">
    <canvas id="chart"></canvas>
    <div id="xhair"></div>
    <div id="tooltip"></div>
  </div>
  <div class="panel" id="summary-panel">
    <h2>Observing windows</h2>
    <div id="summary"></div>
  </div>
</main>
</div>
<footer>
<ul>
<li>Airmass: Kasten &amp; Young (1989), normalized so X(90&deg;) = 1; curves are hidden below the horizon and the dotted line marks X = 2.</li>
<li>Twilight shading marks day / civil / nautical / astro / night.</li>
<li>Entered ICRS coordinates are precessed to the mean equator/equinox of date (IAU 2006; nutation omitted), so altitudes agree with the desktop app to ~0.01&deg;. The Simbad button looks a name up in CDS Simbad (needs internet; mirrors tried in order).</li>
<li>Chart navigation: the wheel pans the time window, Ctrl/&#8984;+wheel zooms around the cursor, dragging the left or right plot edge resizes the window, dragging inside pans, and double-clicking resets to the full range. Hovering the plot shows each target&rsquo;s altitude, azimuth, airmass and parallactic angle at that instant.</li>
<li>Initial workspace generated by airmass-gui on __GENERATED__; edits stay in this browser (localStorage) until you export or reset.</li>
<li>Sharing: the URL encodes the full workspace (&#35;w= fragment) — Copy link in the Workspace panel copies it, and opening it loads that exact setup.</li>
<li>Workspaces: save named copies in this browser, import an exported JSON file, or download the chart as a PNG image (Export JSON keeps the desktop-app file shape).</li>
<li>Windows: the dashed line marks your airmass threshold; shaded bands show when each target is usable, colored ticks on the time axis mark transits, and the panel below the chart lists per-night windows, transit times and airmass.</li>
<li>Moon: with Moon checked, a dashed curve shows the Moon&rsquo;s airmass and the tooltip adds its altitude, illumination fraction, and angular separation from each target.</li>
<li>Touch and keys: a single finger pans, two fingers pinch to zoom, and a tap shows the tooltip; the arrow keys pan, + / - zoom around the center, and 0 or r resets the window. With Civil / Nautical twilight checked, thin vertical lines mark the sun&rsquo;s -6&deg; and -12&deg; crossings.</li>
</ul>
</footer>
</div>
<script>
"use strict";

/*JS-CORE-BEGIN*/
// Pure astronomy / parsing helpers (no DOM, no Intl).
// Cross-checked against astropy in tests/test_site.py.

/**
 * Wrap an angle into [0, 360) degrees.
 *
 * @param {number} x - Angle in degrees, any sign.
 * @returns {number} Equivalent angle in [0, 360).
 */
function wrap360(x) { return ((x % 360) + 360) % 360; }
/**
 * Convert degrees to radians.
 *
 * @param {number} x - Angle in degrees.
 * @returns {number} Angle in radians.
 */
function d2r(x) { return x * Math.PI / 180; }
/**
 * Convert radians to degrees.
 *
 * @param {number} x - Angle in radians.
 * @returns {number} Angle in degrees.
 */
function r2d(x) { return x * 180 / Math.PI; }

/**
 * Convert Unix epoch milliseconds to Julian Date.
 *
 * @param {number} ms - UTC epoch milliseconds.
 * @returns {number} Julian Date.
 */
function jdFromUtcMs(ms) { return ms / 86400000 + 2440587.5; }

/**
 * Greenwich mean sidereal time from Meeus (1998).
 *
 * Agrees with the IAU-2006 definition to < 0.0011 deg.
 *
 * @param {number} jd - Julian Date.
 * @returns {number} GMST in degrees, [0, 360).
 */
function gmstMeeus(jd) {
  var d = jd - 2451545.0;
  var t = d / 36525.0;
  return wrap360(280.46061837 + 360.98564736629 * d + 0.000387933 * t + (t * t) / 38710000);
}

/**
 * Local sidereal time for a geodetic longitude.
 *
 * @param {number} jd - Julian Date.
 * @param {number} lonDeg - Geodetic longitude in degrees, positive east.
 * @returns {number} LST in degrees, [0, 360).
 */
function lstDeg(jd, lonDeg) { return wrap360(gmstMeeus(jd) + lonDeg); }

/**
 * Geometric altitude of an equatorial source.
 *
 * @param {number} raH - Right ascension in hours.
 * @param {number} decDeg - Declination in degrees.
 * @param {number} jd - Julian Date.
 * @param {number} latDeg - Observer latitude in degrees.
 * @param {number} lonDeg - Observer longitude in degrees, positive east.
 * @returns {number} Altitude in degrees; negative below the horizon.
 */
function altitude(raH, decDeg, jd, latDeg, lonDeg) {
  var ha = d2r(wrap360(lstDeg(jd, lonDeg) - raH * 15));
  var s = Math.sin(d2r(latDeg)) * Math.sin(d2r(decDeg)) +
          Math.cos(d2r(latDeg)) * Math.cos(d2r(decDeg)) * Math.cos(ha);
  return r2d(Math.asin(Math.max(-1, Math.min(1, s))));
}

/**
 * Horizon coordinates of an equatorial source.
 *
 * @param {number} raH - Right ascension in hours.
 * @param {number} decDeg - Declination in degrees.
 * @param {number} jd - Julian Date.
 * @param {number} latDeg - Observer latitude in degrees.
 * @param {number} lonDeg - Observer longitude in degrees, positive east.
 * @returns {{alt: number, az: number}} Altitude in degrees and azimuth in
 *     degrees, clockwise from north through east.
 */
function horizonCoords(raH, decDeg, jd, latDeg, lonDeg) {
  var ha = d2r(wrap360(lstDeg(jd, lonDeg) - raH * 15));
  var phi = d2r(latDeg), dec = d2r(decDeg);
  var cp = Math.cos(phi), sp = Math.sin(phi);
  var cd = Math.cos(dec), sd = Math.sin(dec);
  var ch = Math.cos(ha), sh = Math.sin(ha);
  var zUp = sd * sp + cd * cp * ch;
  var alt = r2d(Math.asin(Math.max(-1, Math.min(1, zUp))));
  var az = wrap360(r2d(Math.atan2(-cd * sh, sd * cp - cd * sp * ch)));
  return { alt: alt, az: az };
}

/**
 * Parallactic angle q: angle at the object between the vertical circle and the hour circle.
 *
 * 0 on the meridian, positive west of it; NaN where the hour circle is
 * undefined (object at the celestial pole or at the zenith).
 *
 * @param {number} raH - Right ascension in hours.
 * @param {number} decDeg - Declination in degrees.
 * @param {number} jd - Julian Date.
 * @param {number} latDeg - Observer latitude in degrees.
 * @param {number} lonDeg - Observer longitude in degrees, positive east.
 * @returns {number} q in degrees, or NaN.
 */
function parallacticAngle(raH, decDeg, jd, latDeg, lonDeg) {
  var hor = horizonCoords(raH, decDeg, jd, latDeg, lonDeg);
  if (Math.abs(decDeg) >= 89.99 || hor.alt >= 89.99) return NaN;
  var ha = d2r(wrap360(lstDeg(jd, lonDeg) - raH * 15));
  var phi = d2r(latDeg), dec = d2r(decDeg);
  return r2d(Math.atan2(
    Math.cos(phi) * Math.sin(ha),
    Math.sin(phi) * Math.cos(dec) - Math.cos(phi) * Math.sin(dec) * Math.cos(ha)
  ));
}

// Cubic-in-T coefficients (a0..a3) of element (i,j) of the IAU 2006
// precession matrix; see precessIcrsToTete.
var PREC = [
  [[0.9999999996062562, -2.1969297619512713e-08, -0.00029719232752057473, -1.0114765190709189e-07],
   [-6.441078856070561e-11, -0.02236038696016588, -6.7450623683643375e-06, 2.2182280752204036e-06],
   [-1.8423325298910764e-11, -0.00971659701165348, 2.0810738950801485e-06, 9.63543008198074e-07]],
  [[6.934748246821439e-11, 0.022360387230982312, 6.746186239891443e-06, -2.2185904659856523e-06],
   [0.9999999996697424, -1.848286552434681e-08, -0.0002499873528838399, -1.2605619559802516e-07],
   [-1.444713293875262e-10, 1.1680395998077555e-07, -0.0001086306631635869, -1.7402145308112042e-08]],
  [[7.123689993411714e-12, 0.009716596387706123, -2.0836605130397537e-06, -9.627094011907582e-07],
   [-1.4592286977355824e-10, -1.3290190160297146e-07, -0.000108630885301105, 2.0003846174793192e-08],
   [0.9999999999365136, -3.486436381556712e-09, -4.7204974649895375e-05, 2.4908549929831116e-08]]
];

/**
 * Precess ICRS coordinates to the mean equator/equinox of date.
 *
 * IAU 2006 precession, nutation omitted. PREC is a cubic-in-T fit of the
 * ERFA pn06 matrix over T in [-1, 2] (residual < 1e-8 rad).
 *
 * @param {number} raDeg - Right ascension in degrees, J2000.
 * @param {number} decDeg - Declination in degrees, J2000.
 * @param {number} jd - Julian Date of the target epoch.
 * @returns {{raDeg: number, dec: number}} Coordinates in the mean
 *     equator/equinox of date, degrees.
 */
function precessIcrsToTete(raDeg, decDeg, jd) {
  var T = (jd - 2451545.0) / 36525.0;
  var T2 = T * T, T3 = T2 * T;
  var x = Math.cos(d2r(decDeg)) * Math.cos(d2r(raDeg));
  var y = Math.cos(d2r(decDeg)) * Math.sin(d2r(raDeg));
  var z = Math.sin(d2r(decDeg));
  var M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  var i, j, c;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      c = PREC[i][j];
      M[i][j] = c[0] + c[1] * T + c[2] * T2 + c[3] * T3;
    }
  }
  var wx = M[0][0] * x + M[0][1] * y + M[0][2] * z;
  var wy = M[1][0] * x + M[1][1] * y + M[1][2] * z;
  var wz = M[2][0] * x + M[2][1] * y + M[2][2] * z;
  var decT = r2d(Math.asin(Math.max(-1, Math.min(1, wz))));
  var raT = wrap360(r2d(Math.atan2(wy, wx)));
  return { raDeg: raT, dec: decT };
}
/**
 * Parse a SIMBAD ASCII object record.
 *
 * Name comes from the first non-C.D.S. line of the header; RA/Dec from
 * the ICRS coordinate line.
 *
 * @param {string} text - Response body for output.format=ASCII.
 * @returns {?{name: string, raDeg: number, dec: number}} J2000 name, RA
 *     in degrees and Dec in degrees, or null when no coordinates are
 *     present.
 */
function simbadParseRecord(text) {
  var lines = String(text).split("\n");
  var i, s, m;
  var name = null;
  for (i = 0; i < lines.length && i < 4; i++) {
    s = lines[i].replace(/^\s+|\s+$/g, "");
    if (s !== "" && s.indexOf("C.D.S") !== 0) { name = s; break; }
  }
  var re = /Coordinates\(ICRS,ep=J2000,eq=2000\):\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([+-]?\d+)\s+(\d+)\s+([\d.]+)/;
  for (i = 0; i < lines.length; i++) {
    m = re.exec(lines[i]);
    if (m) {
      var raDeg = (parseFloat(m[1]) + parseFloat(m[2]) / 60 + parseFloat(m[3]) / 3600) * 15;
      var dec = Math.abs(parseFloat(m[4])) + parseFloat(m[5]) / 60 + parseFloat(m[6]) / 3600;
      if (m[4].charAt(0) === "-") dec = -dec;
      return { name: name || "Unknown", raDeg: raDeg, dec: dec };
    }
  }
  return null;
}


// Kasten & Young (1989) airmass at 90 deg; normalizes the formula to 1 at the zenith.
var ZENITH_X = 1 / (1 + 0.50572 * Math.pow(96.07995, -1.6364));

/**
 * Kasten & Young (1989) airmass, normalized so X(90 deg) = 1.
 *
 * @param {number} altDeg - Altitude in degrees; negative values
 *     extrapolate below the horizon.
 * @returns {number} Airmass X.
 */
function kastyAir(altDeg) {
  return (1 / (Math.sin(d2r(altDeg)) + 0.50572 * Math.pow(altDeg + 6.07995, -1.6364))) / ZENITH_X;
}

/**
 * Airmass for altitude in degrees; Infinity below the horizon.
 *
 * @param {number} altDeg - Altitude in degrees.
 * @returns {number} Airmass X, or Infinity at or below 0 deg.
 */
function airmass(altDeg) { return altDeg > 0 ? kastyAir(altDeg) : Infinity; }

/**
 * Airmass at the horizon (altitude 0 deg), ~= 37.93.
 *
 * @returns {number} Airmass X.
 */
function horizonX() { return kastyAir(0); }

/**
 * Invert the airmass formula: X -> altitude in degrees.
 *
 * Bisection over [0, 90] deg; X is clipped to [1, horizonX()].
 *
 * @param {number} x - Airmass.
 * @returns {number} Altitude in degrees.
 */
function altFromAirmass(x) {
  var target = Math.min(Math.max(x, 1), horizonX());
  if (target <= 1) return 90;
  var lo = 0, hi = 90;
  for (var i = 0; i < 100; i++) {
    var mid = 0.5 * (lo + hi);
    if (hi - lo <= 1e-9) break;
    if (kastyAir(mid) > target) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/**
 * Low-precision geocentric Sun position (Meeus).
 *
 * @param {number} jd - Julian Date.
 * @returns {{raH: number, dec: number}} RA in hours and Dec in degrees.
 */
function sunRaDec(jd) {
  var n = jd - 2451545.0;
  var L = 280.460 + 0.9856474 * n;
  var g = d2r(357.529 + 0.9856003 * n);
  var lam = d2r(L + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g));
  var eps = d2r(23.439 - 0.0000004 * n);
  var dec = r2d(Math.asin(Math.sin(eps) * Math.sin(lam)));
  var raH = wrap360(r2d(Math.atan2(Math.cos(eps) * Math.sin(lam), Math.cos(lam)))) / 15;
  return { raH: raH, dec: dec };
}

/**
 * Sun's geometric altitude for an observer.
 *
 * @param {number} jd - Julian Date.
 * @param {number} latDeg - Observer latitude in degrees.
 * @param {number} lonDeg - Observer longitude in degrees, positive east.
 * @returns {number} Solar altitude in degrees.
 */
function sunAltitude(jd, latDeg, lonDeg) {
  var sun = sunRaDec(jd);
  return altitude(sun.raH, sun.dec, jd, latDeg, lonDeg);
}

/**
 * Classify sky darkness from solar altitude.
 *
 * @param {number} altDeg - Solar altitude in degrees.
 * @returns {string} One of "day", "civil", "nautical", "astro", "night",
 *     per the 0/-6/-12/-18 deg thresholds.
 */
function classifySunAlt(altDeg) {
  if (altDeg > 0) return "day";
  if (altDeg > -6) return "civil";
  if (altDeg > -12) return "nautical";
  if (altDeg > -18) return "astro";
  return "night";
}

/**
 * Split a time span into solar-shading intervals.
 *
 * Sun-altitude crossings of 0/-6/-12/-18 deg are located on a 10-min
 * sample grid with linear interpolation.
 *
 * @param {number} startMs - Span start, epoch ms.
 * @param {number} endMs - Span end, epoch ms.
 * @param {number} latDeg - Observer latitude in degrees.
 * @param {number} lonDeg - Observer longitude in degrees, positive east.
 * @returns {number[][]} Rows [startMs, endMs, kind], kind per
 *     classifySunAlt.
 */
function sunShading(startMs, endMs, latDeg, lonDeg) {
  var step = 10 * 60 * 1000;
  var n = Math.ceil((endMs - startMs) / step) + 1;
  var t = new Array(n), alt = new Array(n);
  var i;
  for (i = 0; i < n; i++) {
    t[i] = startMs + i * step;
    alt[i] = sunAltitude(jdFromUtcMs(t[i]), latDeg, lonDeg);
  }
  var events = [];
  var thresholds = [0, -6, -12, -18];
  for (var thI = 0; thI < thresholds.length; thI++) {
    var th = thresholds[thI];
    for (i = 0; i < n - 1; i++) {
      var a0 = alt[i] - th, a1 = alt[i + 1] - th;
      if (a0 === 0) continue;
      if (Math.sign(a0) !== Math.sign(a1)) {
        events.push(t[i] + (a0 / (a0 - a1)) * (t[i + 1] - t[i]));
      }
    }
  }
  events.sort(function (a, b) { return a - b; });
  var bounds = [startMs].concat(events, [endMs]);
  var out = [];
  for (i = 0; i < bounds.length - 1; i++) {
    var a = bounds[i], b = bounds[i + 1];
    if (b - a < 86.4) continue; // 1e-6 day, in ms
    out.push([a, b, classifySunAlt(sunAltitude(jdFromUtcMs(0.5 * (a + b)), latDeg, lonDeg))]);
  }
  return out;
}

/**
 * Contiguous true runs in a boolean sample array.
 *
 * Samples sit at startMs + i * stepMs; run endpoints extend half a step
 * in each direction.
 *
 * @param {boolean[]} ok - Sample flags.
 * @param {number} startMs - Epoch ms of the first sample.
 * @param {number} stepMs - Sample spacing, ms.
 * @returns {number[][]} Rows [startMs, endMs].
 */
function trueRuns(ok, startMs, stepMs) {
  var runs = [], i = 0, n = ok.length;
  while (i < n) {
    if (!ok[i]) { i++; continue; }
    var j = i;
    while (j + 1 < n && ok[j + 1]) j++;
    runs.push([startMs + i * stepMs - stepMs / 2, startMs + j * stepMs + stepMs / 2]);
    i = j + 1;
  }
  return runs;
}

/**
 * Transit times: instants when the local sidereal time passes the target's RA.
 *
 * The wrapped LST-RA difference crossing 180 deg downward; 10-min scan
 * with bisection refinement.
 *
 * @param {number} raH - Target right ascension in hours.
 * @param {number} startMs - Scan start, epoch ms.
 * @param {number} endMs - Scan end, epoch ms.
 * @param {number} lonDeg - Observer longitude in degrees, positive east.
 * @returns {number[]} Transit times in epoch ms.
 */
function transitTimes(raH, startMs, endMs, lonDeg) {
  var times = [];
  var step = 10 * 60 * 1000;
  var prev = wrap360(lstDeg(jdFromUtcMs(startMs), lonDeg) - raH * 15);
  for (var t = startMs + step; t <= endMs; t += step) {
    var d = wrap360(lstDeg(jdFromUtcMs(t), lonDeg) - raH * 15);
    if (prev > 180 && d <= 180) {
      var lo = t - step, hi = t;
      for (var it = 0; it < 50; it++) {
        var mid = 0.5 * (lo + hi);
        var dm = wrap360(lstDeg(jdFromUtcMs(mid), lonDeg) - raH * 15);
        if (dm > 180) lo = mid; else hi = mid;
      }
      times.push(0.5 * (lo + hi));
    }
    prev = d;
  }
  return times;
}

/**
 * Times at which the Sun's altitude crosses a threshold.
 *
 * Both rising and falling crossings; 10-min scan with bisection
 * refinement.
 *
 * @param {number} startMs - Scan start, epoch ms.
 * @param {number} endMs - Scan end, epoch ms.
 * @param {number} latDeg - Observer latitude in degrees.
 * @param {number} lonDeg - Observer longitude in degrees, positive east.
 * @param {number} altThreshold - Threshold solar altitude in degrees.
 * @returns {number[]} Crossing times in epoch ms.
 */
function twilightEvents(startMs, endMs, latDeg, lonDeg, altThreshold) {
  var out = [];
  var step = 10 * 60 * 1000;
  var prev = sunAltitude(jdFromUtcMs(startMs), latDeg, lonDeg) - altThreshold;
  for (var t = startMs + step; t <= endMs; t += step) {
    var a = sunAltitude(jdFromUtcMs(t), latDeg, lonDeg) - altThreshold;
    if ((prev < 0 && a >= 0) || (prev > 0 && a <= 0)) {
      var lo = t - step, hi = t;
      var fLo = prev;
      for (var it = 0; it < 50; it++) {
        var mid = 0.5 * (lo + hi);
        var fMid = sunAltitude(jdFromUtcMs(mid), latDeg, lonDeg) - altThreshold;
        if ((fLo < 0 && fMid > 0) || (fLo > 0 && fMid < 0)) { hi = mid; }
        else { lo = mid; fLo = fMid; }
      }
      out.push(0.5 * (lo + hi));
    }
    prev = a;
  }
  return out;
}

/**
 * Times at which the Sun crosses 0, -6, -12, -18 deg, with direction.
 *
 * The IAU twilight limits, lettered S (sunrise/sunset), C (civil),
 * N (nautical) and A (astronomical). Built on twilightEvents; each
 * crossing records the Sun's direction, dir, of +1 rising or -1
 * setting, from the altitude 5 min either side of the crossing.
 *
 * @param {number} startMs - Scan start, epoch ms.
 * @param {number} endMs - Scan end, epoch ms.
 * @param {number} latDeg - Observer latitude in degrees.
 * @param {number} lonDeg - Observer longitude in degrees, positive east.
 * @returns {Object[]} Crossings sorted by time; each entry is
 *     {ms (epoch ms), alt (the threshold in degrees), dir (+1 or -1)}.
 */
function sunEvents(startMs, endMs, latDeg, lonDeg) {
  var thresholds = [0, -6, -12, -18];
  var out = [];
  for (var ti = 0; ti < thresholds.length; ti++) {
    var evs = twilightEvents(startMs, endMs, latDeg, lonDeg, thresholds[ti]);
    for (var ei = 0; ei < evs.length; ei++) {
      var a = sunAltitude(jdFromUtcMs(evs[ei] - 5 * 60 * 1000), latDeg, lonDeg);
      var b = sunAltitude(jdFromUtcMs(evs[ei] + 5 * 60 * 1000), latDeg, lonDeg);
      out.push({ ms: evs[ei], alt: thresholds[ti], dir: b > a ? 1 : -1 });
    }
  }
  out.sort(function (p, q) { return p.ms - q.ms; });
  return out;
}

// ------------------------------------------------------------------- Moon
// Truncated NAO 1954/ELP theory (105 terms): geocentric ecliptic
// coordinates in the mean equinox of date, nutation omitted. Ported to
// ES5 from the astronomy-engine CalcMoon reference
// (https://github.com/cosinekitty/astronomy, MIT license); the
// mean-equinox-of-date convention matches precessIcrsToTete above.
// Agrees with astropy's geocentric apparent position to ~0.015 deg
// (the residual is nutation + delta T).
var MOON_PI2 = 2 * Math.PI;
var MOON_ARC = 3600 * (180 / Math.PI); // arcseconds per radian
var MOON_EARTH_RADIUS_AU = 6378.1366 / 1.4959787069098932e+8;
/**
 * Fractional part of x.
 *
 * @param {number} x - Any real number.
 * @returns {number} x - floor(x), in [0, 1).
 */
function moFrac(x) { return x - Math.floor(x); }
/**
 * Mean obliquity of the ecliptic.
 *
 * IAU 1980 polynomial in T (Julian centuries from J2000).
 *
 * @param {number} jd - Julian Date.
 * @returns {number} Obliquity in degrees.
 */
function meanObliqDeg(jd) {
  var t = (jd - 2451545.0) / 36525.0;
  var asec = (((((-0.0000000434 * t
      - 0.000000576) * t
      + 0.00200340) * t
      - 0.0001831) * t
      - 46.836769) * t + 84381.406);
  return asec / 3600.0;
}
/**
 * Geocentric Moon position from the truncated NAO 1954/ELP theory.
 *
 * 105 terms; ecliptic coordinates in the mean equinox of date, nutation
 * omitted. Agrees with astropy's geocentric apparent position to
 * ~0.015 deg (residual is nutation + delta T).
 *
 * @param {number} jd - Julian Date.
 * @returns {{geo_eclip_lon: number, geo_eclip_lat: number, distance_au: number}}
 *     Ecliptic longitude in radians, latitude in radians, and geocentric
 *     distance in AU.
 */
function calcMoon(jd) {
    var T = (jd - 2451545.0) / 36525;
    function DeclareArray1(xmin, xmax) {
        var array = [];
        var i;
        for (i = 0; i <= xmax - xmin; ++i) {
            array.push(0);
        }
        return { min: xmin, array: array };
    }
    function DeclareArray2(xmin, xmax, ymin, ymax) {
        var array = [];
        for (var i = 0; i <= xmax - xmin; ++i) {
            array.push(DeclareArray1(ymin, ymax));
        }
        return { min: xmin, array: array };
    }
    function ArrayGet2(a, x, y) {
        var m = a.array[x - a.min];
        return m.array[y - m.min];
    }
    function ArraySet2(a, x, y, v) {
        var m = a.array[x - a.min];
        m.array[y - m.min] = v;
    }
    var S, MAX, ARG, FAC, I, J, T2, DGAM, DLAM, N, GAM1C, SINPI, L0, L, LS, F, D, DL0, DL, DLS, DF, DD, DS;
    var coArray = DeclareArray2(-6, 6, 1, 4);
    var siArray = DeclareArray2(-6, 6, 1, 4);
    function CO(x, y) {
        return ArrayGet2(coArray, x, y);
    }
    function SI(x, y) {
        return ArrayGet2(siArray, x, y);
    }
    function SetCO(x, y, v) {
        return ArraySet2(coArray, x, y, v);
    }
    function SetSI(x, y, v) {
        return ArraySet2(siArray, x, y, v);
    }
    function AddThe(c1, s1, c2, s2, func) {
        func(c1 * c2 - s1 * s2, s1 * c2 + c1 * s2);
    }
    function Sine(phi) {
        return Math.sin(MOON_PI2 * phi);
    }
    T2 = T * T;
    DLAM = 0;
    DS = 0;
    GAM1C = 0;
    SINPI = 3422.7000;
    var S1 = Sine(0.19833 + 0.05611 * T);
    var S2 = Sine(0.27869 + 0.04508 * T);
    var S3 = Sine(0.16827 - 0.36903 * T);
    var S4 = Sine(0.34734 - 5.37261 * T);
    var S5 = Sine(0.10498 - 5.37899 * T);
    var S6 = Sine(0.42681 - 0.41855 * T);
    var S7 = Sine(0.14943 - 5.37511 * T);
    DL0 = 0.84 * S1 + 0.31 * S2 + 14.27 * S3 + 7.26 * S4 + 0.28 * S5 + 0.24 * S6;
    DL = 2.94 * S1 + 0.31 * S2 + 14.27 * S3 + 9.34 * S4 + 1.12 * S5 + 0.83 * S6;
    DLS = -6.40 * S1 - 1.89 * S6;
    DF = 0.21 * S1 + 0.31 * S2 + 14.27 * S3 - 88.70 * S4 - 15.30 * S5 + 0.24 * S6 - 1.86 * S7;
    DD = DL0 - DLS;
    DGAM = (-3332E-9 * Sine(0.59734 - 5.37261 * T)
        - 539E-9 * Sine(0.35498 - 5.37899 * T)
        - 64E-9 * Sine(0.39943 - 5.37511 * T));
    L0 = MOON_PI2 * moFrac(0.60643382 + 1336.85522467 * T - 0.00000313 * T2) + DL0 / MOON_ARC;
    L = MOON_PI2 * moFrac(0.37489701 + 1325.55240982 * T + 0.00002565 * T2) + DL / MOON_ARC;
    LS = MOON_PI2 * moFrac(0.99312619 + 99.99735956 * T - 0.00000044 * T2) + DLS / MOON_ARC;
    F = MOON_PI2 * moFrac(0.25909118 + 1342.22782980 * T - 0.00000892 * T2) + DF / MOON_ARC;
    D = MOON_PI2 * moFrac(0.82736186 + 1236.85308708 * T - 0.00000397 * T2) + DD / MOON_ARC;
    for (I = 1; I <= 4; ++I) {
        switch (I) {
            case 1:
                ARG = L;
                MAX = 4;
                FAC = 1.000002208;
                break;
            case 2:
                ARG = LS;
                MAX = 3;
                FAC = 0.997504612 - 0.002495388 * T;
                break;
            case 3:
                ARG = F;
                MAX = 4;
                FAC = 1.000002708 + 139.978 * DGAM;
                break;
            case 4:
                ARG = D;
                MAX = 6;
                FAC = 1.0;
                break;
            default: throw "Internal error: I = " + I;
        }
        SetCO(0, I, 1);
        SetCO(1, I, Math.cos(ARG) * FAC);
        SetSI(0, I, 0);
        SetSI(1, I, Math.sin(ARG) * FAC);
        for (J = 2; J <= MAX; ++J) {
            AddThe(CO(J - 1, I), SI(J - 1, I), CO(1, I), SI(1, I), function (c, s) { SetCO(J, I, c); SetSI(J, I, s); });
        }
        for (J = 1; J <= MAX; ++J) {
            SetCO(-J, I, CO(J, I));
            SetSI(-J, I, -SI(J, I));
        }
    }
    function Term(p, q, r, s) {
        var result = { x: 1, y: 0 };
        var I = [0, p, q, r, s]; // I[0] is not used; it is a placeholder
        for (var k = 1; k <= 4; ++k)
            if (I[k] !== 0)
                AddThe(result.x, result.y, CO(I[k], k), SI(I[k], k), function (c, s) { result.x = c; result.y = s; });
        return result;
    }
    function AddSol(coeffl, coeffs, coeffg, coeffp, p, q, r, s) {
        var result = Term(p, q, r, s);
        DLAM += coeffl * result.y;
        DS += coeffs * result.y;
        GAM1C += coeffg * result.x;
        SINPI += coeffp * result.x;
    }
    AddSol(13.9020, 14.0600, -0.0010, 0.2607, 0, 0, 0, 4);
    AddSol(0.4030, -4.0100, 0.3940, 0.0023, 0, 0, 0, 3);
    AddSol(2369.9120, 2373.3600, 0.6010, 28.2333, 0, 0, 0, 2);
    AddSol(-125.1540, -112.7900, -0.7250, -0.9781, 0, 0, 0, 1);
    AddSol(1.9790, 6.9800, -0.4450, 0.0433, 1, 0, 0, 4);
    AddSol(191.9530, 192.7200, 0.0290, 3.0861, 1, 0, 0, 2);
    AddSol(-8.4660, -13.5100, 0.4550, -0.1093, 1, 0, 0, 1);
    AddSol(22639.5000, 22609.0700, 0.0790, 186.5398, 1, 0, 0, 0);
    AddSol(18.6090, 3.5900, -0.0940, 0.0118, 1, 0, 0, -1);
    AddSol(-4586.4650, -4578.1300, -0.0770, 34.3117, 1, 0, 0, -2);
    AddSol(3.2150, 5.4400, 0.1920, -0.0386, 1, 0, 0, -3);
    AddSol(-38.4280, -38.6400, 0.0010, 0.6008, 1, 0, 0, -4);
    AddSol(-0.3930, -1.4300, -0.0920, 0.0086, 1, 0, 0, -6);
    AddSol(-0.2890, -1.5900, 0.1230, -0.0053, 0, 1, 0, 4);
    AddSol(-24.4200, -25.1000, 0.0400, -0.3000, 0, 1, 0, 2);
    AddSol(18.0230, 17.9300, 0.0070, 0.1494, 0, 1, 0, 1);
    AddSol(-668.1460, -126.9800, -1.3020, -0.3997, 0, 1, 0, 0);
    AddSol(0.5600, 0.3200, -0.0010, -0.0037, 0, 1, 0, -1);
    AddSol(-165.1450, -165.0600, 0.0540, 1.9178, 0, 1, 0, -2);
    AddSol(-1.8770, -6.4600, -0.4160, 0.0339, 0, 1, 0, -4);
    AddSol(0.2130, 1.0200, -0.0740, 0.0054, 2, 0, 0, 4);
    AddSol(14.3870, 14.7800, -0.0170, 0.2833, 2, 0, 0, 2);
    AddSol(-0.5860, -1.2000, 0.0540, -0.0100, 2, 0, 0, 1);
    AddSol(769.0160, 767.9600, 0.1070, 10.1657, 2, 0, 0, 0);
    AddSol(1.7500, 2.0100, -0.0180, 0.0155, 2, 0, 0, -1);
    AddSol(-211.6560, -152.5300, 5.6790, -0.3039, 2, 0, 0, -2);
    AddSol(1.2250, 0.9100, -0.0300, -0.0088, 2, 0, 0, -3);
    AddSol(-30.7730, -34.0700, -0.3080, 0.3722, 2, 0, 0, -4);
    AddSol(-0.5700, -1.4000, -0.0740, 0.0109, 2, 0, 0, -6);
    AddSol(-2.9210, -11.7500, 0.7870, -0.0484, 1, 1, 0, 2);
    AddSol(1.2670, 1.5200, -0.0220, 0.0164, 1, 1, 0, 1);
    AddSol(-109.6730, -115.1800, 0.4610, -0.9490, 1, 1, 0, 0);
    AddSol(-205.9620, -182.3600, 2.0560, 1.4437, 1, 1, 0, -2);
    AddSol(0.2330, 0.3600, 0.0120, -0.0025, 1, 1, 0, -3);
    AddSol(-4.3910, -9.6600, -0.4710, 0.0673, 1, 1, 0, -4);
    AddSol(0.2830, 1.5300, -0.1110, 0.0060, 1, -1, 0, 4);
    AddSol(14.5770, 31.7000, -1.5400, 0.2302, 1, -1, 0, 2);
    AddSol(147.6870, 138.7600, 0.6790, 1.1528, 1, -1, 0, 0);
    AddSol(-1.0890, 0.5500, 0.0210, 0.0000, 1, -1, 0, -1);
    AddSol(28.4750, 23.5900, -0.4430, -0.2257, 1, -1, 0, -2);
    AddSol(-0.2760, -0.3800, -0.0060, -0.0036, 1, -1, 0, -3);
    AddSol(0.6360, 2.2700, 0.1460, -0.0102, 1, -1, 0, -4);
    AddSol(-0.1890, -1.6800, 0.1310, -0.0028, 0, 2, 0, 2);
    AddSol(-7.4860, -0.6600, -0.0370, -0.0086, 0, 2, 0, 0);
    AddSol(-8.0960, -16.3500, -0.7400, 0.0918, 0, 2, 0, -2);
    AddSol(-5.7410, -0.0400, 0.0000, -0.0009, 0, 0, 2, 2);
    AddSol(0.2550, 0.0000, 0.0000, 0.0000, 0, 0, 2, 1);
    AddSol(-411.6080, -0.2000, 0.0000, -0.0124, 0, 0, 2, 0);
    AddSol(0.5840, 0.8400, 0.0000, 0.0071, 0, 0, 2, -1);
    AddSol(-55.1730, -52.1400, 0.0000, -0.1052, 0, 0, 2, -2);
    AddSol(0.2540, 0.2500, 0.0000, -0.0017, 0, 0, 2, -3);
    AddSol(0.0250, -1.6700, 0.0000, 0.0031, 0, 0, 2, -4);
    AddSol(1.0600, 2.9600, -0.1660, 0.0243, 3, 0, 0, 2);
    AddSol(36.1240, 50.6400, -1.3000, 0.6215, 3, 0, 0, 0);
    AddSol(-13.1930, -16.4000, 0.2580, -0.1187, 3, 0, 0, -2);
    AddSol(-1.1870, -0.7400, 0.0420, 0.0074, 3, 0, 0, -4);
    AddSol(-0.2930, -0.3100, -0.0020, 0.0046, 3, 0, 0, -6);
    AddSol(-0.2900, -1.4500, 0.1160, -0.0051, 2, 1, 0, 2);
    AddSol(-7.6490, -10.5600, 0.2590, -0.1038, 2, 1, 0, 0);
    AddSol(-8.6270, -7.5900, 0.0780, -0.0192, 2, 1, 0, -2);
    AddSol(-2.7400, -2.5400, 0.0220, 0.0324, 2, 1, 0, -4);
    AddSol(1.1810, 3.3200, -0.2120, 0.0213, 2, -1, 0, 2);
    AddSol(9.7030, 11.6700, -0.1510, 0.1268, 2, -1, 0, 0);
    AddSol(-0.3520, -0.3700, 0.0010, -0.0028, 2, -1, 0, -1);
    AddSol(-2.4940, -1.1700, -0.0030, -0.0017, 2, -1, 0, -2);
    AddSol(0.3600, 0.2000, -0.0120, -0.0043, 2, -1, 0, -4);
    AddSol(-1.1670, -1.2500, 0.0080, -0.0106, 1, 2, 0, 0);
    AddSol(-7.4120, -6.1200, 0.1170, 0.0484, 1, 2, 0, -2);
    AddSol(-0.3110, -0.6500, -0.0320, 0.0044, 1, 2, 0, -4);
    AddSol(0.7570, 1.8200, -0.1050, 0.0112, 1, -2, 0, 2);
    AddSol(2.5800, 2.3200, 0.0270, 0.0196, 1, -2, 0, 0);
    AddSol(2.5330, 2.4000, -0.0140, -0.0212, 1, -2, 0, -2);
    AddSol(-0.3440, -0.5700, -0.0250, 0.0036, 0, 3, 0, -2);
    AddSol(-0.9920, -0.0200, 0.0000, 0.0000, 1, 0, 2, 2);
    AddSol(-45.0990, -0.0200, 0.0000, -0.0010, 1, 0, 2, 0);
    AddSol(-0.1790, -9.5200, 0.0000, -0.0833, 1, 0, 2, -2);
    AddSol(-0.3010, -0.3300, 0.0000, 0.0014, 1, 0, 2, -4);
    AddSol(-6.3820, -3.3700, 0.0000, -0.0481, 1, 0, -2, 2);
    AddSol(39.5280, 85.1300, 0.0000, -0.7136, 1, 0, -2, 0);
    AddSol(9.3660, 0.7100, 0.0000, -0.0112, 1, 0, -2, -2);
    AddSol(0.2020, 0.0200, 0.0000, 0.0000, 1, 0, -2, -4);
    AddSol(0.4150, 0.1000, 0.0000, 0.0013, 0, 1, 2, 0);
    AddSol(-2.1520, -2.2600, 0.0000, -0.0066, 0, 1, 2, -2);
    AddSol(-1.4400, -1.3000, 0.0000, 0.0014, 0, 1, -2, 2);
    AddSol(0.3840, -0.0400, 0.0000, 0.0000, 0, 1, -2, -2);
    AddSol(1.9380, 3.6000, -0.1450, 0.0401, 4, 0, 0, 0);
    AddSol(-0.9520, -1.5800, 0.0520, -0.0130, 4, 0, 0, -2);
    AddSol(-0.5510, -0.9400, 0.0320, -0.0097, 3, 1, 0, 0);
    AddSol(-0.4820, -0.5700, 0.0050, -0.0045, 3, 1, 0, -2);
    AddSol(0.6810, 0.9600, -0.0260, 0.0115, 3, -1, 0, 0);
    AddSol(-0.2970, -0.2700, 0.0020, -0.0009, 2, 2, 0, -2);
    AddSol(0.2540, 0.2100, -0.0030, 0.0000, 2, -2, 0, -2);
    AddSol(-0.2500, -0.2200, 0.0040, 0.0014, 1, 3, 0, -2);
    AddSol(-3.9960, 0.0000, 0.0000, 0.0004, 2, 0, 2, 0);
    AddSol(0.5570, -0.7500, 0.0000, -0.0090, 2, 0, 2, -2);
    AddSol(-0.4590, -0.3800, 0.0000, -0.0053, 2, 0, -2, 2);
    AddSol(-1.2980, 0.7400, 0.0000, 0.0004, 2, 0, -2, 0);
    AddSol(0.5380, 1.1400, 0.0000, -0.0141, 2, 0, -2, -2);
    AddSol(0.2630, 0.0200, 0.0000, 0.0000, 1, 1, 2, 0);
    AddSol(0.4260, 0.0700, 0.0000, -0.0006, 1, 1, -2, -2);
    AddSol(-0.3040, 0.0300, 0.0000, 0.0003, 1, -1, 2, 0);
    AddSol(-0.3720, -0.1900, 0.0000, -0.0027, 1, -1, -2, 2);
    AddSol(0.4180, 0.0000, 0.0000, 0.0000, 0, 0, 4, 0);
    AddSol(-0.3300, -0.0400, 0.0000, 0.0000, 3, 0, 2, 0);
    function ADDN(coeffn, p, q, r, s) {
        return coeffn * Term(p, q, r, s).y;
    }
    N = 0;
    N += ADDN(-526.069, 0, 0, 1, -2);
    N += ADDN(-3.352, 0, 0, 1, -4);
    N += ADDN(+44.297, +1, 0, 1, -2);
    N += ADDN(-6.000, +1, 0, 1, -4);
    N += ADDN(+20.599, -1, 0, 1, 0);
    N += ADDN(-30.598, -1, 0, 1, -2);
    N += ADDN(-24.649, -2, 0, 1, 0);
    N += ADDN(-2.000, -2, 0, 1, -2);
    N += ADDN(-22.571, 0, +1, 1, -2);
    N += ADDN(+10.985, 0, -1, 1, -2);
    DLAM += (+0.82 * Sine(0.7736 - 62.5512 * T) + 0.31 * Sine(0.0466 - 125.1025 * T)
        + 0.35 * Sine(0.5785 - 25.1042 * T) + 0.66 * Sine(0.4591 + 1335.8075 * T)
        + 0.64 * Sine(0.3130 - 91.5680 * T) + 1.14 * Sine(0.1480 + 1331.2898 * T)
        + 0.21 * Sine(0.5918 + 1056.5859 * T) + 0.44 * Sine(0.5784 + 1322.8595 * T)
        + 0.24 * Sine(0.2275 - 5.7374 * T) + 0.28 * Sine(0.2965 + 2.6929 * T)
        + 0.33 * Sine(0.3132 + 6.3368 * T));
    S = F + DS / MOON_ARC;
    var lat_seconds = (1.000002708 + 139.978 * DGAM) * (18518.511 + 1.189 + GAM1C) * Math.sin(S) - 6.24 * Math.sin(3 * S) + N;
    return {
        geo_eclip_lon: MOON_PI2 * moFrac((L0 + DLAM / MOON_ARC) / MOON_PI2),
        geo_eclip_lat: (Math.PI / (180 * 3600)) * lat_seconds,
        distance_au: (MOON_ARC * MOON_EARTH_RADIUS_AU) / (0.999953253 * SINPI)
    };
}

/**
 * Moon's equatorial position and geocentric distance.
 *
 * @param {number} jd - Julian Date.
 * @returns {{raH: number, dec: number, distAu: number}} RA in hours,
 *     Dec in degrees, geocentric distance in AU.
 */
function moonRaDec(jd) {
  var m = calcMoon(jd);
  var eps = d2r(meanObliqDeg(jd));
  var cosL = Math.cos(m.geo_eclip_lon);
  var sinL = Math.sin(m.geo_eclip_lon);
  var x = Math.cos(m.geo_eclip_lat) * cosL;
  var y = Math.cos(m.geo_eclip_lat) * sinL;
  var z = Math.sin(m.geo_eclip_lat);
  var ye = y * Math.cos(eps) - z * Math.sin(eps);
  var ze = y * Math.sin(eps) + z * Math.cos(eps);
  var raH = r2d(Math.atan2(ye, x)) / 15.0;
  if (raH < 0) raH += 24.0;
  return { raH: raH, dec: r2d(Math.asin(ze)), distAu: m.distance_au };
}
/**
 * Angular separation between two equatorial positions.
 *
 * @param {number} raH1 - RA of position 1, hours.
 * @param {number} dec1 - Dec of position 1, degrees.
 * @param {number} raH2 - RA of position 2, hours.
 * @param {number} dec2 - Dec of position 2, degrees.
 * @returns {number} Separation in degrees.
 */
function angularSep(raH1, dec1, raH2, dec2) {
  var d1 = d2r(dec1), d2 = d2r(dec2);
  var dra = d2r(raH1 * 15.0 - raH2 * 15.0);
  var c = Math.sin(d1) * Math.sin(d2) + Math.cos(d1) * Math.cos(d2) * Math.cos(dra);
  return r2d(Math.acos(Math.max(-1, Math.min(1, c))));
}
/**
 * Moon's elongation and illuminated fraction.
 *
 * @param {number} jd - Julian Date.
 * @returns {{sepDeg: number, illum: number}} Solar-Moon separation in
 *     degrees and illuminated fraction, 0 (new) to 1 (full).
 */
function moonPhase(jd) {
  var m = moonRaDec(jd);
  var s = sunRaDec(jd);
  return {
    sepDeg: angularSep(m.raH, m.dec, s.raH, s.dec),
    illum: (1 - Math.cos(d2r(angularSep(m.raH, m.dec, s.raH, s.dec)))) / 2
  };
}


/**
 * Split a sexagesimal entry into its numeric parts.
 *
 * @param {string} text - Entry with colons, h/m/s or
 *     degree/arcminute/arcsecond marks, or space separators.
 * @returns {number[]} One number per part, in order; NaN where a part is
 *     not numeric.
 */
function _parts(text) {
  var s = String(text).trim().toLowerCase();
  var tokens = ["h", "m", "s", "deg", "d", "\u00b0", "\u2032", "\u2033"];
  var i;
  for (i = 0; i < tokens.length; i++) s = s.split(tokens[i]).join(" ");
  return s.replace(/:/g, " ").split(/\s+/).filter(function (p) { return p !== ""; }).map(Number);
}

/**
 * Parse a right ascension entry to decimal hours.
 *
 * Accepts h:m:s, h m s, decimal hours, or decimal degrees (values > 24
 * are divided by 15).
 *
 * @param {string} text - RA entry.
 * @returns {number} RA in hours, [0, 24).
 * @throws {Error} When the entry is empty or malformed.
 */
function parseRa(text) {
  var s = String(text).trim();
  if (!s) throw new Error("RA is empty");
  if (s.indexOf(":") >= 0 || /[hms]/.test(s.toLowerCase())) {
    var vals = _parts(s);
    if (vals.length < 1 || vals.length > 3 || vals.some(function (v) { return !isFinite(v); })) {
      throw new Error("unparseable RA: " + text);
    }
    while (vals.length < 3) vals.push(0);
    if (Math.abs(vals[1]) >= 60 || Math.abs(vals[2]) >= 60) {
      throw new Error("unparseable RA: " + text);
    }
    return (vals[0] + vals[1] / 60 + vals[2] / 3600) % 24;
  }
  var value = Number(s);
  if (!isFinite(value)) throw new Error("unparseable RA: " + text);
  return (value > 24 ? value / 15 : value) % 24;
}

/**
 * Parse a declination entry to decimal degrees.
 *
 * Accepts d:m:s, d m s, or decimal degrees; a negative leading degree
 * part signs the whole entry.
 *
 * @param {string} text - Dec entry.
 * @returns {number} Declination in degrees, [-90, 90].
 * @throws {Error} When the entry is empty or malformed.
 */
function parseDec(text) {
  var s = String(text).trim();
  if (!s) throw new Error("Dec is empty");
  if (s.indexOf(":") >= 0 || /[\u00b0\u2032\u2033]/.test(s) || /[dms]/.test(s.toLowerCase())) {
    var vals = _parts(s);
    if (vals.length < 1 || vals.length > 3 || vals.some(function (v) { return !isFinite(v); })) {
      throw new Error("unparseable Dec: " + text);
    }
    while (vals.length < 3) vals.push(0);
    if (Math.abs(vals[1]) >= 60 || Math.abs(vals[2]) >= 60) {
      throw new Error("unparseable Dec: " + text);
    }
    var sign = vals[0] < 0 ? -1 : 1;
    return sign * (Math.abs(vals[0]) + vals[1] / 60 + vals[2] / 3600);
  }
  var value = Number(s);
  if (!isFinite(value)) throw new Error("unparseable Dec: " + text);
  if (Math.abs(value) > 90) throw new Error("Dec out of range: " + text);
  return value;
}

// ------------------------------------------------- shareable URL fragments
// The full workspace state lives in location.hash as "#w=<urlencoded JSON>";
// encodeShareFragment / decodeShareFragment are the engine-tested pair that
// writes and reads it.
/**
 * Encode a workspace as a location.hash fragment.
 *
 * @param {string} json - Workspace JSON text.
 * @returns {string} "#w=" + URL-encoded text.
 */
function encodeShareFragment(json) { return "#w=" + encodeURIComponent(json); }

/**
 * Escape a string for safe HTML text content.
 *
 * @param {string} s - Text.
 * @returns {string} Text with &, <, >, double quote and single quote escaped.
 */
function htmlEscape(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Read a workspace from a location.hash fragment.
 *
 * @param {string} hash - location.hash value, or empty.
 * @returns {?Object} The parsed workspace object, or null when the
 *     fragment is absent or not a JSON object.
 */
function decodeShareFragment(hash) {
  var h = String(hash || "");
  if (h.indexOf("#w=") !== 0) return null;
  var text;
  try { text = decodeURIComponent(h.slice(3)); } catch (e) { return null; }
  var s;
  try { s = JSON.parse(text); } catch (e2) { return null; }
  if (!s || typeof s !== "object" || Array.isArray(s)) return null;
  return s;
}

/*JS-CORE-END*/

// --------------------------------------------------------------- embedded data
var INITIAL =
__INITIAL__;
var OBSERVATORIES =
__OBSERVATORIES__;
var PALETTE =
__PALETTE__;
var SHADE =
__SHADE__;
var THEME = {
  bg: "__FIG_BG__",
  axes_bg: "__AXES_BG__",
  fg: "__FG__",
  fg_dim: "__FG_DIM__",
  border: "__BORDER__",
  error: "__ERROR_FG__",
};
var STATE_KEY = "airmass-gui-site-state";
var FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';

/**
 * Shortcut for document.getElementById.
 *
 * @param {string} id - Element id.
 * @returns {HTMLElement}
 */
function $(id) { return document.getElementById(id); }
/**
 * Set the status line.
 *
 * @param {string} msg - Message; empty clears it.
 */
function setStatus(msg) { $("status").textContent = msg || ""; }

// ------------------------------------------------------------------- state
/**
 * Deep-copy a JSON-compatible value.
 *
 * @param {*} o - Value to copy.
 * @returns {*} Independent copy of the value.
 */
function deepCopy(o) { return JSON.parse(JSON.stringify(o)); }

/**
 * Fill the display options with their defaults.
 *
 * Keeps older saved states and imported files working.
 *
 * @param {Object} s - Workspace state; its options object is completed
 *     in place.
 * @returns {Object} The same state object.
 */
function normalizeState(s) {
  var opts = {
    threshold: 1.7,
    windows: true,
    moon: false,
    civil: false,
    nautical: false,
    dusk: true,
  };
  if (s.options) {
    for (var k in opts) { if (s.options[k] != null) opts[k] = s.options[k]; }
  }
  s.options = opts;
  return s;
}

/**
 * Check that a value has the workspace shape.
 *
 * @param {*} s - Value to check.
 * @returns {boolean} True when s has observatory, objects, and time_range.
 */
function stateLooksValid(s) {
  return !!(s && s.observatory && Array.isArray(s.objects) && s.time_range);
}

/**
 * Load the initial state: URL fragment, then saved state, then the embedded workspace.
 *
 * @returns {Object} Normalized state.
 */
function loadState() {
  var s = null;
  try {
    var fromUrl = decodeShareFragment(window.location.hash);
    if (stateLooksValid(fromUrl)) s = fromUrl;
  } catch (e) { s = null; }
  if (s === null) {
    try {
      var raw = localStorage.getItem(STATE_KEY);
      if (raw) {
        var stored = JSON.parse(raw);
        if (stateLooksValid(stored)) s = stored;
      }
    } catch (e) { /* fall through to INITIAL */ }
  }
  if (s === null) s = deepCopy(INITIAL);
  return normalizeState(s);
}

var state = loadState();

/**
 * Persist the state to browser storage and the URL fragment.
 */
function saveState() {
  try { localStorage.setItem(STATE_KEY, JSON.stringify(state)); } catch (e) { /* private mode */ }
  try {
    window.history.replaceState(null, "", encodeShareFragment(JSON.stringify(state)));
  } catch (e) { /* e.g. pages opened from file:// */ }
}

// ---------------------------------------------------------- time / timezone
var _fmtCache = {};
/**
 * Cached Intl.DateTimeFormat for 'YYYY-MM-DD HH:MM:SS' parts in a time zone.
 *
 * @param {string} tz - IANA time zone name.
 * @returns {Intl.DateTimeFormat}
 */
function tzFormatter(tz) {
  if (!_fmtCache[tz]) {
    _fmtCache[tz] = new Intl.DateTimeFormat("en-CA", {
      timeZone: tz, hourCycle: "h23",
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  }
  return _fmtCache[tz];
}
/**
 * Cached Intl.DateTimeFormat for short time-axis labels in a time zone.
 *
 * @param {string} tz - IANA time zone name.
 * @returns {Intl.DateTimeFormat}
 */
function fmtLabel(tz) {
  if (!_fmtCache["label:" + tz]) {
    _fmtCache["label:" + tz] = new Intl.DateTimeFormat("en-US", {
      timeZone: tz, hourCycle: "h23",
      month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit",
    });
  }
  return _fmtCache["label:" + tz];
}
/**
 * Local date parts of an instant in a time zone.
 *
 * @param {string} tz - IANA time zone name.
 * @param {number} ms - Epoch ms.
 * @returns {Object} year, month, day, hour, minute, second.
 */
function tzParts(tz, ms) {
  var out = {};
  var parts = tzFormatter(tz).formatToParts(new Date(ms));
  for (var i = 0; i < parts.length; i++) out[parts[i].type] = Number(parts[i].value);
  return out;
}
/**
 * Validate a time zone name, falling back to UTC.
 *
 * @param {string} tz - IANA time zone name, or empty.
 * @returns {string} A usable time zone name.
 */
function resolveTz(tz) {
  if (!tz) return "UTC";
  try { new Intl.DateTimeFormat("en-CA", { timeZone: tz }); return tz; }
  catch (e) { return "UTC"; }
}

/**
 * Convert a naive wall time in a time zone to UTC epoch ms.
 *
 * Iterates up to four times to fix the UTC offset (DST transitions).
 *
 * @param {string} tz - IANA time zone name.
 * @param {number} y - Year.
 * @param {number} mo - Month, 1-12.
 * @param {number} d - Day of month.
 * @param {number} h - Hour, 0-23.
 * @param {number} mi - Minute, 0-59.
 * @param {number} s - Second, 0-59.
 * @returns {number} Epoch ms.
 */
function wallToUtc(tz, y, mo, d, h, mi, s) {
  var want = Date.UTC(y, mo - 1, d, h, mi, s);
  var guess = want;
  for (var i = 0; i < 4; i++) {
    var p = tzParts(tz, guess);
    var got = Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second);
    if (got === want) return guess;
    guess += want - got;
  }
  return guess;
}

/**
 * UTC epoch ms of the workspace start stamp (local wall time).
 *
 * @returns {?number} null when the start string is not 'YYYY-MM-DD HH:MM'.
 */
function startUtcMs() {
  var m = /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})/.exec(String(state.time_range.start || ""));
  if (!m) return null;
  return wallToUtc(resolveTz(state.observatory.timezone),
                   +m[1], +m[2], +m[3], +m[4], +m[5], 0);
}
/**
 * UTC epoch ms of start + duration.
 *
 * @returns {?number} null when startUtcMs is null.
 */
function endUtcMs() {
  var s = startUtcMs();
  return s == null ? null : s + state.time_range.days * 86400000;
}
/**
 * Zero-pad an integer to two digits.
 *
 * @param {number} n - Integer.
 * @returns {string}
 */
function pad2(n) { return (n < 10 ? "0" : "") + n; }
/**
 * Format an instant as 'YYYY-MM-DD HH:MM' in the observatory's time zone.
 *
 * @param {number} ms - Epoch ms.
 * @returns {string}
 */
function localStamp(ms) {
  var p = tzParts(resolveTz(state.observatory.timezone), ms);
  return p.year + "-" + pad2(p.month) + "-" + pad2(p.day) +
         " " + pad2(p.hour) + ":" + pad2(p.minute);
}
/**
 * First local midnight at or after an instant.
 *
 * @param {string} tz - IANA time zone name.
 * @param {number} afterMs - Epoch ms.
 * @returns {number} Epoch ms of the next local midnight.
 */
function nextLocalMidnight(tz, afterMs) {
  var ms = afterMs;
  for (var i = 0; i < 5; i++) {
    var p = tzParts(tz, ms);
    var mid = wallToUtc(tz, p.year, p.month, p.day, 0, 0, 0);
    if (mid >= afterMs) return mid;
    ms = mid + 86400000;
  }
  return afterMs;
}
/**
 * Time-axis tick instants for a view.
 *
 * Chooses a step (1/12 h up to 96 h) keeping the tick count at 12 or
 * fewer, anchored to local midnight.
 *
 * @param {number} startMs - View start, epoch ms.
 * @param {number} endMs - View end, epoch ms.
 * @param {string} tz - IANA time zone name.
 * @returns {number[]} Tick instants in epoch ms.
 */
function timeTicks(startMs, endMs, tz) {
  var hours = (endMs - startMs) / 3600000;
  var candidates = [1 / 12, 1 / 6, 1 / 4, 0.25, 0.5, 1, 2, 3, 4, 6, 12, 24, 48, 72, 96];
  var stepH = 96;
  for (var i = 0; i < candidates.length; i++) {
    if (hours / candidates[i] <= 12) { stepH = candidates[i]; break; }
  }
  var stepMs = Math.round(stepH * 3600000);
  var base = nextLocalMidnight(tz, startMs - stepMs);
  if (stepH < 24) base -= stepMs; // anchor the grid to local midnight
  var first = base + Math.ceil((startMs - base) / stepMs) * stepMs;
  var ticks = [];
  for (var t = first; t <= endMs; t += stepMs) ticks.push(t);
  return ticks;
}

// ------------------------------------------------------------------ controls
/**
 * Push the state into every control on the page.
 */
function syncAllControls() {
  var obs = state.observatory;
  $("obs-title").textContent = obs.name || "Custom observatory";
  var isPreset = false;
  for (var i = 0; i < OBSERVATORIES.length; i++) {
    if (OBSERVATORIES[i].name === obs.preset) isPreset = true;
  }
  $("preset").value = isPreset ? obs.preset : "Custom";
  $("lat").value = obs.lat;
  $("lon").value = obs.lon;
  $("elev").value = obs.elevation_m;
  $("tz").value = obs.timezone || "";
  $("start").value = String(state.time_range.start || "").replace(" ", "T");
  $("days").value = state.time_range.days;
  $("windows-chk").checked = !!state.options.windows;
  $("threshold").value = state.options.threshold;
  $("moon-chk").checked = !!state.options.moon;
  $("civil-chk").checked = !!state.options.civil;
  $("nautical-chk").checked = !!state.options.nautical;
  $("dusk-chk").checked = !!state.options.dusk;
  $("summary-panel").style.display = state.options.windows ? "" : "none";
  renderTargets();
}

/**
 * Fill the preset combobox with the observatory list plus Custom.
 */
function buildPresetSelect() {
  var sel = $("preset");
  sel.innerHTML = "";
  OBSERVATORIES.forEach(function (p) {
    var opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.name;
    sel.appendChild(opt);
  });
  var custom = document.createElement("option");
  custom.value = "Custom";
  custom.textContent = "Custom";
  sel.appendChild(custom);
}

/**
 * Write the observatory form fields into the state, dropping the preset key when a field drifts from the preset.
 */
function commitObsFields() {
  var obs = state.observatory;
  var lat = parseFloat($("lat").value);
  var lon = parseFloat($("lon").value);
  var elev = parseFloat($("elev").value);
  var tz = $("tz").value.trim();
  if (isFinite(lat)) obs.lat = lat;
  if (isFinite(lon)) obs.lon = lon;
  if (isFinite(elev)) obs.elevation_m = elev;
  if (tz) obs.timezone = tz;
  if (obs.preset) {
    var p = null;
    for (var i = 0; i < OBSERVATORIES.length; i++) {
      if (OBSERVATORIES[i].name === obs.preset) p = OBSERVATORIES[i];
    }
    var changed = p === null ||
      Math.abs(obs.lat - p.lat) > 1e-4 ||
      Math.abs(obs.lon - p.lon) > 1e-4 ||
      Math.abs(obs.elevation_m - p.elevation_m) > 0.5 ||
      obs.timezone !== p.timezone;
    if (changed) {
      obs.preset = null;
      $("preset").value = "Custom";
    }
  }
  obs.name = obs.preset || "Custom observatory";
}

/**
 * Observe an observatory field edit: commit, save, redraw.
 */
function onObsField() {
  commitObsFields();
  saveState();
  scheduleRedraw();
}

/**
 * Next unused palette color for a new target.
 *
 * @returns {string} Hex color.
 */
function nextColor() {
  var used = state.objects.map(function (o) { return o.color; });
  for (var i = 0; i < PALETTE.length; i++) {
    if (used.indexOf(PALETTE[i]) < 0) return PALETTE[i];
  }
  return PALETTE[state.objects.length % PALETTE.length];
}

/**
 * Rebuild the per-target edit rows.
 */
function renderTargets() {
  var box = $("targets");
  box.innerHTML = "";
  state.objects.forEach(function (obj, i) {
    var row = document.createElement("div");
    row.className = "target-row";
    row.innerHTML =
      '<input type="color" title="Color">' +
      '<input placeholder="Name" title="Name">' +
      '<input placeholder="RA (h:m:s / deg)" title="RA">' +
      '<input placeholder="Dec (d:m:s / deg)" title="Dec">' +
      '<button class="rm" title="Remove" type="button">&#215;</button>';
    var kids = row.children;
    kids[0].value = /^#[0-9a-fA-F]{6}$/.test(obj.color || "") ? obj.color : PALETTE[i % PALETTE.length];
    kids[1].value = obj.name || "";
    kids[2].value = obj.ra || "";
    kids[3].value = obj.dec || "";
    kids[0].addEventListener("input", function () { obj.color = this.value; afterTargetChange(); });
    kids[1].addEventListener("input", function () { obj.name = this.value; afterTargetChange(); });
    kids[2].addEventListener("input", function () { obj.ra = this.value; afterTargetChange(); });
    kids[3].addEventListener("input", function () { obj.dec = this.value; afterTargetChange(); });
    kids[4].addEventListener("click", function () {
      state.objects.splice(i, 1);
      renderTargets();
      afterTargetChange();
    });
    box.appendChild(row);
  });
}

/**
 * Save and redraw after a target row edit.
 */
function afterTargetChange() {
  saveState();
  scheduleRedraw();
}

var WS_KEY = "airmass-gui:workspaces";

/**
 * Read named workspaces from browser storage.
 *
 * @returns {Object} Name -> state map, {} when absent or corrupt.
 */
function readWorkspaces() {
  try {
    var raw = localStorage.getItem(WS_KEY);
    var ws = raw ? JSON.parse(raw) : {};
    if (ws && typeof ws === "object" && !Array.isArray(ws)) return ws;
  } catch (e) { /* ignore */ }
  return {};
}

/**
 * Write named workspaces to browser storage.
 *
 * @param {Object} ws - Name -> state map.
 */
function writeWorkspaces(ws) {
  try { localStorage.setItem(WS_KEY, JSON.stringify(ws)); } catch (e) { /* private mode */ }
}

/**
 * Fill the workspace select box with the stored names.
 */
function refreshWsSelect() {
  var sel = $("ws-select");
  var names = Object.keys(readWorkspaces()).sort();
  sel.innerHTML = "";
  if (names.length === 0) {
    var opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(no workspaces saved in this browser)";
    sel.appendChild(opt);
    return;
  }
  names.forEach(function (name) {
    var opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  });
}

/**
 * Switch the page to a workspace document.
 *
 * @param {Object} s - Workspace state to load.
 * @param {string} sourceLabel - Label for the status line (e.g. a file name).
 * @returns {boolean} False, with a status message, when s is not a valid workspace.
 */
function loadWorkspaceDoc(s, sourceLabel) {
  if (!stateLooksValid(s)) {
    setStatus("Not a valid airmass-gui workspace");
    return false;
  }
  state = normalizeState(deepCopy(s));
  view = null;
  hideTooltip();
  syncAllControls();
  saveState();
  draw();
  setStatus("Loaded " + sourceLabel);
  return true;
}

/**
 * Download the state as a JSON workspace file.
 */
function exportWorkspace() {
  var doc = {
    version: 1,
    created: INITIAL.created || "",
    modified: new Date().toISOString().slice(0, 19).replace("T", " "),
    observatory: state.observatory,
    objects: state.objects,
    time_range: state.time_range,
  };
  var blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "airmass-workspace.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
}

// --------------------------------------------------------------------- chart
// Chart canvas and its 2D context.
var canvas = $("chart");
var ctx = canvas.getContext("2d");
var lastPlot = null;  // plot rectangle from the most recent draw()
var view = null;      // current time window {t0, t1}; null = full range
var dragState = null; // in-progress edge/pan drag

/**
 * Size the canvas to its container width and the device pixel ratio.
 *
 * @returns {{w: number, h: number}} CSS pixel size of the canvas.
 */
function sizeCanvas() {
  var dpr = window.devicePixelRatio || 1;
  var w = Math.max(320, canvas.clientWidth);
  var h = Math.round(Math.min(620, Math.max(340, w * 0.495)));
  canvas.style.height = h + "px";
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w: w, h: h };
}

/**
 * Format an instant as 'HH:MM' in a time zone.
 *
 * @param {string} tz - IANA time zone name.
 * @param {number} ms - Epoch ms.
 * @returns {string}
 */
function hmStr(tz, ms) {
  var p = tzParts(tz, ms);
  return pad2(p.hour) + ":" + pad2(p.minute);
}

/**
 * Clear the observing-window summary panel.
 */
function clearSummary() {
  $("summary").innerHTML = "";
  $("summary-panel").style.display = "none";
}

/**
 * List the per-target, per-local-night observing windows in the summary panel.
 *
 * A window is a run with X <= threshold; each run is assigned to the local
 * day where it starts (a run crossing midnight is listed in full on its
 * start day). Transit times and their airmass are listed per day.
 */
function renderSummary() {
  var box = $("summary");
  box.innerHTML = "";
  if (!state.options.windows) {
    $("summary-panel").style.display = "none";
    return;
  }
  var obs = state.observatory;
  var tz = resolveTz(obs.timezone);
  var startMs = startUtcMs(), endMs = endUtcMs();
  if (startMs == null || endMs == null) {
    $("summary-panel").style.display = "none";
    return;
  }
  var thrAlt = altFromAirmass(state.options.threshold);
  var step = Math.max(60000, (endMs - startMs) / 800);
  var n = Math.ceil((endMs - startMs) / step) + 1;
  var days = [];
  var dayStart = startMs;
  while (dayStart < endMs) {
    var dayEnd = Math.min(nextLocalMidnight(tz, dayStart + 1), endMs);
    if (dayEnd - dayStart < 3600000) dayEnd = endMs; // degenerate tz data: ensure progress
    days.push([dayStart, dayEnd]);
    dayStart = dayEnd;
  }
  var html = "", anyTarget = false;
  var totals = [], okArrays = [];
  state.objects.forEach(function (obj, oi) {
    var name = obj.name || "Target " + (oi + 1);
    var raH, dec;
    try {
      raH = parseRa(obj.ra);
      dec = parseDec(obj.dec);
    } catch (e) { return; }
    anyTarget = true;
    var altAt = [];
    for (var k = 0; k < n; k++) {
      var ms = startMs + k * step;
      if (ms > endMs) break;
      var jd = jdFromUtcMs(ms);
      var tt = precessIcrsToTete(raH * 15, dec, jd);
      altAt.push(altitude(tt.raDeg / 15, tt.dec, jd, obs.lat, obs.lon));
    }
    var ok = altAt.map(function (a) { return a >= thrAlt; });
    var runs = trueRuns(ok, startMs, step);
    okArrays.push(ok);
    var tot = 0;
    runs.forEach(function (run) { tot += (run[1] - run[0]) / 3600000; });
    totals.push({ name: name, h: tot });
    html += '<div class="sum-target"><b>' + htmlEscape(name) + "</b>";
    var shownOnDay = {};
    runs.forEach(function (run) {
      var dayIdx = 0;
      for (var di = 0; di < days.length; di++) {
        if (run[0] >= days[di][0]) { dayIdx = di; break; }
        dayIdx = di;
      }
      if (shownOnDay[dayIdx] === undefined) shownOnDay[dayIdx] = [];
      shownOnDay[dayIdx].push(run);
    });
    days.forEach(function (day, di) {
      var p = tzParts(tz, day[0] + 1000);
      var label = p.year + "-" + pad2(p.month) + "-" + pad2(p.day);
      var dayRuns = shownOnDay[di] || [];
      var dayEndIdx = Math.min(n - 1, Math.ceil((day[1] - startMs) / step));
      var everUp = false, maxAlt = -90;
      for (var ki = Math.max(0, Math.floor((day[0] - startMs) / step));
           ki <= dayEndIdx; ki++) {
        if (altAt[ki] > 0) everUp = true;
        if (altAt[ki] > maxAlt) maxAlt = altAt[ki];
      }
      var parts = [];
      if (!everUp) {
        parts.push("below horizon");
      } else if (dayRuns.length === 0) {
        parts.push("no time with X \u2264 " + state.options.threshold.toFixed(2) +
                   " (max alt " + maxAlt.toFixed(1) + "\u00b0)");
      } else {
        dayRuns.forEach(function (run) {
          parts.push(hmStr(tz, run[0]) + "\u2013" + hmStr(tz, run[1]) +
                     " \u00b7 " + ((run[1] - run[0]) / 3600000).toFixed(1) + " h");
        });
        var trs = transitTimes(raH, day[0], day[1], obs.lon);
        trs.forEach(function (tm) {
          var jd = jdFromUtcMs(tm);
          var tt = precessIcrsToTete(raH * 15, dec, jd);
          var altT = altitude(tt.raDeg / 15, tt.dec, jd, obs.lat, obs.lon);
          parts.push("transit " + hmStr(tz, tm) +
                     " \u00b7 X " + (isFinite(airmass(altT)) ? airmass(altT).toFixed(2) : "\u2014"));
        });
      }
      html += '<span class="sum-day">' + label + ":  " + parts.join(";  ") + "</span>";
    });
    html += "</div>";
  });
  if (anyTarget) {
    var anyOk = okArrays[0].slice();
    okArrays.forEach(function (arr) {
      for (var ai = 0; ai < arr.length; ai++) { if (arr[ai]) anyOk[ai] = true; }
    });
    var anyRuns = trueRuns(anyOk, startMs, step);
    var anyH = 0;
    anyRuns.forEach(function (run) { anyH += (run[1] - run[0]) / 3600000; });
    var head = [];
    totals.forEach(function (t) { head.push(htmlEscape(t.name) + " " + t.h.toFixed(1) + " h"); });
    head.push("any target " + anyH.toFixed(1) + " h");
    html = '<div class="sum-total"><b>Cumulative X \u2264 ' +
           state.options.threshold.toFixed(2) + ':</b>  ' + head.join(";  ") + "</div>";
  }
  if (!anyTarget) html = '<div class="sum-empty">Add a target to list its observing windows.</div>';
  box.innerHTML = html;
  $("summary-panel").style.display = "";
}

/**
 * Render the chart: shading, window bands, target curves, moon, axes, and legend.
 *
 * Also writes the observing-window summary.
 */
function draw() {
  var size = sizeCanvas();
  var w = size.w, h = size.h;
  var obs = state.observatory;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = THEME.bg;
  ctx.fillRect(0, 0, w, h);

  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = THEME.fg;
  ctx.font = "500 13px " + FONT;
  ctx.fillText("Airmass chart \u2014 " + (obs.name || "Custom observatory"), w / 2, 20);

  var tz = resolveTz(obs.timezone);
  var badTz = tz !== (obs.timezone || "").trim();
  var startMs = startUtcMs();
  var endMs = endUtcMs();
  var msg = null;
  if (badTz) msg = "Time zone '" + obs.timezone + "' not recognized \u2014 using UTC.";
  if (startMs == null) msg = "Start time is not parseable (expected YYYY-MM-DD HH:MM).";
  else if (!isFinite(state.time_range.days) || state.time_range.days <= 0) {
    msg = "Duration must be a positive number of days.";
  }

  ctx.font = "11px " + FONT;
  ctx.fillStyle = THEME.fg_dim;
  if (startMs != null && msg === null) {
    var vrSub = viewRange();
    var zoomed = vrSub.t0 !== startMs || vrSub.t1 !== endMs;
    ctx.fillText(localStamp(vrSub.t0) + " \u2192 " + localStamp(vrSub.t1) + " local" +
                 (zoomed ? "  \u00b7  zoomed (double-click to reset)" : ""), w / 2, 36);
  } else {
    ctx.fillText("Fix the time range to draw the chart", w / 2, 36);
  }
  setStatus(msg);
  if (msg !== null) {
    clearSummary();
    ctx.fillStyle = THEME.error;
    ctx.font = "13px " + FONT;
    ctx.fillText(msg, w / 2, h / 2);
    return;
  }

  var M = { l: 56, r: 52, t: 52, b: 58 };
  var plot = { x: M.l, y: M.t, w: w - M.l - M.r, h: h - M.t - M.b };
  lastPlot = plot;
  var vr = viewRange();
  var v0 = vr.t0, v1 = vr.t1;
  function X(ms) { return plot.x + ((ms - v0) / (v1 - v0)) * plot.w; }
  function Y(alt) { return plot.y + (1 - alt / 90) * plot.h; }

  // axis background + twilight shading
  ctx.fillStyle = THEME.axes_bg;
  ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
  var shading = sunShading(v0, v1, obs.lat, obs.lon);
  for (var si = 0; si < shading.length; si++) {
    var iv = shading[si];
    ctx.fillStyle = SHADE[iv[2]];
    ctx.fillRect(X(iv[0]), plot.y, X(iv[1]) - X(iv[0]), plot.h);
  }

  // twilight marker lines: thin verticals at the sun's -6 / -12 deg crossings
  if (state.options.civil || state.options.nautical) {
    var twSpecs = [];
    if (state.options.civil) twSpecs.push({ alt: -6, color: "#ffd966" });
    if (state.options.nautical) twSpecs.push({ alt: -12, color: "#89b4d6" });
    for (var twi = 0; twi < twSpecs.length; twi++) {
      var twEvs = twilightEvents(v0, v1, obs.lat, obs.lon, twSpecs[twi].alt);
      ctx.strokeStyle = twSpecs[twi].color;
      ctx.lineWidth = 1;
      for (var evi = 0; evi < twEvs.length; evi++) {
        var ex = X(twEvs[evi]);
        ctx.beginPath(); ctx.moveTo(ex, plot.y); ctx.lineTo(ex, plot.y + plot.h); ctx.stroke();
      }
    }
  }

  var ticks = timeTicks(v0, v1, tz);

  // grid
  ctx.strokeStyle = "rgba(69, 71, 90, 0.6)";
  ctx.lineWidth = 1;
  var airTicks = [1, 1.5, 2, 4, 8, 20];
  for (var gi = 0; gi < airTicks.length; gi++) {
    var gy = Y(altFromAirmass(airTicks[gi]));
    ctx.beginPath(); ctx.moveTo(plot.x, gy); ctx.lineTo(plot.x + plot.w, gy); ctx.stroke();
  }
  for (var gi2 = 0; gi2 < ticks.length; gi2++) {
    var gx = X(ticks[gi2]);
    ctx.beginPath(); ctx.moveTo(gx, plot.y); ctx.lineTo(gx, plot.y + plot.h); ctx.stroke();
  }

  // X = 2 reference line
  var y2 = Y(altFromAirmass(2));
  ctx.save();
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = THEME.fg_dim;
  ctx.beginPath(); ctx.moveTo(plot.x, y2); ctx.lineTo(plot.x + plot.w, y2); ctx.stroke();
  ctx.restore();
  ctx.fillStyle = THEME.fg_dim;
  ctx.font = "11px " + FONT;
  ctx.textAlign = "right";
  ctx.fillText("X = 2", plot.x + plot.w - 4, y2 + 13);

  // parse all targets up front (shared by window bands, curves, and summary)
  var errors = [], below = [], legend = [];
  var targets = [];
  state.objects.forEach(function (obj, oi) {
    var name = obj.name || "Target " + (oi + 1);
    var color = obj.color || PALETTE[0];
    try {
      targets.push({ name: name, color: color,
                     raH: parseRa(obj.ra), dec: parseDec(obj.dec) });
    } catch (e) {
      errors.push(name + ": " + e.message);
    }
  });

  // airmass threshold line + per-target usable-window bands (behind curves)
  if (state.options.windows) {
    var thrAlt = altFromAirmass(state.options.threshold);
    var yt = Y(thrAlt);
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = THEME.fg_dim;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(plot.x, yt); ctx.lineTo(plot.x + plot.w, yt); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = THEME.fg_dim;
    ctx.font = "11px " + FONT;
    ctx.textAlign = "right";
    ctx.fillText("X = " + state.options.threshold.toFixed(2), plot.x + plot.w - 4, yt - 5);

    var bstep = Math.max(60000, (v1 - v0) / 800);
    var bn = Math.ceil((v1 - v0) / bstep) + 1;
    ctx.save();
    ctx.globalAlpha = 0.12;
    targets.forEach(function (tg) {
      var ok = [];
      for (var k = 0; k < bn; k++) {
        var ms = v0 + k * bstep;
        if (ms > v1) break;
        var jd = jdFromUtcMs(ms);
        var tt = precessIcrsToTete(tg.raH * 15, tg.dec, jd);
        ok.push(altitude(tt.raDeg / 15, tt.dec, jd, obs.lat, obs.lon) >= thrAlt);
      }
      var runs = trueRuns(ok, v0, bstep);
      ctx.fillStyle = tg.color;
      for (var ri = 0; ri < runs.length; ri++) {
        var xa = X(Math.max(runs[ri][0], v0));
        var xb = X(Math.min(runs[ri][1], v1));
        if (xb > xa) ctx.fillRect(xa, plot.y, xb - xa, plot.h);
      }
    });
    ctx.restore();
    // transit ticks on the time axis, colored per target
    targets.forEach(function (tg) {
      var trs = transitTimes(tg.raH, v0, v1, obs.lon);
      ctx.fillStyle = tg.color;
      for (var tti = 0; tti < trs.length; tti++) {
        ctx.fillRect(X(trs[tti]) - 1.5, plot.y + plot.h - 8, 3, 8);
      }
    });
  }

  // target curves
  var step = Math.min(5 * 60 * 1000, Math.max(1000, (v1 - v0) / 400));
  var n = Math.ceil((v1 - v0) / step) + 1;
  targets.forEach(function (tg) {
    var name = tg.name, color = tg.color, raH = tg.raH, dec = tg.dec;
    var pts = [];
    for (var k = 0; k < n; k++) {
      var ms = v0 + k * step;
      if (ms > v1) break;
      var jd = jdFromUtcMs(ms);
      var tt = precessIcrsToTete(raH * 15, dec, jd);
      pts.push([ms, airmass(altitude(tt.raDeg / 15, tt.dec, jd, obs.lat, obs.lon))]);
    }
    var finite = pts.filter(function (p) { return isFinite(p[1]); });
    if (finite.length === 0) return;
    var cstep = Math.max(15 * 60 * 1000, (endMs - startMs) / 400);
    var rises = false;
    for (var ck = 0; ck <= Math.ceil((endMs - startMs) / cstep); ck++) {
      var cjd = jdFromUtcMs(startMs + ck * cstep);
      var ctt = precessIcrsToTete(raH * 15, dec, cjd);
      if (altitude(ctt.raDeg / 15, ctt.dec, cjd, obs.lat, obs.lon) > 0) { rises = true; break; }
    }
    if (!rises) below.push(name);
    var minP = finite[0];
    for (var fi = 1; fi < finite.length; fi++) {
      if (finite[fi][1] < minP[1]) minP = finite[fi];
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    var open = false;
    for (var pi = 0; pi < pts.length; pi++) {
      if (!isFinite(pts[pi][1])) { open = false; continue; }
      var px = X(pts[pi][0]), py = Y(altFromAirmass(pts[pi][1]));
      if (!open) { ctx.moveTo(px, py); open = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = "11px " + FONT;
    ctx.textAlign = "left";
    ctx.fillText("min " + minP[1].toFixed(2), X(minP[0]) + 5, Y(altFromAirmass(minP[1])) + 14);
    legend.push({ name: name, color: color });
  });

  // Moon curve (dashed, desaturated) + legend entry
  if (state.options.moon) {
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#9aa5c4";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    var moonOpen = false;
    for (var mk = 0; mk < n; mk++) {
      var ms = v0 + mk * step;
      if (ms > v1) break;
      var jd = jdFromUtcMs(ms);
      var mr = moonRaDec(jd);
      var mAlt = altitude(mr.raH, mr.dec, jd, obs.lat, obs.lon);
      var mX = airmass(mAlt);
      if (!isFinite(mX)) { moonOpen = false; continue; }
      var mpx = X(ms), mpy = Y(altFromAirmass(mX));
      if (!moonOpen) { ctx.moveTo(mpx, mpy); moonOpen = true; } else ctx.lineTo(mpx, mpy);
    }
    ctx.stroke();
    ctx.restore();
    legend.push({ name: "Moon", color: "#9aa5c4" });
  }

  // plot border
  ctx.strokeStyle = THEME.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);

  // tick marks + labels
  ctx.font = "11px " + FONT;
  ctx.strokeStyle = THEME.fg;
  ctx.fillStyle = THEME.fg;
  ctx.textAlign = "right";
  for (var li = 0; li < airTicks.length; li++) {
    var ly = Y(altFromAirmass(airTicks[li]));
    ctx.fillText(String(airTicks[li]), plot.x - 6, ly + 3.5);
    ctx.beginPath(); ctx.moveTo(plot.x - 4, ly); ctx.lineTo(plot.x, ly); ctx.stroke();
  }
  ctx.textAlign = "left";
  var altTicks = [0, 15, 30, 45, 60, 75, 90];
  for (var ai = 0; ai < altTicks.length; ai++) {
    var ay = Y(altTicks[ai]);
    ctx.fillText(altTicks[ai] + "\u00b0", plot.x + plot.w + 6, ay + 3.5);
    ctx.beginPath(); ctx.moveTo(plot.x + plot.w, ay); ctx.lineTo(plot.x + plot.w + 4, ay); ctx.stroke();
  }
  var fmt = fmtLabel(tz);
  var lastRight = -Infinity;
  ctx.textAlign = "center";
  for (var ti = 0; ti < ticks.length; ti++) {
    var tx = X(ticks[ti]);
    ctx.beginPath(); ctx.moveTo(tx, plot.y + plot.h); ctx.lineTo(tx, plot.y + plot.h + 4); ctx.stroke();
    var label = fmt.format(new Date(ticks[ti]));
    var tw = ctx.measureText(label).width;
    if (tx - tw / 2 >= lastRight + 4) {
      ctx.fillText(label, tx, plot.y + plot.h + 17);
      lastRight = tx + tw / 2;
    }
  }

  // Dawn/dusk ticks: S/C/N/A letters with rising/setting arrows below the axis
  if (state.options.dusk) {
    var duskSpecs = [
      { alt: 0, letter: "S", color: THEME.fg },
      { alt: -6, letter: "C", color: "#ffd966" },
      { alt: -12, letter: "N", color: "#89b4d6" },
      { alt: -18, letter: "A", color: "#55779f" }
    ];
    var duskEvs = sunEvents(v0, v1, obs.lat, obs.lon);
    var lastLx = -Infinity;
    ctx.font = "10px " + FONT;
    for (var de = 0; de < duskEvs.length; de++) {
      var ev = duskEvs[de];
      var spec = null;
      for (var ds = 0; ds < duskSpecs.length; ds++) {
        if (duskSpecs[ds].alt === ev.alt) spec = duskSpecs[ds];
      }
      var dx = X(ev.ms);
      if (dx - lastLx < 13) continue; // keep the letters readable when events crowd
      lastLx = dx;
      ctx.fillStyle = spec.color;
      ctx.textAlign = "center";
      ctx.fillText(spec.letter, dx, plot.y + plot.h + 31);
      var axx = dx + 6, ayy = plot.y + plot.h + 24;
      ctx.strokeStyle = spec.color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(axx, ev.dir > 0 ? ayy + 7 : ayy);
      ctx.lineTo(axx, ev.dir > 0 ? ayy : ayy + 7);
      ctx.stroke();
      ctx.beginPath();
      if (ev.dir > 0) {
        ctx.moveTo(axx - 2.5, ayy + 3); ctx.lineTo(axx, ayy); ctx.lineTo(axx + 2.5, ayy + 3);
      } else {
        ctx.moveTo(axx - 2.5, ayy + 4); ctx.lineTo(axx, ayy + 7); ctx.lineTo(axx + 2.5, ayy + 4);
      }
      ctx.stroke();
    }
  }

  // axis labels
  ctx.textAlign = "center";
  ctx.fillText("Local time (" + (obs.timezone || "UTC") + ")", plot.x + plot.w / 2, h - 8);
  ctx.save();
  ctx.translate(12, plot.y + plot.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Airmass X", 0, 0);
  ctx.restore();
  ctx.save();
  ctx.translate(w - 8, plot.y + plot.h / 2);
  ctx.rotate(Math.PI / 2);
  ctx.fillText("Altitude", 0, 0);
  ctx.restore();

  // legend (upper left)
  if (legend.length > 0) {
    var pad = 6, rowH = 15, sw = 10;
    var maxW = 0;
    for (var le = 0; le < legend.length; le++) {
      maxW = Math.max(maxW, ctx.measureText(legend[le].name).width);
    }
    var lw = maxW + sw + 2 * pad + 6, lh = legend.length * rowH + 2 * pad;
    var lx = plot.x + 8, ly2 = plot.y + 8;
    ctx.fillStyle = THEME.axes_bg;
    ctx.fillRect(lx, ly2, lw, lh);
    ctx.strokeStyle = THEME.border;
    ctx.strokeRect(lx, ly2, lw, lh);
    ctx.textAlign = "left";
    for (var le2 = 0; le2 < legend.length; le2++) {
      var yy = ly2 + pad + 10 + le2 * rowH;
      ctx.fillStyle = legend[le2].color;
      ctx.fillRect(lx + pad, yy - 8, sw, sw);
      ctx.fillStyle = THEME.fg;
      ctx.fillText(legend[le2].name, lx + pad + sw + 6, yy);
    }
  }

  // parse errors (lower left)
  ctx.textAlign = "left";
  ctx.fillStyle = THEME.error;
  for (var ei = 0; ei < errors.length; ei++) {
    ctx.fillText(errors[ei], plot.x + 8,
                 plot.y + plot.h - 10 - (errors.length - 1 - ei) * 14);
  }

  // below horizon for whole range (upper right)
  ctx.textAlign = "right";
  ctx.fillStyle = THEME.fg_dim;
  for (var bi = 0; bi < below.length; bi++) {
    ctx.fillText(below[bi] + ": below horizon for whole range",
                 plot.x + plot.w - 8, plot.y + 16 + bi * 14);
  }

  renderSummary();
}

// --------------------------------------------------------- view interaction
// Minimum zoomed span (ms) and pixel width of the resizable plot edges.
var MIN_WINDOW_MS = 10 * 60 * 1000;
var EDGE_PX = 8;
var tooltipEl = $("tooltip");
var xhairEl = $("xhair");

/**
 * Time span currently shown, clamped to the range.
 *
 * @returns {{t0: ?number, t1: ?number}} Span in epoch ms, or {null, null}
 *     when the range is invalid.
 */
function viewRange() {
  var s = startUtcMs(), e = endUtcMs();
  if (s == null || e == null || !(e > s)) return { t0: null, t1: null };
  if (view == null) return { t0: s, t1: e };
  var t0 = Math.max(s, view.t0), t1 = Math.min(e, view.t1);
  if (t1 - t0 < MIN_WINDOW_MS) {
    var mid = 0.5 * (t0 + t1);
    var nt0 = mid - MIN_WINDOW_MS / 2, nt1 = mid + MIN_WINDOW_MS / 2;
    if (nt0 < s) { nt0 = s; nt1 = s + MIN_WINDOW_MS; }
    else if (nt1 > e) { nt1 = e; nt0 = e - MIN_WINDOW_MS; }
    t0 = Math.max(s, nt0); t1 = Math.min(e, nt1);
  }
  return { t0: t0, t1: t1 };
}

/**
 * Reset the zoomed view to the full range.
 */
function invalidateView() { view = null; hideTooltip(); }

/**
 * Clamp a proposed time span into the range at the minimum width.
 *
 * @param {number} t0 - Proposed start, epoch ms.
 * @param {number} t1 - Proposed end, epoch ms.
 * @returns {?{t0: number, t1: number}} The clamped span, or null when it
 *     already is the full range.
 */
function clampWindow(t0, t1) {
  var s = startUtcMs(), e = endUtcMs();
  if (t0 < s) { t1 += s - t0; t0 = s; }
  if (t1 > e) { t0 -= t1 - e; t1 = e; }
  t0 = Math.max(s, t0); t1 = Math.min(e, t1);
  if (t1 - t0 < MIN_WINDOW_MS) {
    var mid = 0.5 * (t0 + t1);
    var nt0 = mid - MIN_WINDOW_MS / 2, nt1 = mid + MIN_WINDOW_MS / 2;
    if (nt0 < s) { nt0 = s; nt1 = s + MIN_WINDOW_MS; }
    else if (nt1 > e) { nt1 = e; nt0 = e - MIN_WINDOW_MS; }
    t0 = Math.max(s, nt0); t1 = Math.min(e, nt1);
  }
  if (t0 === s && t1 === e) return null;
  return { t0: t0, t1: t1 };
}

/**
 * Hide the hover tooltip and crosshair.
 */
function hideTooltip() {
  tooltipEl.style.display = "none";
  xhairEl.style.display = "none";
}

/**
 * Pointer position relative to the canvas.
 *
 * @param {MouseEvent|Touch} e - Mouse or touch event.
 * @returns {{x: number, y: number}}
 */
function canvasPos(e) {
  var rect = canvas.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

/**
 * Fill and position the hover tooltip for a pointer event.
 *
 * Lists the local time, then per target the altitude, azimuth, airmass,
 * parallactic angle, and (with the moon curve on) the Moon separation.
 *
 * @param {MouseEvent|{clientX: number, clientY: number}} e
 */
function updateTooltip(e) {
  var plot = lastPlot;
  var vr = viewRange();
  var pos = canvasPos(e);
  if (!plot || vr.t0 == null ||
      pos.x < plot.x || pos.x > plot.x + plot.w ||
      pos.y < plot.y || pos.y > plot.y + plot.h) { hideTooltip(); return; }
  var obs = state.observatory;
  var ms = vr.t0 + (pos.x - plot.x) / plot.w * (vr.t1 - vr.t0);
  var jd = jdFromUtcMs(ms);
  var moon = state.options.moon ? moonRaDec(jd) : null;
  var lines = [localStamp(ms) + " local"];
  state.objects.forEach(function (obj, oi) {
    var name = obj.name || "Target " + (oi + 1);
    var raH, dec;
    try {
      raH = parseRa(obj.ra);
      dec = parseDec(obj.dec);
    } catch (err) {
      lines.push(name + ":  (unparseable coordinates)");
      return;
    }
    var tt = precessIcrsToTete(raH * 15, dec, jd);
    var hd = horizonCoords(tt.raDeg / 15, tt.dec, jd, obs.lat, obs.lon);
    if (hd.alt <= 0) {
      lines.push(name + ":  alt " + hd.alt.toFixed(1) + "\u00b0  (below horizon)");
      return;
    }
    var q = parallacticAngle(tt.raDeg / 15, tt.dec, jd, obs.lat, obs.lon);
    var line = name + ":  alt " + hd.alt.toFixed(1) + "\u00b0  az " + hd.az.toFixed(1) +
               "\u00b0  X " + airmass(hd.alt).toFixed(2) +
               "  q " + (isFinite(q) ? q.toFixed(1) + "\u00b0" : "\u2014");
    if (moon) {
      line += "  moon " + angularSep(tt.raDeg / 15, tt.dec, moon.raH, moon.dec).toFixed(1) + "\u00b0";
    }
    lines.push(line);
  });
  if (moon) {
    var mAlt = altitude(moon.raH, moon.dec, jd, obs.lat, obs.lon);
    var mp = moonPhase(jd);
    lines.push("Moon:  alt " + mAlt.toFixed(1) + "\u00b0  " +
               Math.round(mp.illum * 100) + "% lit");
  }
  tooltipEl.textContent = lines.join("\n");
  tooltipEl.style.display = "block";
  var boxRect = canvas.parentElement.getBoundingClientRect();
  var relX = e.clientX - boxRect.left, relY = e.clientY - boxRect.top;
  var tw = tooltipEl.offsetWidth, th = tooltipEl.offsetHeight;
  var left = relX + 14;
  if (left + tw > boxRect.width - 4) left = relX - tw - 14;
  tooltipEl.style.left = Math.max(4, left) + "px";
  tooltipEl.style.top = Math.max(4, Math.min(relY + 14, boxRect.height - th - 4)) + "px";
  var canvasTop = canvas.getBoundingClientRect().top - boxRect.top;
  xhairEl.style.display = "block";
  xhairEl.style.left = relX + "px";
  xhairEl.style.top = canvasTop + plot.y + "px";
  xhairEl.style.height = plot.h + "px";
}

// Wheel: ctrl/meta zooms about the cursor; a plain wheel pans.
canvas.addEventListener("wheel", function (e) {
  var vr = viewRange();
  if (vr.t0 == null || !lastPlot) return;
  e.preventDefault();
  if (e.ctrlKey || e.metaKey) {
    var f = Math.max(0, Math.min(1, (canvasPos(e).x - lastPlot.x) / lastPlot.w));
    var anchor = vr.t0 + f * (vr.t1 - vr.t0);
    var factor = Math.exp(e.deltaY * 0.003);
    view = clampWindow(anchor - (anchor - vr.t0) * factor, anchor + (vr.t1 - anchor) * factor);
  } else {
    var dMs = (e.deltaX || e.deltaY) * (vr.t1 - vr.t0) / 1000;
    view = clampWindow(vr.t0 + dMs, vr.t1 + dMs);
  }
  hideTooltip();
  scheduleRedraw();
}, { passive: false });

// Mouse down: start an edge-resize or pan drag.
canvas.addEventListener("mousedown", function (e) {
  if (e.button !== 0) return;
  var vr = viewRange();
  if (vr.t0 == null || !lastPlot || lastPlot.w < 40) return;
  var pos = canvasPos(e);
  var inY = pos.y >= lastPlot.y && pos.y <= lastPlot.y + lastPlot.h;
  var mode;
  if (inY && pos.x < lastPlot.x + EDGE_PX) mode = "left";
  else if (inY && pos.x > lastPlot.x + lastPlot.w - EDGE_PX) mode = "right";
  else if (pos.x >= lastPlot.x && pos.x <= lastPlot.x + lastPlot.w && inY) mode = "pan";
  else return;
  dragState = { mode: mode, x0: pos.x, t0: vr.t0, t1: vr.t1 };
  canvas.style.cursor = mode === "pan" ? "grabbing" : "ew-resize";
  hideTooltip();
  e.preventDefault();
});

// Mouse move: continue the in-progress drag.
window.addEventListener("mousemove", function (e) {
  if (!dragState || !lastPlot) return;
  var scale = (dragState.t1 - dragState.t0) / lastPlot.w;
  var dx = canvasPos(e).x - dragState.x0;
  if (dragState.mode === "pan") {
    view = clampWindow(dragState.t0 + dx * scale, dragState.t1 + dx * scale);
  } else if (dragState.mode === "left") {
    view = clampWindow(
      Math.min(dragState.t0 + dx * scale, dragState.t1 - MIN_WINDOW_MS), dragState.t1);
  } else {
    view = clampWindow(
      dragState.t0, Math.max(dragState.t1 + dx * scale, dragState.t0 + MIN_WINDOW_MS));
  }
  scheduleRedraw();
});

// Mouse up: end the drag.
window.addEventListener("mouseup", function () {
  if (dragState) { dragState = null; canvas.style.cursor = ""; }
});

// Mouse move (no drag): set the cursor and update the tooltip.
canvas.addEventListener("mousemove", function (e) {
  if (dragState) return;
  var pos = canvasPos(e);
  var vr = viewRange();
  if (lastPlot && vr.t0 != null &&
      pos.x >= lastPlot.x && pos.x <= lastPlot.x + lastPlot.w &&
      pos.y >= lastPlot.y && pos.y <= lastPlot.y + lastPlot.h) {
    var atEdge = pos.x < lastPlot.x + EDGE_PX || pos.x > lastPlot.x + lastPlot.w - EDGE_PX;
    canvas.style.cursor = atEdge ? "ew-resize" : "crosshair";
    updateTooltip(e);
  } else {
    canvas.style.cursor = "";
    hideTooltip();
  }
});

canvas.addEventListener("mouseleave", hideTooltip);

// Double click: reset zoom to the full range.
canvas.addEventListener("dblclick", function () {
  if (view != null) { view = null; hideTooltip(); scheduleRedraw(); }
});

var touchState = null;
/**
 * Distance between the two touches of a gesture.
 *
 * @param {TouchEvent} e
 * @returns {number} Distance in CSS px.
 */
function touchDist(e) {
  var dx = e.touches[0].clientX - e.touches[1].clientX;
  var dy = e.touches[0].clientY - e.touches[1].clientY;
  return Math.sqrt(dx * dx + dy * dy);
}
// Touch start: begin a one-finger pan or a two-finger pinch.
canvas.addEventListener("touchstart", function (e) {
  var vr = viewRange();
  if (vr.t0 == null || !lastPlot || lastPlot.w < 40) return;
  if (e.touches.length === 2) {
    var rect = canvas.getBoundingClientRect();
    var midFrac = (0.5 * (e.touches[0].clientX + e.touches[1].clientX) - rect.left - lastPlot.x) / lastPlot.w;
    touchState = { mode: "pinch", d0: Math.max(1, touchDist(e)),
                   f0: Math.max(0, Math.min(1, midFrac)), t0: vr.t0, t1: vr.t1 };
    hideTooltip();
    e.preventDefault();
  } else if (e.touches.length === 1) {
    var pos = canvasPos(e.touches[0]);
    if (pos.x < lastPlot.x || pos.x > lastPlot.x + lastPlot.w ||
        pos.y < lastPlot.y || pos.y > lastPlot.y + lastPlot.h) return;
    touchState = { mode: "pan", x0: pos.x, t0: vr.t0, t1: vr.t1, moved: false,
                   clientX: e.touches[0].clientX, clientY: e.touches[0].clientY };
    hideTooltip();
    e.preventDefault();
  }
}, { passive: false });

// Touch move: pan or pinch-zoom the view.
canvas.addEventListener("touchmove", function (e) {
  if (!touchState || !lastPlot) return;
  if (touchState.mode === "pan" && e.touches.length === 1) {
    touchState.moved = true;
    var scale = (touchState.t1 - touchState.t0) / lastPlot.w;
    var dx = canvasPos(e.touches[0]).x - touchState.x0;
    view = clampWindow(touchState.t0 + dx * scale, touchState.t1 + dx * scale);
  } else if (touchState.mode === "pinch" && e.touches.length === 2) {
    var ratio = Math.max(0.2, Math.min(5, touchDist(e) / touchState.d0));
    var anchor = touchState.t0 + touchState.f0 * (touchState.t1 - touchState.t0);
    view = clampWindow(anchor - (anchor - touchState.t0) / ratio,
                       anchor + (touchState.t1 - anchor) / ratio);
  }
  hideTooltip();
  scheduleRedraw();
  e.preventDefault();
}, { passive: false });

// Touch end: a tap shows the tooltip; pinch-to-one-finger continues as a pan.
canvas.addEventListener("touchend", function (e) {
  if (touchState && touchState.mode === "pan" && !touchState.moved &&
      e.touches.length === 0) {
    updateTooltip({ clientX: touchState.clientX, clientY: touchState.clientY });
  }
  if (e.touches.length === 0) {
    touchState = null;
  } else if (e.touches.length === 1 && touchState && touchState.mode === "pinch") {
    var vr2 = viewRange();
    var pos2 = canvasPos(e.touches[0]);
    touchState = { mode: "pan", x0: pos2.x, t0: vr2.t0, t1: vr2.t1, moved: true };
  }
  e.preventDefault();
}, { passive: false });

// Keys: arrows pan, +/- zoom about the center, 0/r reset.
window.addEventListener("keydown", function (e) {
  var tag = e.target && e.target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  var vr = viewRange();
  var span = vr.t0 != null ? vr.t1 - vr.t0 : 0;
  var handled = true;
  if (e.key === "ArrowLeft") {
    if (vr.t0 != null) view = clampWindow(vr.t0 - 0.1 * span, vr.t1 - 0.1 * span);
  } else if (e.key === "ArrowRight") {
    if (vr.t0 != null) view = clampWindow(vr.t0 + 0.1 * span, vr.t1 + 0.1 * span);
  } else if (e.key === "+" || e.key === "=") {
    if (vr.t0 != null) {
      var anchor = 0.5 * (vr.t0 + vr.t1);
      view = clampWindow(anchor - (anchor - vr.t0) * 0.8, anchor + (vr.t1 - anchor) * 0.8);
    }
  } else if (e.key === "-" || e.key === "_") {
    if (vr.t0 != null) {
      var anchor2 = 0.5 * (vr.t0 + vr.t1);
      view = clampWindow(anchor2 - (anchor2 - vr.t0) * 1.25, anchor2 + (vr.t1 - anchor2) * 1.25);
    }
  } else if (e.key === "0" || e.key === "r" || e.key === "R") {
    view = null;
  } else {
    handled = false;
  }
  if (handled) { e.preventDefault(); hideTooltip(); scheduleRedraw(); }
});

// ------------------------------------------------------------------ wiring
var redrawTimer = null;
/**
 * Schedule a chart redraw 150 ms from now, replacing any pending one.
 */
function scheduleRedraw() {
  if (redrawTimer) clearTimeout(redrawTimer);
  redrawTimer = setTimeout(function () { redrawTimer = null; draw(); }, 150);
}

// Preset change: swap in a preset's observatory, or clear the preset key for Custom.
$("preset").addEventListener("change", function () {
  var name = this.value;
  if (name !== "Custom") {
    for (var i = 0; i < OBSERVATORIES.length; i++) {
      if (OBSERVATORIES[i].name === name) {
        state.observatory = {
          name: name,
          lat: OBSERVATORIES[i].lat,
          lon: OBSERVATORIES[i].lon,
          elevation_m: OBSERVATORIES[i].elevation_m,
          timezone: OBSERVATORIES[i].timezone,
          preset: name,
        };
        break;
      }
    }
    syncAllControls();
  } else {
    state.observatory.preset = null;
  }
  commitObsFields();
  saveState();
  scheduleRedraw();
});
["lat", "lon", "elev", "tz"].forEach(function (id) {
  $(id).addEventListener("input", onObsField);
});
// Start stamp: store as 'YYYY-MM-DD HH:MM'; invalidate the zoomed view.
$("start").addEventListener("input", function () {
  if (this.value) state.time_range.start = this.value.replace("T", " ");
  invalidateView();
  saveState();
  scheduleRedraw();
});
$("days").addEventListener("input", function () {
  var v = parseFloat(this.value);
  if (isFinite(v) && v > 0) state.time_range.days = v;
  invalidateView();
  saveState();
  scheduleRedraw();
});
$("windows-chk").addEventListener("change", function () {
  state.options.windows = this.checked;
  saveState();
  scheduleRedraw();
});
$("moon-chk").addEventListener("change", function () {
  state.options.moon = this.checked;
  saveState();
  scheduleRedraw();
});
$("civil-chk").addEventListener("change", function () {
  state.options.civil = this.checked;
  saveState();
  scheduleRedraw();
});
$("nautical-chk").addEventListener("change", function () {
  state.options.nautical = this.checked;
  saveState();
  scheduleRedraw();
});
$("dusk-chk").addEventListener("change", function () {
  state.options.dusk = this.checked;
  saveState();
  scheduleRedraw();
});
// Threshold: enforce the 1.05 airmass minimum.
$("threshold").addEventListener("input", function () {
  var v = parseFloat(this.value);
  if (isFinite(v) && v >= 1.05) {
    state.options.threshold = v;
    saveState();
    scheduleRedraw();
  }
});
// Add target: validate RA/Dec, append the object, clear the form.
$("add-btn").addEventListener("click", function () {
  var name = $("new-name").value.trim();
  var ra = $("new-ra").value.trim();
  var dec = $("new-dec").value.trim();
  try {
    parseRa(ra);
    parseDec(dec);
  } catch (e) {
    setStatus(e.message);
    return;
  }
  state.objects.push({
    name: name || "Target " + (state.objects.length + 1),
    ra: ra,
    dec: dec,
    color: nextColor(),
  });
  $("new-name").value = "";
  $("new-ra").value = "";
  $("new-dec").value = "";
  renderTargets();
  setStatus("");
  afterTargetChange();
});

/**
 * SIMBAD mirrors tried in order, as in the desktop app (see
 * airmass_gui.lookup for the same list).
 */
var SIMBAD_ENDPOINTS = [
  "https://simbad.cfa.harvard.edu/simbad/sim-id",
  "https://simbad.idia.ac.za/simbad/sim-id",
  "https://simbad.u-strasbg.fr/simbad/sim-id",
];

/**
 * Fetch a SIMBAD mirror response as text, with a 30 s timeout.
 *
 * @param {string} url - Fully formed SIMBAD query URL.
 * @returns {Promise<string>} Response body.
 * @throws {Error} On timeout, network failure, or non-2xx status.
 */
function simbadFetchText(url) {
  var controller = new AbortController();
  var timer = setTimeout(function () { controller.abort(); }, 30000);
  return fetch(url, { signal: controller.signal })
    .catch(function (error) {
      throw new Error(error && error.name === "AbortError" ? "timed out" : error.message);
    })
    .then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.text();
    })
    .finally(function () { clearTimeout(timer); });
}

/**
 * Look up a target name in SIMBAD, trying each mirror in turn.
 *
 * @param {string} name - Object name or identifier.
 * @returns {Promise<{name: string, raDeg: number, dec: number}>} Record
 *     per simbadParseRecord.
 * @throws {Error} When every mirror fails.
 */
async function simbadLookup(name) {
  var errors = [];
  for (var i = 0; i < SIMBAD_ENDPOINTS.length; i++) {
    var endpoint = SIMBAD_ENDPOINTS[i];
    try {
      var text = await simbadFetchText(
        endpoint + "?Ident=" + encodeURIComponent(name) +
        "&output.service=bibformat&output.format=ASCII"
      );
      var record = simbadParseRecord(text);
      if (record) return record;
      errors.push("no coordinates in response");
    } catch (error) {
      errors.push(error.message);
    }
  }
  throw new Error("Simbad lookup failed for \u201c" + name + "\u201d: " + errors.join("; "));
}

/**
 * Format non-negative hours or degrees as 'H:MM:SS.ss'.
 *
 * @param {number} value - Hours or degrees, >= 0.
 * @param {number} fracDigits - Fractional digits for the seconds field.
 * @returns {string} e.g. '18:36:56.30'.
 */
function sexagesimalString(value, fracDigits) {
  var top = Math.floor(value);
  var mid = Math.floor((value - top) * 60);
  var secText = ((value - top) * 3600 - mid * 60).toFixed(fracDigits);
  if (parseFloat(secText) >= 60) { secText = (0).toFixed(fracDigits); mid += 1; }
  if (mid === 60) { mid = 0; top += 1; }
  return top + ":" + String(mid).padStart(2, "0") + ":" + secText.padStart(5, "0");
}

// Simbad lookup: fill the form from the record, then add the target.
$("simbad-btn").addEventListener("click", function () {
  var name = $("new-name").value.trim();
  if (!name) {
    setStatus("Enter a name to look up in Simbad");
    return;
  }
  setStatus("Looking up \u201c" + name + "\u201d in Simbad\u2026");
  simbadLookup(name)
    .then(function (record) {
      $("new-name").value = record.name;
      $("new-ra").value = sexagesimalString(wrap360(record.raDeg) / 15, 2);
      $("new-dec").value = (record.dec < 0 ? "-" : "+") + sexagesimalString(Math.abs(record.dec), 2);
      $("add-btn").click();
      setStatus("Added \u201c" + record.name + "\u201d from Simbad");
    })
    .catch(function (error) {
      setStatus(error.message);
    });
});
// Copy a shareable link encoding the current state in the URL fragment.
$("copylink-btn").addEventListener("click", function () {
  var url = window.location.href.split("#")[0] + encodeShareFragment(JSON.stringify(state));
  function done(ok) {
    setStatus(ok ? "Link copied"
                 : "Could not copy automatically \u2014 copy the address bar instead");
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(url)
      .then(function () { done(true); }, function () { done(false); });
  } else {
    var ta = document.createElement("textarea");
    ta.value = url;
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    ta.remove();
    done(ok);
  }
});
$("export-btn").addEventListener("click", exportWorkspace);
// Save the current state under a name in browser storage.
$("ws-save-btn").addEventListener("click", function () {
  var name = window.prompt("Save workspace as:");
  if (!name || !name.trim()) return;
  name = name.trim();
  var ws = readWorkspaces();
  ws[name] = deepCopy(state);
  writeWorkspaces(ws);
  refreshWsSelect();
  $("ws-select").value = name;
  setStatus("Saved \u201c" + name + "\u201d");
});
// Load the selected named workspace.
$("ws-load-btn").addEventListener("click", function () {
  var name = $("ws-select").value;
  if (!name) { setStatus("No workspace selected"); return; }
  loadWorkspaceDoc(readWorkspaces()[name], "\u201c" + name + "\u201d");
});
// Delete the selected named workspace after confirmation.
$("ws-delete-btn").addEventListener("click", function () {
  var name = $("ws-select").value;
  if (!name) { setStatus("No workspace selected"); return; }
  if (!window.confirm("Delete workspace \u201c" + name + "\u201d?")) return;
  var ws = readWorkspaces();
  delete ws[name];
  writeWorkspaces(ws);
  refreshWsSelect();
  setStatus("Deleted \u201c" + name + "\u201d");
});
$("import-btn").addEventListener("click", function () { $("import-file").click(); });
// Import: parse the chosen JSON file and load it as the state.
$("import-file").addEventListener("change", function () {
  var file = this.files && this.files[0];
  this.value = "";
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function () {
    var s;
    try { s = JSON.parse(String(reader.result)); }
    catch (e) { setStatus("Could not parse \u201c" + file.name + "\u201d as JSON"); return; }
    loadWorkspaceDoc(s, "\u201c" + file.name + "\u201d");
  };
  reader.onerror = function () { setStatus("Could not read \u201c" + file.name + "\u201d"); };
  reader.readAsText(file);
});
// PNG export: re-draw, then download the canvas as an image.
$("png-btn").addEventListener("click", function () {
  hideTooltip();
  draw();
  var obsName = (state.observatory.name || "chart").toLowerCase()
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  var stamp = String(state.time_range.start || "").replace(" ", "T").slice(0, 16);
  var a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = "airmass-" + (obsName || "chart") + (stamp ? "-" + stamp : "") + ".png";
  document.body.appendChild(a);
  a.click();
  a.remove();
});
// Reset: clear the saved state and return to the embedded workspace.
$("reset-btn").addEventListener("click", function () {
  try { localStorage.removeItem(STATE_KEY); } catch (e) { /* ignore */ }
  state = normalizeState(deepCopy(INITIAL));
  view = null;
  hideTooltip();
  syncAllControls();
  setStatus("");
  draw();
});
window.addEventListener("resize", scheduleRedraw);

buildPresetSelect();
refreshWsSelect();
syncAllControls();
draw();
</script>
</body>
</html>
"""


def build_site(data: dict, out_path: str | Path) -> Path:
    """Write the workspace as an interactive single-file HTML page.

    The page carries the workspace as its initial state, plus the
    observatory presets, the palette, and the pre-blended twilight
    shading colors, and embeds the whole chart engine as JavaScript.

    Parameters
    ----------
    data : dict
        Workspace per the workspace module.
    out_path : str or Path
        Destination for the page; parent directories are created.

    Returns
    -------
    Path
        Path of the file written.
    """
    obs = from_dict(data.get("observatory", {}))
    shade = {
        kind: _blend_hex(color, alpha, AXES_BG)
        for kind, (color, alpha) in SHADE.items()
    }
    page = _TEMPLATE
    page = page.replace("__OBS_NAME__", html.escape(str(obs.name)))
    page = page.replace("__GENERATED__", datetime.now().isoformat(timespec="seconds"))
    page = page.replace("__INITIAL__", _js_json(data))
    page = page.replace("__OBSERVATORIES__", _js_json([to_dict(p) for p in PRESETS.values()]))
    page = page.replace("__PALETTE__", _js_json(PALETTE))
    page = page.replace("__SHADE__", _js_json(shade))
    for token, value in (
        ("__FIG_BG__", FIG_BG),
        ("__AXES_BG__", AXES_BG),
        ("__FG__", FG),
        ("__FG_DIM__", FG_DIM),
        ("__BORDER__", BORDER),
        ("__ERROR_FG__", ERROR_FG),
    ):
        page = page.replace(token, value)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
