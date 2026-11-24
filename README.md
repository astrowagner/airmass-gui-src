# Airmass Charts

A tkinter GUI for creating airmass (X) charts of astronomical targets as seen from
an observatory. Default observatory: Mount Graham (LBT). Default demo target: Vega.

## Quick start (no setup)

From the project directory, run the bundled launcher:

```bash
./airmass-gui             # open the GUI
./airmass-gui --site      # write the interactive page to site/index.html
```

The launcher runs the project's local virtualenv (`.venv/`) directly — no
`pip install` or `source activate` needed.

## Install

```bash
pip install .            # or: pip install -e . for development
```

Requires Python >= 3.10 and an X display for the GUI (`tkinter` is used; on Linux
install `python3-tk` if missing).

## Run

```bash
airmass-gui
```

Headless render (no display needed; uses the saved workspace):

```bash
airmass-gui --no-gui --out chart.png
```

## Interactive site

```bash
airmass-gui --site                  # writes site/index.html
airmass-gui --site out/page.html    # or anywhere else
```

Writes a single self-contained HTML page — no other files, no build step.
It works fully offline; only the Simbad name lookup needs internet. Open it
directly in a browser or drop the file into a GitHub Pages repository.

The page is fully interactive: edit the observatory (preset or custom
coordinates/time zone), the start time and duration, and the target list, with
the chart re-rendering live. Edits persist in the browser (localStorage) until
you click **Reset**; **Export JSON** downloads the workspace as a JSON file the
desktop app can open (`File -> Open...`).

**Copy link** shares the exact setup through the URL's `#w=` fragment; the
Workspace panel also keeps named workspaces in the browser (**Import JSON**
loads one back) and **PNG** downloads the current chart. The **Windows**
option adds an airmass threshold line, per-target usable-window bands,
transit ticks on the time axis, and a per-night summary panel; **Moon**
overlays the Moon's airmass (truncated NAO 1954/ELP theory, ~0.015 deg from
the full ephemeris) with illumination and per-target moon separation in the
hover tooltip; **Civil** / **Nautical twilight** mark the sun's -6 and -12
degree crossings as thin vertical lines. The chart is navigable by wheel and
drag or by touch (one finger pans, two fingers pinch to zoom, tap shows the
tooltip); keyboard shortcuts: arrow keys pan, `+` / `-` zoom around the
center, `0` or `r` resets the window.

The astronomy (sun position, precession, airmass) is computed in embedded
JavaScript: entered ICRS coordinates are precessed to the mean equator/equinox
of date (IAU 2006; nutation omitted) before altitudes are computed, so curves
agree with the desktop app to ~0.01 deg (the difference is the omitted
nutation). The **Simbad** button in the Targets panel looks a name up in CDS
Simbad (mirrors tried in order), fills in the coordinates, and adds the target;
that lookup needs internet access.

## Update

Re-run the install after pulling changes:

```bash
pip install .
```

## Workspace

The workspace (observatory, targets, time range) auto-saves to
`~/.config/airmass-gui/workspace.json` on every change (targets added/removed/edited,
observatory or time range changed) and on exit. `File -> Open...` / `File -> Save As...`
work with any JSON workspace file.

## Targets

- RA: `HH:MM:SS.ss`, `HHhMMmSSs`, decimal hours, or decimal degrees (values > 24).
- Dec: `DD:MM:SS.ss`, `DDdMMmSSs`, or decimal degrees.
- Name lookup: type a name (e.g. `Vega`) and press **Lookup (Simbad)** — coordinates
  are fetched from CDS Simbad, filled in, and the target is added. Requires internet
  access to `simbad.u-strasbg.fr` (or `simbad.cds.unistra.fr`).

## Notes

- Chart axes: the vertical axis is altitude — 0° (horizon) at the bottom, 90°
  (zenith) at the top — and the right axis is labeled in degrees. The left axis
  shows airmass X at each value's true (nonlinear) altitude, so it reads inverted:
  X = 1 at the top, growing toward the horizon (ticks at 1, 1.5, 2, 4, 8, 20).
  Tick sets: `ALT_TICKS` / `AIRMASS_TICKS` in `src/airmass_gui/chart.py`.
- Airmass formula: Kasten & Young (1989), normalized so X(90°) = 1; curves are
  hidden below the horizon, and a dotted line marks X = 2.
- Coordinates and solar altitude via astropy; the x-axis is the observatory's
  local time (zoneinfo); twilight shading marks day / civil / nautical / astro / night.
- Dark theme: the window, controls, and chart all render in a dark color scheme
  (colors in `THEME` in `src/airmass_gui/app.py` and in `src/airmass_gui/chart.py`);
  the navigation toolbar icons lighten automatically on the dark background
  (matplotlib >= 3.9).
- LBT default: 32 deg 26' 34" N, 111 deg 09' 18" W, 2400 m, America/Phoenix.
- Tests: `python -m unittest discover -s tests -v`

## License

MIT — see [LICENSE](LICENSE).
