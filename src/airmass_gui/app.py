"""tkinter front-end for airmass charts."""
from __future__ import annotations

import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from . import __version__, lookup  # noqa: E402
from .astro import format_dec_dms, format_ra_hours, sky_coord_from_entries  # noqa: E402
from .chart import FIG_BG, PALETTE, render_figure  # noqa: E402
from .observatories import CUSTOM, PRESETS, Observatory, from_dict, to_dict  # noqa: E402
from .workspace import default_path, load, new_workspace, save  # noqa: E402

# Colors of the dark theme.
THEME = {
    "bg": "#1e1e2e",
    "bg_alt": "#313244",
    "fg": "#cdd6f4",
    "fg_dim": "#9aa5c4",
    "accent": "#89b4fa",
    "active": "#45475a",
    "active_hi": "#585b70",
}


def _apply_style(root: tk.Tk) -> None:
    """Apply a dark theme to the ttk widgets and tkinter option defaults.

    Parameters
    ----------
    root : tk.Tk
        Main window the styles and option defaults attach to.
    """
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    bg, fg = THEME["bg"], THEME["fg"]
    style.configure(".", background=bg, foreground=fg,
                    bordercolor=THEME["active"], lightcolor=THEME["active"], darkcolor=bg)
    style.configure("TFrame", background=bg)
    style.configure("TLabel", background=bg, foreground=fg)
    style.configure("Status.TLabel", background=THEME["bg_alt"], foreground=fg,
                    relief="sunken", padding=(4, 2))
    style.configure("TLabelframe", background=bg, foreground=fg)
    style.configure("TLabelframe.Label", background=bg, foreground=fg)
    style.configure("TEntry", fieldbackground=THEME["bg_alt"], foreground=fg, insertcolor=fg,
                    bordercolor=THEME["active"], lightcolor=bg, darkcolor=bg)
    style.map("TEntry", fieldbackground=[("disabled", bg)],
              foreground=[("disabled", THEME["fg_dim"])])
    style.configure("TCombobox", fieldbackground=THEME["bg_alt"], background=bg, foreground=fg,
                    arrowcolor=fg, bordercolor=THEME["active"], lightcolor=bg, darkcolor=bg)
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", THEME["bg_alt"]), ("disabled", bg)],
        foreground=[("readonly", fg), ("disabled", THEME["fg_dim"])],
    )
    style.configure("TButton", background=THEME["active"], foreground=fg, bordercolor=THEME["active"],
                    lightcolor=THEME["active"], darkcolor=THEME["active"], focuscolor=THEME["active"],
                    padding=(8, 3))
    style.map("TButton",
              background=[("active", THEME["active_hi"]), ("pressed", "#2a2b3d")],
              foreground=[("active", fg), ("pressed", fg)])
    style.configure("TSpinbox", fieldbackground=THEME["bg_alt"], foreground=fg, insertcolor=fg,
                    arrowcolor=fg, bordercolor=THEME["active"], lightcolor=bg, darkcolor=bg)
    style.map("TSpinbox", fieldbackground=[("disabled", bg)],
              foreground=[("disabled", THEME["fg_dim"])])
    style.configure("TSeparator", background=THEME["active"])
    root.option_add("*TCombobox*Listbox*Background", THEME["bg_alt"])
    root.option_add("*TCombobox*Listbox*Foreground", fg)
    # The matplotlib navigation toolbar builds its tk buttons at creation time
    # and lightens its icons when the button background is dark, so set the
    # defaults before any toolbar exists.
    root.option_add("*Button*background", bg)
    root.option_add("*Button*foreground", fg)
    root.option_add("*Button*activeBackground", THEME["active"])
    root.option_add("*Button*activeForeground", fg)
    root.option_add("*Checkbutton*selectColor", THEME["active"])
    root.option_add("*Label*background", bg)
    root.option_add("*Label*foreground", fg)


class App:
    """tkinter window for airmass charts: controls on the left, chart on the right.

    The workspace dict (see the workspace module) is loaded from disk at
    startup and auto-saved on every change.
    """
    def __init__(self, root: tk.Tk | None = None):
        """Build the control panel, load the workspace, and arm the first redraw.

        Parameters
        ----------
        root : tkinter.Tk, optional
            Existing root window to build into; a new one is created when
            None (the default).
        """
        self.root = root or tk.Tk()
        self.root.title(f"Airmass Charts {__version__}")
        self.root.geometry("1280x780")
        self.root.config(bg=THEME["bg"])
        _apply_style(self.root)

        self.workspace_path = default_path()
        self.data = load(self.workspace_path) if self.workspace_path.exists() else new_workspace()

        self._redraw_job: int | None = None
        self._toolbar: NavigationToolbar2Tk | None = None

        self._build_menu()
        self._build_ui()
        self._sync_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self._schedule_redraw()

    def run(self) -> None:
        """Run the tkinter main loop."""
        self.root.mainloop()

    # ------------------------------------------------------------------ menu

    def _build_menu(self):
        """Build the File and Help menus."""
        menu_opts = {
            "bg": THEME["bg"],
            "fg": THEME["fg"],
            "activebackground": THEME["active"],
            "activeforeground": THEME["fg"],
            "relief": "flat",
            "bd": 0,
            "highlightthickness": 0,
        }
        menubar = tk.Menu(self.root, **menu_opts)
        file_menu = tk.Menu(menubar, tearoff=0, **menu_opts)
        file_menu.add_command(label="New workspace", command=self.new_workspace)
        file_menu.add_command(label="Open...", command=self.open_workspace)
        file_menu.add_command(label="Save As...", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_exit)
        help_menu = tk.Menu(menubar, tearoff=0, **menu_opts)
        help_menu.add_command(label="About", command=self.about)
        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    # -------------------------------------------------------------------- ui

    def _build_ui(self):
        """Build the control column, chart area, and status bar."""
        outer = ttk.Frame(self.root, padding=8)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        controls = ttk.Frame(outer, width=340)
        controls.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self._build_observatory(controls)
        self._build_time_range(controls)
        self._build_targets(controls)

        plot_area = ttk.Frame(outer)
        plot_area.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(8, 0))
        plot_area.rowconfigure(0, weight=1)
        plot_area.columnconfigure(0, weight=1)
        self.canvas_holder = ttk.Frame(plot_area)
        self.canvas_holder.grid(row=0, column=0, sticky="nsew")
        self.canvas_holder.rowconfigure(0, weight=1)
        self.canvas_holder.columnconfigure(0, weight=1)

        self.workspace_var = tk.StringVar()
        ttk.Label(outer, textvariable=self.workspace_var, wraplength=1100,
                  foreground=THEME["fg_dim"]).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(outer, textvariable=self.status_var, style="Status.TLabel", anchor="w").grid(
            row=3, column=0, columnspan=2, sticky="ew"
        )

    def _entry_row(self, parent, label, var, handler):
        """Add a labeled entry bound to a change handler.

        Parameters
        ----------
        parent : ttk widget
            Frame the row is packed into.
        label : str
            Left-hand label text.
        var : tk.StringVar
            Variable the entry edits.
        handler : callable
            Invoked on KeyRelease and Return.
        """
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=22, anchor="w").pack(side="left")
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<KeyRelease>", handler)
        entry.bind("<Return>", handler)

    def _build_observatory(self, parent):
        """Build the observatory panel: preset combobox and the latitude, longitude, elevation, and time-zone entries.

    Parameters
    ----------
    parent : tkinter.Widget
        Frame to build the panel into.
        """
        frame = ttk.LabelFrame(parent, text="Observatory", padding=6)
        frame.pack(fill="x", pady=(0, 6))

        self.obs_var = tk.StringVar()
        combo = ttk.Combobox(frame, textvariable=self.obs_var, state="readonly")
        combo["values"] = list(PRESETS) + [CUSTOM]
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", self._on_obs_select)

        self.lat_var = tk.StringVar()
        self.lon_var = tk.StringVar()
        self.elev_var = tk.StringVar()
        self.tz_var = tk.StringVar()
        self._entry_row(frame, "Latitude (deg):", self.lat_var, self._on_obs_field_edit)
        self._entry_row(frame, "Longitude (deg):", self.lon_var, self._on_obs_field_edit)
        self._entry_row(frame, "Elevation (m):", self.elev_var, self._on_obs_field_edit)
        self._entry_row(frame, "Time zone:", self.tz_var, self._on_obs_field_edit)

    def _build_time_range(self, parent):
        """Build the time-range panel: local start stamp and duration in days.

    Parameters
    ----------
    parent : tkinter.Widget
        Frame to build the panel into.
        """
        frame = ttk.LabelFrame(parent, text="Time range (local)", padding=6)
        frame.pack(fill="x", pady=(0, 6))
        self.start_var = tk.StringVar()
        self.days_var = tk.StringVar()
        self._entry_row(frame, "Start (YYYY-MM-DD HH:MM):", self.start_var, self._on_change)
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="Duration (days):", width=22, anchor="w").pack(side="left")
        spin = ttk.Spinbox(row, from_=0.25, to=30.0, increment=0.25, width=10, textvariable=self.days_var)
        spin.pack(side="left")
        spin.bind("<KeyRelease>", self._on_change)
        spin.bind("<Return>", self._on_change)

    def _build_targets(self, parent):
        """Build the targets panel: object listbox, RA/Dec entries, add/update/remove buttons, and the Simbad lookup row.

    Parameters
    ----------
    parent : tkinter.Widget
        Frame to build the panel into.
        """
        frame = ttk.LabelFrame(parent, text="Targets", padding=6)
        frame.pack(fill="x")

        self.obj_list = tk.Listbox(
            frame,
            height=4,
            exportselection=False,
            bg=THEME["bg_alt"],
            fg=THEME["fg"],
            selectbackground=THEME["accent"],
            selectforeground=THEME["bg"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.obj_list.pack(fill="x")
        self.obj_list.bind("<ButtonRelease-1>", self._on_select_object)

        self.name_var = tk.StringVar()
        self.ra_var = tk.StringVar()
        self.dec_var = tk.StringVar()
        self.color_var = tk.StringVar()
        self._entry_row(frame, "Name:", self.name_var, self._on_change)
        self._entry_row(frame, "RA (h:m:s / deg):", self.ra_var, self._on_change)
        self._entry_row(frame, "Dec (d:m:s / deg):", self.dec_var, self._on_change)
        color_row = ttk.Frame(frame)
        color_row.pack(fill="x", pady=1)
        ttk.Label(color_row, text="Color:", width=22, anchor="w").pack(side="left")
        color_combo = ttk.Combobox(color_row, textvariable=self.color_var, state="readonly", values=PALETTE)
        color_combo.pack(side="left")
        color_combo.bind("<<ComboboxSelected>>", self._on_change)

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(buttons, text="Add", command=self._add_object).pack(side="left", padx=(0, 4))
        ttk.Button(buttons, text="Update", command=self._update_object).pack(side="left", padx=(0, 4))
        ttk.Button(buttons, text="Remove", command=self._remove_object).pack(side="left")

        lookup_row = ttk.Frame(frame)
        lookup_row.pack(fill="x", pady=(8, 0))
        self.lookup_var = tk.StringVar()
        entry = ttk.Entry(lookup_row, textvariable=self.lookup_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", self._lookup)
        self.lookup_btn = ttk.Button(lookup_row, text="Lookup (Simbad)", command=self._lookup)
        self.lookup_btn.pack(side="left", padx=(4, 0))
        self.lookup_status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.lookup_status_var, wraplength=310,
                  foreground=THEME["fg_dim"]).pack(
            fill="x", pady=(4, 0)
        )

    # ------------------------------------------------------------- ui sync

    def _set(self, var: tk.StringVar, value: str):
        """Set a tk variable from code.

        Parameters
        ----------
        var : tk.StringVar
            Variable to update.
        value : str
            New value.
        """
        var.set(value)

    def _sync_ui(self):
        """Push the workspace into every control and name the workspace path in the footer."""
        obs = from_dict(self.data["observatory"])
        preset = self.data["observatory"].get("preset")
        self.obs_var.set(preset if preset in PRESETS else CUSTOM)
        self._set(self.lat_var, f"{obs.lat:.5f}")
        self._set(self.lon_var, f"{obs.lon:.5f}")
        self._set(self.elev_var, f"{obs.elevation_m:.0f}")
        self._set(self.tz_var, obs.timezone)
        time_range = self.data.get("time_range", {})
        self._set(self.start_var, time_range.get("start", ""))
        self._set(self.days_var, f"{time_range.get('days', 1.0):g}")
        self._refresh_objects()
        self.workspace_var.set(f"Workspace: {self.workspace_path} (auto-saved on every change)")

    def _refresh_objects(self):
        """Rebuild the object listbox from the workspace objects."""
        self.obj_list.delete(0, "end")
        for i, obj in enumerate(self.data.get("objects", [])):
            self.obj_list.insert("end", f"{i + 1}. {obj.get('name', '?')}   RA {obj.get('ra', '?')}   Dec {obj.get('dec', '?')}")

    # ------------------------------------------------------------- handlers

    def _on_obs_select(self, _event=None):
        """Fill the observatory fields when a preset is chosen from the combobox.

    Parameters
    ----------
    _event : tkinter.Event, optional
        Callback event from the widget binding; not used.
        """
        name = self.obs_var.get()
        if name in PRESETS:
            obs = PRESETS[name]
            self._set(self.lat_var, f"{obs.lat:.5f}")
            self._set(self.lon_var, f"{obs.lon:.5f}")
            self._set(self.elev_var, f"{obs.elevation_m:.0f}")
            self._set(self.tz_var, obs.timezone)
        self._on_change()

    def _on_obs_field_edit(self, _event=None):
        """Mark the observatory Custom when an entry drifts from the selected preset.

    Parameters
    ----------
    _event : tkinter.Event, optional
        Callback event from the widget binding; not used.
        """
        name = self.obs_var.get()
        if name in PRESETS:
            obs = PRESETS[name]
            try:
                changed = (
                    abs(float(self.lat_var.get()) - obs.lat) > 1e-4
                    or abs(float(self.lon_var.get()) - obs.lon) > 1e-4
                    or abs(float(self.elev_var.get()) - obs.elevation_m) > 0.5
                    or self.tz_var.get().strip() != obs.timezone
                )
            except ValueError:
                changed = True
            if changed:
                self.obs_var.set(CUSTOM)
        self._on_change()

    def _on_change(self, _event=None):
        """Commit the observatory and time-range fields, save, and schedule a redraw.

    Parameters
    ----------
    _event : tkinter.Event, optional
        Callback event from the widget binding; not used.
        """
        self._commit_observatory()
        self._commit_time_range()
        self._save()
        self._schedule_redraw()

    def _commit_observatory(self):
        """Write the observatory panel into the workspace, keeping the preset key while the fields match the preset."""
        try:
            lat = float(self.lat_var.get())
            lon = float(self.lon_var.get())
            elev = float(self.elev_var.get())
        except ValueError:
            return
        name = self.obs_var.get()
        if name in PRESETS:
            obs_dict = to_dict(PRESETS[name])
            obs_dict["preset"] = name
        else:
            obs_dict = to_dict(Observatory("Custom observatory", lat, lon, elev, self.tz_var.get().strip() or "UTC"))
            obs_dict["preset"] = None
        self.data["observatory"] = obs_dict

    def _commit_time_range(self):
        """Write the start stamp and duration into the workspace, ignoring unparseable values."""
        time_range = self.data.setdefault("time_range", {})
        try:
            dt = datetime.fromisoformat(self.start_var.get().strip())
            time_range["start"] = dt.isoformat(timespec="minutes")
        except ValueError:
            pass
        try:
            time_range["days"] = float(self.days_var.get())
        except ValueError:
            pass

    def _on_select_object(self, _event=None):
        """Load the selected listbox object into the RA/Dec entries.

    Parameters
    ----------
    _event : tkinter.Event, optional
        Callback event from the widget binding; not used.
        """
        sel = self.obj_list.curselection()
        if not sel:
            return
        obj = self.data["objects"][sel[0]]
        self._set(self.name_var, obj.get("name", ""))
        self._set(self.ra_var, obj.get("ra", ""))
        self._set(self.dec_var, obj.get("dec", ""))
        self._set(self.color_var, obj.get("color", PALETTE[0]))

    def _next_color(self) -> str:
        """Next unused palette color for a new object.

        Returns
        -------
        str
            Hex color from PALETTE (chart module).
        """
        used = [obj.get("color") for obj in self.data.get("objects", [])]
        for color in PALETTE:
            if color not in used:
                return color
        return PALETTE[len(self.data.get("objects", [])) % len(PALETTE)]

    def _object_from_entries(self) -> dict:
        """Read the target entries into a workspace object dict, validating RA/Dec.

        Returns
        -------
        dict
            name, ra, dec, color (see the workspace module for the schema).

        Raises
        ------
        ValueError
            If the RA or Dec entry does not parse.
        """
        name = self.name_var.get().strip() or f"Target {len(self.data.get('objects', [])) + 1}"
        ra = self.ra_var.get().strip()
        dec = self.dec_var.get().strip()
        sky_coord_from_entries(ra, dec)  # raises ValueError on bad input
        return {
            "name": name,
            "ra": ra,
            "dec": dec,
            "color": self.color_var.get() or self._next_color(),
        }

    def _add_object(self):
        """Add the entry fields as a new object, or report a parse error in the status bar."""
        try:
            obj = self._object_from_entries()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        self.data.setdefault("objects", []).append(obj)
        self._refresh_objects()
        self.obj_list.selection_set(len(self.data["objects"]) - 1)
        self._on_change()
        self.status_var.set(f"Added {obj['name']}")

    def _update_object(self):
        """Replace the selected object with the entry fields, or report why not."""
        sel = self.obj_list.curselection()
        if not sel:
            self.status_var.set("Select an object to update")
            return
        try:
            obj = self._object_from_entries()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        self.data["objects"][sel[0]] = obj
        self._refresh_objects()
        self._on_change()
        self.status_var.set(f"Updated {obj['name']}")

    def _remove_object(self):
        """Remove the selected object from the workspace."""
        sel = self.obj_list.curselection()
        if not sel:
            self.status_var.set("Select an object to remove")
            return
        removed = self.data["objects"].pop(sel[0])
        self._refresh_objects()
        self._on_change()
        self.status_var.set(f"Removed {removed.get('name', 'object')}")

    # -------------------------------------------------------------- simbad

    def _lookup(self, _event=None):
        """Start a background Simbad lookup for the name in the lookup entry.

    Parameters
    ----------
    _event : tkinter.Event, optional
        Callback event from the widget binding; not used.
        """
        name = self.lookup_var.get().strip()
        if not name:
            self.status_var.set("Type a target name to look up (e.g., Vega)")
            return
        self.lookup_btn.config(state="disabled")
        self.lookup_status_var.set(f"Querying Simbad for {name} ...")
        threading.Thread(target=self._lookup_worker, args=(name,), daemon=True).start()

    def _lookup_worker(self, name: str):
        """Query Simbad in a worker thread and report back on the main loop.

        Parameters
        ----------
        name : str
            Target name to query.
        """
        try:
            result = lookup.simbad_lookup(name)
            self.root.after(0, self._lookup_done, name, result, None)
        except Exception as exc:
            self.root.after(0, self._lookup_done, name, None, f"{exc.__class__.__name__}: {exc}")

    def _lookup_done(self, name: str, result, error):
        """Add the looked-up target on success, or show the error in the status bar.

        Parameters
        ----------
        name : str
            Name that was queried.
        result : dict or None
            Lookup result per lookup.simbad_lookup, or None on failure.
        error : str or None
            Error message, or None on success.
        """
        self.lookup_btn.config(state="normal")
        if error is not None:
            self.lookup_status_var.set(f"Lookup failed: {error}")
            return
        ra_text = format_ra_hours(result["ra"])
        dec_text = format_dec_dms(result["dec"])
        target_name = result.get("name") or name
        self._set(self.name_var, target_name)
        self._set(self.ra_var, ra_text)
        self._set(self.dec_var, dec_text)
        obj = {"name": target_name, "ra": ra_text, "dec": dec_text, "color": self._next_color()}
        self.data.setdefault("objects", []).append(obj)
        self._refresh_objects()
        self.obj_list.selection_set(len(self.data["objects"]) - 1)
        self._on_change()
        extra = f" ({result['matches']} matches)" if result.get("matches", 1) > 1 else ""
        self.lookup_status_var.set(f"{result.get('info', '')}{extra}")
        self.status_var.set(f"Added {target_name} from Simbad")

    # --------------------------------------------------------------- chart

    def _schedule_redraw(self):
        """Schedule a chart redraw 150 ms from now, replacing any pending one."""
        if self._redraw_job is not None:
            self.root.after_cancel(self._redraw_job)
        self._redraw_job = self.root.after(150, self._redraw)

    def _redraw(self, *_):
        """Rebuild the chart canvas and navigation toolbar from the workspace.

    Parameters
    ----------
    *_
        Callback arguments (a scheduled after() job or event); not used.
        """
        self._redraw_job = None
        for child in self.canvas_holder.winfo_children():
            child.destroy()
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        try:
            fig = render_figure(self.data)
        except Exception as exc:
            self.status_var.set(f"Chart error: {exc}")
            return
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_holder)
        canvas.get_tk_widget().config(bg=FIG_BG)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        canvas.draw()
        self._toolbar = NavigationToolbar2Tk(canvas, self.canvas_holder)
        self._toolbar.config(bg=THEME["bg"])
        self._toolbar.grid(row=1, column=0, sticky="ew")

    # -------------------------------------------------------- workspace

    def _save(self, *_):
        """Save the workspace to disk, reporting failures in the status bar.

    Parameters
    ----------
    *_
        Callback arguments (a scheduled after() job or event); not used.
        """
        try:
            save(self.workspace_path, self.data)
        except Exception as exc:
            self.status_var.set(f"Save failed: {exc}")

    def on_exit(self):
        """Save the workspace and close the window."""
        self._save()
        self.root.destroy()

    def new_workspace(self):
        """Discard the current workspace for a fresh default one."""
        self.data = new_workspace()
        self._sync_ui()
        self._on_change()

    def open_workspace(self):
        """Open a workspace JSON from a file dialog and switch to it."""
        path = filedialog.askopenfilename(
            title="Open workspace", filetypes=[("JSON workspace", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.data = load(Path(path))
        except Exception as exc:
            messagebox.showerror("Airmass Charts", f"Could not open workspace:\n{exc}")
            return
        self.workspace_path = Path(path)
        self._sync_ui()
        self._on_change()
        self.status_var.set(f"Opened {path}")

    def save_as(self):
        """Save the workspace to a path chosen in a file dialog."""
        path = filedialog.asksaveasfilename(
            title="Save workspace as", defaultextension=".json", filetypes=[("JSON workspace", "*.json")]
        )
        if not path:
            return
        self.workspace_path = Path(path)
        self._save()
        self._sync_ui()
        self.status_var.set(f"Workspace saved to {path}")

    def about(self):
        """Show the about box with version, method credits, and the workspace path."""
        messagebox.showinfo(
            "Airmass Charts",
            f"Airmass Charts {__version__}\n\n"
            "Airmass: Kasten & Young (1989).\n"
            "Coordinates and solar altitude: astropy.\n"
            "Name lookup: CDS Simbad (requires internet).\n\n"
            f"Workspace: {self.workspace_path}\n"
            "(auto-saved on every change and on exit)",
        )
