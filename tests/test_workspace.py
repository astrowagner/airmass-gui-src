"""Unit tests for airmass_gui.workspace (workspace JSON save/load)."""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from airmass_gui.workspace import load, new_workspace, save


class WorkspaceTest(unittest.TestCase):
    """Save/load round-trips and default-filling."""
    def test_roundtrip(self):
        """save then load preserves objects, observatory, and time_range."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workspace.json"
            ws = new_workspace()
            ws["objects"].append({"name": "Sirius", "ra": "06:45:08.867", "dec": "-16:42:58.02", "color": "#ff7f0e"})
            save(path, ws)
            data = load(path)
            self.assertEqual(data["objects"], ws["objects"])
            self.assertEqual(data["observatory"]["name"], ws["observatory"]["name"])
            self.assertEqual(data["time_range"], ws["time_range"])
            self.assertIn("modified", data)

    def test_load_partial(self):
        """Missing keys in a saved file are filled with defaults."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workspace.json"
            path.write_text(json.dumps({"version": 1, "objects": []}))
            data = load(path)
            self.assertEqual(data["objects"], [])
            self.assertIn("observatory", data)
            self.assertIn("time_range", data)

    def test_load_migrates_renamed_preset(self):
        """Old preset display names are rewritten to the current names on load."""
        ws = new_workspace()
        ws["observatory"] = {
            "name": "LBT (Large Binocular Telescope)",
            "lat": 32.44278,
            "lon": -111.155,
            "elevation_m": 2400.0,
            "timezone": "America/Phoenix",
            "preset": "LBT (Large Binocular Telescope)",
        }
        ws["objects"] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workspace.json"
            save(path, ws)
            data = load(path)
        self.assertEqual(data["observatory"]["preset"], "Mount Graham (LBT)")
        self.assertEqual(data["observatory"]["name"], "Mount Graham (LBT)")

    def test_save_creates_dirs_and_updates_modified(self):
        """save creates parent directories, sets modified, leaves no .tmp."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "dir" / "workspace.json"
            data = new_workspace()
            save(path, data)
            modified = json.loads(path.read_text())["modified"]
            datetime.fromisoformat(modified)  # must parse
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_default_object_is_vega(self):
        """The default workspace is Vega at LBT."""
        ws = new_workspace()
        self.assertEqual(ws["objects"][0]["name"], "Vega")
        self.assertEqual(ws["observatory"]["name"], "Mount Graham (LBT)")
