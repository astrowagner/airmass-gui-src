"""Workspace persistence: the JSON document exchanged between the desktop app,
the generated site, and the user's filesystem.

The workspace is a JSON object with keys version, created, modified,
observatory, objects, and time_range. Objects are target entries
(name, ra, dec, color, plus optional extra fields such as magnitude,
spectral type, or an ephemeris).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from .observatories import DEFAULT_PRESET, PRESETS, RENAME_MAP, to_dict

# Schema version; bumped when the on-disk format changes incompatibly.
WORKSPACE_VERSION = 1

# Seed target for a fresh workspace.
DEFAULT_OBJECT = {
    "name": "Vega",
    "ra": "18:36:56.30142",
    "dec": "+38:47:01.4506",
    "color": "#1f77b4",
}


def default_path() -> Path:
    """Path of the auto-saved workspace.

    Returns
    -------
    Path
        ~/.config/airmass-gui/workspace.json
    """
    return Path.home() / ".config" / "airmass-gui" / "workspace.json"


def new_workspace() -> dict:
    """Create a fresh workspace document centered on the default preset.

    The time range starts at today's local 12:00 and spans one day.

    Returns
    -------
    dict
        A complete workspace with version, created, modified, observatory,
        objects, and time_range keys.
    """
    now = datetime.now().replace(microsecond=0)
    return {
        "version": WORKSPACE_VERSION,
        "created": now.isoformat(timespec="seconds"),
        "modified": now.isoformat(timespec="seconds"),
        "observatory": {**to_dict(PRESETS[DEFAULT_PRESET]), "preset": DEFAULT_PRESET},
        "objects": [dict(DEFAULT_OBJECT)],
        "time_range": {
            "start": now.replace(hour=12, minute=0, second=0).isoformat(timespec="minutes"),
            "days": 1.0,
        },
    }


def load(path: Path) -> dict:
    """Load a workspace document, merging it over the defaults.

    Unknown keys in the file are dropped; known keys keep the file's value.
    A missing or malformed objects entry becomes an empty list. Preset
    display names renamed after they were saved are rewritten to their
    current form. The preset table is authoritative: when the
    observatory carries a preset name (after the rename), its name,
    coordinates, elevation, and time zone are taken from the table and
    the saved numbers are ignored. An observatory with no preset name
    is kept as saved.

    Parameters
    ----------
    path : Path
        Path of the JSON file to read.

    Returns
    -------
    dict
        The merged workspace.

    Raises
    ------
    ValueError
        If the file does not contain a JSON object.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("workspace file must contain a JSON object")
    workspace = new_workspace()
    workspace.update({key: value for key, value in raw.items() if key in workspace})
    if not isinstance(workspace.get("objects"), list):
        workspace["objects"] = []
    obs = workspace.get("observatory")
    if isinstance(obs, dict):
        for key in ("preset", "name"):
            if obs.get(key) in RENAME_MAP:
                obs[key] = RENAME_MAP[obs[key]]
        if obs.get("preset") in PRESETS:
            obs.update(to_dict(PRESETS[obs["preset"]]))
    return workspace


def save(path: Path, data: dict) -> None:
    """Write a workspace atomically (temp file + rename), stamping modified.

    Parameters
    ----------
    path : Path
        Destination path; parent directories are created as needed.
    data : dict
        Workspace document to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["modified"] = datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)
