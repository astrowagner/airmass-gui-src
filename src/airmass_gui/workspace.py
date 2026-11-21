"""Workspace persistence (atomic JSON save/load)."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from .observatories import DEFAULT_PRESET, PRESETS, to_dict

WORKSPACE_VERSION = 1

DEFAULT_OBJECT = {
    "name": "Vega",
    "ra": "18:36:56.30142",
    "dec": "+38:47:01.4506",
    "color": "#1f77b4",
}


def default_path() -> Path:
    return Path.home() / ".config" / "airmass-gui" / "workspace.json"


def new_workspace() -> dict:
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
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("workspace file must contain a JSON object")
    workspace = new_workspace()
    workspace.update({key: value for key, value in raw.items() if key in workspace})
    if not isinstance(workspace.get("objects"), list):
        workspace["objects"] = []
    return workspace


def save(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["modified"] = datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)
