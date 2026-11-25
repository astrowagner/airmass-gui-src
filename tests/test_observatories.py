"""Audit tests: the observatory preset table against the reference file.

tests/fixtures/observatories_reference.csv is the verbatim reference
table from the 2026-11 audit: site coordinates from the astropy site
registry (github.com/astropy/astropy-data, coordinates/sites.json) and
the Wikipedia pages named in its source column, checked by sha256.
PRESETS in src/airmass_gui/observatories.py is a copy of that table
and must agree with it row for row.
"""
import csv
import hashlib
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

from airmass_gui.observatories import PRESETS, RENAME_MAP

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "tests" / "fixtures" / "observatories_reference.csv"

SHA256 = "52d5873ab9514c9598c8328af836d4846579d4ee052f24d7f77a6c62affedfcb"


def reference_rows():
    """Rows of the reference table as dicts (name, lat_deg, lon_deg, elev_m, timezone, source)."""
    with REFERENCE.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ReferenceFileTest(unittest.TestCase):
    """The reference CSV is the verbatim audit table."""

    def test_sha256(self):
        """The file is byte-identical to the audit table."""
        digest = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
        self.assertEqual(digest, SHA256, "reference file does not match the audit table")

    def test_rows_and_sums(self):
        """33 rows; the column sums match the audit's values to 1e-4."""
        rows = reference_rows()
        self.assertEqual(len(rows), 33)
        self.assertLess(abs(sum(float(r["lat_deg"]) for r in rows) - 546.69294), 1e-4)
        self.assertLess(abs(sum(float(r["lon_deg"]) for r in rows) + 2352.00545), 1e-4)
        self.assertLess(abs(sum(float(r["elev_m"]) for r in rows) - 74990), 1e-4)


class PresetTableTest(unittest.TestCase):
    """PRESETS agrees with the reference table, one row at a time."""

    @classmethod
    def setUpClass(cls):
        cls.rows = reference_rows()

    def test_presets_match_reference(self):
        """Every preset equals its reference row: lat/lon to 1e-5 deg,
        elevation to 0.5 m, and the same time zone."""
        for row in self.rows:
            obs = PRESETS.get(row["name"])
            self.assertIsNotNone(obs, "no preset for %r" % row["name"])
            self.assertLess(abs(obs.lat - float(row["lat_deg"])), 1e-5,
                            row["name"] + " lat")
            self.assertLess(abs(obs.lon - float(row["lon_deg"])), 1e-5,
                            row["name"] + " lon")
            self.assertLess(abs(obs.elevation_m - float(row["elev_m"])), 0.5,
                            row["name"] + " elev")
            self.assertEqual(obs.timezone, row["timezone"], row["name"] + " timezone")

    def test_every_reference_row_has_a_preset(self):
        """The preset menu and the reference table have the same names."""
        for row in self.rows:
            self.assertIn(row["name"], PRESETS)
        self.assertEqual(len(PRESETS), len(self.rows),
                         "preset menu and reference table differ in size")

    def test_timezones_load(self):
        """Every time zone in the table is a loadable IANA name."""
        for obs in PRESETS.values():
            ZoneInfo(obs.timezone)

    def test_rename_map_values_are_presets(self):
        """Every value of RENAME_MAP is a current preset name."""
        for old, new in RENAME_MAP.items():
            self.assertIn(new, PRESETS, "%r -> %r is not a preset" % (old, new))

    def test_audit_renames_present(self):
        """The six audit renames map old names to current preset names."""
        expected = {
            "Brewster Hill (Hopkins 2.0)": "Mount Hopkins (MMT 6.5m)",
            "Brewster Hill (Whipple 4m)": "Mount Hopkins (Whipple 1.5m)",
            "Kitt Peak (LPL 4m)": "Kitt Peak (Mayall 4m)",
            "Mount McDonald (MPE)": "Mount Locke (McDonald)",
            "San Antonio Mountain (Palomar)": "Palomar Mountain (Hale 200-inch)",
            "San Pedro Martir (LMT 8m)": "San Pedro Martir (OAN 2.1m)",
        }
        for old, new in expected.items():
            self.assertEqual(RENAME_MAP.get(old), new,
                             "missing rename %r -> %r" % (old, new))
