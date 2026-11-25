"""Unit tests for airmass_gui.lookup (Simbad name lookup)."""

import unittest
from pathlib import Path

from airmass_gui import lookup


class ParseRecordTest(unittest.TestCase):
    """Offline ASCII record parser against the archived Vega capture."""

    @classmethod
    def setUpClass(cls):
        cls.record = lookup._parse_ascii_record(
            (Path(__file__).resolve().parent
             / "fixtures" / "simbad_vega.txt").read_text(encoding="utf-8"))

    def test_fields(self):
        """Every astrometry and photometry field parses from the capture."""
        r = self.record
        self.assertEqual(r["name"], "Vega")
        self.assertAlmostEqual(r["ra"], 279.2347, delta=1e-4)
        self.assertAlmostEqual(r["dec"], 38.7837, delta=1e-4)
        self.assertEqual(r["mag"], "0.03")
        self.assertEqual(r["spec"], "A0V")
        self.assertAlmostEqual(r["plx"], 130.23)
        self.assertAlmostEqual(r["plxErr"], 0.36)
        self.assertAlmostEqual(r["pmRa"], 200.94)
        self.assertAlmostEqual(r["pmDec"], 286.23)
        self.assertAlmostEqual(r["pmRaErr"], 0.32)
        self.assertAlmostEqual(r["pmDecErr"], 0.4)
        self.assertEqual(r["rv"], "-13.5")
        self.assertEqual(set(r["fluxes"]),
                         {"U", "B", "V", "R", "I", "J", "H", "K"})
        self.assertAlmostEqual(r["fluxes"]["J"]["value"], -0.177)
        self.assertAlmostEqual(r["fluxes"]["J"]["err"], 0.206)
        self.assertIsNone(r["fluxes"]["U"]["err"])
        self.assertEqual(len(r["idents"]), 65)
        self.assertEqual(r["idTotal"], 65)
        self.assertEqual(r["idents"][0], "TIC 157587146")
        self.assertEqual(r["idents"][-1], "WEB 15681")
        self.assertIn("distance=7.68 pc", r["info"])

    def test_no_coordinates(self):
        """A record without an ICRS coordinate line parses to None."""
        self.assertIsNone(
            lookup._parse_ascii_record("C.D.S.  -  SIMBAD4\n\nSome object\n"))


class LookupTest(unittest.TestCase):
    """Live Simbad lookup (skipped when the mirrors are unreachable)."""
    def test_vega_lookup(self):
        """'Vega' resolves to RA 279.2346, Dec +38.7837 deg."""
        try:
            result = lookup.simbad_lookup("Vega", timeout=25)
        except LookupError as exc:
            self.skipTest(f"Simbad unreachable: {exc}")
        self.assertAlmostEqual(result["ra"], 279.2346, delta=0.01)
        self.assertAlmostEqual(result["dec"], 38.7837, delta=0.01)
        self.assertGreaterEqual(result["matches"], 1)
        self.assertAlmostEqual(result["plx"], 130.23, delta=0.5)
        self.assertGreaterEqual(result["idTotal"], 60)
