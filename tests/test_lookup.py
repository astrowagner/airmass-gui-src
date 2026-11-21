"""Unit tests for airmass_gui.lookup (Simbad name lookup)."""

import unittest

from airmass_gui import lookup


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
