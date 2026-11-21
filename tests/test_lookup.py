import unittest

from airmass_gui import lookup


class LookupTest(unittest.TestCase):
    def test_vega_lookup(self):
        try:
            result = lookup.simbad_lookup("Vega", timeout=25)
        except LookupError as exc:
            self.skipTest(f"Simbad unreachable: {exc}")
        self.assertAlmostEqual(result["ra"], 279.2346, delta=0.01)
        self.assertAlmostEqual(result["dec"], 38.7837, delta=0.01)
        self.assertGreaterEqual(result["matches"], 1)
