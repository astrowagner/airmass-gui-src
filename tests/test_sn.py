"""Unit tests for the S/N estimation model (airmass_gui.sn).

The JavaScript port in the generated site is cross-tested against this
module in test_site.py; these tests pin the physics of the model.
"""
import math
import unittest

from airmass_gui import sn


class SnModelTests(unittest.TestCase):
    """Checks on the photon-counting model and its analytic inverse."""

    def test_area(self):
        """Collecting area scales as the square of the diameter."""
        self.assertAlmostEqual(sn.snr_area(2.0), math.pi * 100.0 ** 2, places=6)
        self.assertAlmostEqual(sn.snr_area(8.4), math.pi * (4.2 * 100.0) ** 2, places=4)

    def test_moon_sep_anchors(self):
        """The darkening curve hits its anchors and decreases with separation."""
        self.assertAlmostEqual(sn.snr_moon_sep(0.0), 4.5, delta=0.02)
        self.assertAlmostEqual(sn.snr_moon_sep(90.0), 2.15, delta=0.02)
        vals = [sn.snr_moon_sep(s) for s in (0, 15, 30, 45, 60, 90, 120)]
        for hi, lo in zip(vals, vals[1:]):
            self.assertGreaterEqual(hi, lo)
        self.assertAlmostEqual(sn.snr_moon_sep(-20.0), sn.snr_moon_sep(0.0), places=9)

    def test_moon_darkening(self):
        """Darkening scales with illumination and vanishes below the horizon."""
        self.assertAlmostEqual(sn.snr_moon_darkening(0.0, 0.0, 60.0), 0.0, places=9)
        self.assertAlmostEqual(sn.snr_moon_darkening(0.0, 1.0, -10.0), 0.0, places=9)
        self.assertAlmostEqual(
            sn.snr_moon_darkening(0.0, 1.0, 90.0), sn.snr_moon_sep(0.0), places=9)
        half = sn.snr_moon_darkening(30.0, 0.5, 45.0)
        full = sn.snr_moon_darkening(30.0, 1.0, 45.0)
        self.assertAlmostEqual(half, full * 0.5, places=9)

    def test_sky_mag(self):
        """A brighter (lower) sky magnitude as the darkening grows."""
        self.assertAlmostEqual(sn.snr_sky_mag(21.9, 0.0), 21.9, places=9)
        self.assertAlmostEqual(sn.snr_sky_mag(21.9, 4.5), 17.4, places=9)

    def test_source_rate_scaling(self):
        """Source rate follows the magnitude law, extinction, and efficiency."""
        base = sn.snr_source_rate(1000.0, 10.0, 1.0, 0.10, 1.0)
        self.assertGreater(base, 0)
        self.assertAlmostEqual(sn.snr_source_rate(1000.0, 11.0, 1.0, 0.10, 1.0),
                               base * 10.0 ** -0.4, places=6)
        self.assertAlmostEqual(sn.snr_source_rate(1000.0, 10.0, 1.0, 0.10, 0.5),
                               base * 0.5, places=6)
        self.assertLess(sn.snr_source_rate(1000.0, 10.0, 2.0, 0.10, 1.0), base)

    def test_sky_rate_per_pixel(self):
        """Sky rate per pixel scales with the pixel area and falls with sky mag."""
        base = sn.snr_sky_rate(1000.0, 21.0, 1.0, 0.10, 1.0, 0.2)
        self.assertGreater(base, 0)
        self.assertAlmostEqual(sn.snr_sky_rate(1000.0, 21.0, 1.0, 0.10, 1.0, 0.4),
                               base * 4.0, places=6)
        self.assertAlmostEqual(sn.snr_sky_rate(1000.0, 22.0, 1.0, 0.10, 1.0, 0.2),
                               base * 10.0 ** -0.4, places=6)

    def test_psf_pixels(self):
        """Effective pixel count scales as (FWHM / scale)^2."""
        self.assertAlmostEqual(sn.snr_psf_pixels(1.0, 0.2), sn.SNR_PSF_COEF * 25.0, places=6)
        self.assertAlmostEqual(sn.snr_psf_pixels(2.0, 0.2), sn.SNR_PSF_COEF * 100.0, places=6)

    def test_solve_inverts_value(self):
        """snr_solve_time is the inverse of snr_value at the target S/N."""
        src, sky, n_pix, dark, rn = 5.0, 2.0, 25.0, 0.1, 5.0
        for target in (5.0, 10.0, 20.0, 50.0):
            t = sn.snr_solve_time(src, sky, n_pix, dark, rn, target)
            self.assertGreater(t, 0)
            got = sn.snr_value(src, sky, n_pix, dark, rn, t)
            self.assertAlmostEqual(got, target, delta=1e-3)

    def test_brighter_target_needs_less_time(self):
        """A brighter target reaches the target S/N in a shorter exposure."""

        def t_for(mag):
            area = sn.snr_area(2.0)
            src = sn.snr_source_rate(area, mag, 1.2, 0.10, 0.5)
            sky = sn.snr_sky_rate(area, 21.0, 1.2, 0.10, 0.5, 0.2)
            n_pix = sn.snr_psf_pixels(1.0, 0.2)
            return sn.snr_solve_time(src, sky, n_pix, 0.1, 5.0, 10.0)

        self.assertLess(t_for(8.0), t_for(12.0))

    def test_exposure_pipeline(self):
        """snr_exposure wires the pieces together; the inverse recovers the target."""
        params = {
            "aperture_m": 2.0, "fwhm_arcsec": 1.0, "pix_scale_arcsec": 0.2,
            "read_noise": 5.0, "dark_rate_per_pix": 0.1, "eff": 0.5,
            "k_ext": 0.10, "mag": 10.0, "airmass": 1.2, "sky_dark": 21.9,
            "moon_sep_deg": 90.0, "moon_illum": 0.0, "moon_alt_deg": -10.0,
        }
        result = sn.snr_exposure(params)
        self.assertGreater(result["area_cm2"], 0)
        self.assertAlmostEqual(result["delta_mag"], 0.0, places=9)
        self.assertAlmostEqual(result["sky_mag"], 21.9, places=9)
        self.assertGreater(result["src_rate"], 0)
        self.assertGreater(result["sky_rate_per_pix"], 0)
        t = sn.snr_solve_time(result["src_rate"], result["sky_rate_per_pix"],
                              result["n_pix"], params["dark_rate_per_pix"],
                              params["read_noise"], 10.0)
        got = sn.snr_value(result["src_rate"], result["sky_rate_per_pix"],
                           result["n_pix"], params["dark_rate_per_pix"],
                           params["read_noise"], t)
        self.assertAlmostEqual(got, 10.0, delta=1e-3)
