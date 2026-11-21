"""Single-exposure S/N estimation for an imaging observation in the V band.

The model is a standard photon-counting calculation: the target and the
sky each feed a Poisson photon stream, the detector adds dark current and
read noise, and the Moon brightens the sky. The exposure time needed to
reach a target S/N is solved analytically (the equation is quadratic in
the exposure time). The same equations are ported to JavaScript in the
generated site (the JS-CORE block of ``airmass_gui.site``) and cross-
tested against this module in ``tests/test_site.py``.

The Moon darkening is anchored to the V-band measurements of
Houdashelt et al. (2005): a full Moon at zero separation raises the sky
by about 4.5 mag arcsec^-2, falling to about 2 mag arcsec^-2 at 90 deg.
"""
from __future__ import annotations

import math

# V-band photon flux at the top of the atmosphere for a V = 0 star,
# integrated over the V band, in photons cm^-2 s^-1. Vega is 3631 Jy;
# with a band width of about 880 A centered on 5500 A this is
# F_lam * dlam / (h c / lam) ~= 995 ph cm^-2 s^-1 A^-1 * 880 A.
SNR_V_ZERP = 8.75e5

# Dark-sky surface brightness, mag arcsec^-2, V band, no Moon.
SNR_SKY_DARK = 21.9

# Effective PSF pixel count, n_pix = SNR_PSF_COEF * (FWHM / scale)^2.
SNR_PSF_COEF = 1.1364


def snr_area(aperture_m: float) -> float:
    """Compute the collecting area of a circular telescope.

    Parameters
    ----------
    aperture_m : float
        Primary diameter D in metres.

    Returns
    -------
    float
        Collecting area in cm^2, pi (D / 2 * 100)^2.
    """
    radius_cm = 0.5 * aperture_m * 100.0
    return math.pi * radius_cm * radius_cm


def snr_moon_sep(sep_deg: float) -> float:
    """Evaluate the full-Moon sky darkening at a given separation.

    A constant plus two exponentials, anchored to give 4.5 mag at 0 deg
    and 2.15 mag at 90 deg, matching the V-band Moon-darkening
    measurements of Houdashelt et al. (2005).

    Parameters
    ----------
    sep_deg : float
        Angular separation between the Moon and the target in degrees.
        Negative values are clamped to 0.

    Returns
    -------
    float
        Sky darkening in magnitudes for a full Moon.
    """
    s = max(0.0, sep_deg)
    return 2.10 + 1.832 * math.exp(-s / 25.0) + 0.568 * math.exp(-s / 6.0)


def snr_moon_darkening(sep_deg: float, illum: float, moon_alt_deg: float) -> float:
    """Evaluate the Moon's sky darkening for a given phase and altitude.

    The full-Moon darkening curve is scaled by the illuminated fraction
    and by the sine of the Moon's altitude, so the contribution vanishes
    at new Moon and below the horizon.

    Parameters
    ----------
    sep_deg : float
        Moon-target separation in degrees.
    illum : float
        Illuminated fraction of the Moon, 0 (new) to 1 (full).
    moon_alt_deg : float
        Moon altitude in degrees.

    Returns
    -------
    float
        Sky darkening in magnitudes, >= 0.
    """
    dim = max(0.0, math.sin(math.radians(moon_alt_deg)))
    return illum * snr_moon_sep(sep_deg) * dim


def snr_sky_mag(sky_dark: float, delta_mag: float) -> float:
    """Compute the sky surface brightness including the Moon.

    Parameters
    ----------
    sky_dark : float
        Dark-sky surface brightness in mag arcsec^-2.
    delta_mag : float
        Moon darkening in magnitudes, >= 0 (a brighter sky).

    Returns
    -------
    float
        Sky surface brightness in mag arcsec^-2, <= sky_dark.
    """
    return sky_dark - delta_mag


def snr_source_rate(area_cm2: float, mag: float, airmass: float,
                    k_ext: float, eff: float) -> float:
    """Compute the target photoelectron rate.

    The V-band zero point is attenuated by the target brightness, by
    atmospheric extinction along the path (X - 1 airmasses of extra
    atmosphere), and by the total system efficiency.

    Parameters
    ----------
    area_cm2 : float
        Collecting area in cm^2.
    mag : float
        Target V magnitude.
    airmass : float
        Airmass X; extinction is applied for (X - 1) extra airmasses.
    k_ext : float
        Extinction coefficient in mag per airmass.
    eff : float
        Total system efficiency (optics times detector), 0 to 1.

    Returns
    -------
    float
        Target photoelectron rate in e- s^-1.
    """
    return (SNR_V_ZERP * area_cm2 * 10.0 ** (-0.4 * mag)
            * 10.0 ** (-0.4 * k_ext * (airmass - 1.0)) * eff)


def snr_sky_rate(area_cm2: float, sky_mag: float, airmass: float,
                 k_ext: float, eff: float, pix_scale_arcsec: float) -> float:
    """Compute the sky photoelectron rate per pixel.

    The sky zero point per square arcsec is integrated over the pixel
    area (plate scale squared), with the same extinction and efficiency
    factors as the source.

    Parameters
    ----------
    area_cm2 : float
        Collecting area in cm^2.
    sky_mag : float
        Sky surface brightness in mag arcsec^-2.
    airmass : float
        Airmass X.
    k_ext : float
        Extinction coefficient in mag per airmass.
    eff : float
        Total system efficiency, 0 to 1.
    pix_scale_arcsec : float
        Plate scale in arcsec per pixel.

    Returns
    -------
    float
        Sky photoelectron rate per pixel in e- s^-1 pixel^-1.
    """
    return (SNR_V_ZERP * area_cm2 * 10.0 ** (-0.4 * sky_mag)
            * 10.0 ** (-0.4 * k_ext * (airmass - 1.0)) * eff
            * pix_scale_arcsec * pix_scale_arcsec)


def snr_psf_pixels(fwhm_arcsec: float, pix_scale_arcsec: float) -> float:
    """Compute the effective number of pixels in the PSF.

    Parameters
    ----------
    fwhm_arcsec : float
        PSF full width at half maximum in arcsec.
    pix_scale_arcsec : float
        Plate scale in arcsec per pixel.

    Returns
    -------
    float
        Effective pixel count, 1.1364 (FWHM / plate scale)^2.
    """
    return SNR_PSF_COEF * (fwhm_arcsec / pix_scale_arcsec) ** 2


def _ac_r(src_rate: float, sky_rate_per_pix: float, n_pix: float,
          dark_rate_per_pix: float, read_noise: float):
    """Return the (a, c, r) coefficients of the S/N equation.

    a is the source rate in e- s^-1, c the sky-plus-dark rate summed
    over the PSF in e- s^-1, and r the total read-noise variance in
    e-^2.

    Parameters
    ----------
    src_rate : float
        Target rate in e- s^-1.
    sky_rate_per_pix : float
        Sky rate per pixel in e- s^-1 pixel^-1.
    n_pix : float
        Effective PSF pixel count.
    dark_rate_per_pix : float
        Dark current per pixel in e- s^-1 pixel^-1.
    read_noise : float
        Read noise per pixel in e- (rms).

    Returns
    -------
    tuple
        (a, c, r) as above.
    """
    a = src_rate
    c = n_pix * (sky_rate_per_pix + dark_rate_per_pix)
    r = n_pix * read_noise * read_noise
    return a, c, r


def snr_value(src_rate: float, sky_rate_per_pix: float, n_pix: float,
              dark_rate_per_pix: float, read_noise: float,
              t_sec: float) -> float:
    """Compute the S/N of a single exposure of duration t.

    SNR = a t / sqrt(a t + c t + r), where a is the source rate, c the
    sky-plus-dark rate over the PSF, and r the read-noise variance. The
    read noise is added once per exposure.

    Parameters
    ----------
    src_rate : float
        Target rate in e- s^-1.
    sky_rate_per_pix : float
        Sky rate per pixel in e- s^-1 pixel^-1.
    n_pix : float
        Effective PSF pixel count.
    dark_rate_per_pix : float
        Dark current per pixel in e- s^-1 pixel^-1.
    read_noise : float
        Read noise per pixel in e- (rms).
    t_sec : float
        Exposure time in seconds.

    Returns
    -------
    float
        Signal-to-noise ratio.
    """
    a, c, r = _ac_r(src_rate, sky_rate_per_pix, n_pix, dark_rate_per_pix,
                    read_noise)
    signal = a * t_sec
    noise = math.sqrt(signal + c * t_sec + r)
    return signal / noise


def snr_solve_time(src_rate: float, sky_rate_per_pix: float, n_pix: float,
                   dark_rate_per_pix: float, read_noise: float,
                   target_snr: float) -> float:
    """Solve for the exposure time that reaches a target S/N.

    Solves SNR = a t / sqrt((a + c) t + r) = T for the positive root,
    which is quadratic in t.

    Parameters
    ----------
    src_rate : float
        Target rate in e- s^-1.
    sky_rate_per_pix : float
        Sky rate per pixel in e- s^-1 pixel^-1.
    n_pix : float
        Effective PSF pixel count.
    dark_rate_per_pix : float
        Dark current per pixel in e- s^-1 pixel^-1.
    read_noise : float
        Read noise per pixel in e- (rms).
    target_snr : float
        Desired S/N, T > 0.

    Returns
    -------
    float
        Exposure time in seconds.

    Raises
    ------
    ZeroDivisionError
        If the source rate is zero (the target cannot be detected).
    """
    a, c, r = _ac_r(src_rate, sky_rate_per_pix, n_pix, dark_rate_per_pix,
                    read_noise)
    ac = a + c
    T = target_snr
    disc = (T * T * ac) ** 2 + 4.0 * a * a * T * T * r
    return (T * T * ac + math.sqrt(disc)) / (2.0 * a * a)


def snr_exposure(params: dict) -> dict:
    """Run the full single-exposure S/N model from its inputs.

    Parameters
    ----------
    params : dict
        Keys: aperture_m (m), fwhm_arcsec (arcsec),
        pix_scale_arcsec (arcsec pixel^-1), read_noise (e-),
        dark_rate_per_pix (e- s^-1 pixel^-1), eff (0 to 1), k_ext (mag
        per airmass), mag (V magnitude), airmass (X), sky_dark (mag
        arcsec^-2), moon_sep_deg (deg), moon_illum (0 to 1), and
        moon_alt_deg (deg).

    Returns
    -------
    dict
        area_cm2 (cm^2), delta_mag (mag), sky_mag (mag arcsec^-2),
        src_rate (e- s^-1), sky_rate_per_pix (e- s^-1 pixel^-1), and
        n_pix (pixels).
    """
    area = snr_area(params["aperture_m"])
    delta = snr_moon_darkening(params["moon_sep_deg"], params["moon_illum"],
                               params["moon_alt_deg"])
    sky_mag = snr_sky_mag(params["sky_dark"], delta)
    src_rate = snr_source_rate(area, params["mag"], params["airmass"],
                               params["k_ext"], params["eff"])
    sky_rate = snr_sky_rate(area, sky_mag, params["airmass"], params["k_ext"],
                            params["eff"], params["pix_scale_arcsec"])
    n_pix = snr_psf_pixels(params["fwhm_arcsec"], params["pix_scale_arcsec"])
    return {
        "area_cm2": area,
        "delta_mag": delta,
        "sky_mag": sky_mag,
        "src_rate": src_rate,
        "sky_rate_per_pix": sky_rate,
        "n_pix": n_pix,
    }
