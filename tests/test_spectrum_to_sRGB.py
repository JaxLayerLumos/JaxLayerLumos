import jax.numpy as jnp
import numpy as onp

import colour

from jaxlayerlumos.colors import composite


def get_cmfs_illuminant_shape(increment=1.0):
    shape = colour.SpectralShape(380, 780, increment)
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
    illuminant = colour.SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)

    return cmfs, illuminant, shape


def get_spectrum_from_sRGB(sRGB):
    cmfs, illuminant, shape = get_cmfs_illuminant_shape()

    XYZ = colour.sRGB_to_XYZ(sRGB)
    spectrum = colour.XYZ_to_sd(XYZ, method="Jakob 2019", cmfs=cmfs, illuminant=illuminant)

    wavelengths = onp.array(spectrum.wavelengths)
    spectrum = onp.array(spectrum.values)

    return wavelengths, spectrum


def get_sRGB_colour(spectrum):
    cmfs, illuminant, shape = get_cmfs_illuminant_shape()

    sd = colour.SpectralDistribution(spectrum, shape)
    XYZ = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant, method='Integration') / 100
    sRGB = colour.XYZ_to_sRGB(XYZ)

    return sRGB


def get_sRGB_jaxcolors(wavelengths, spectrum):
    use_clipping = False
    return composite.spectrum_to_sRGB(wavelengths, spectrum, use_clipping=use_clipping)


def _test_one_sRGB(sRGB):
    wavelengths, spectrum = get_spectrum_from_sRGB(sRGB)

    sRGB_colour = get_sRGB_colour(spectrum)
    sRGB_jaxcolors = onp.array(get_sRGB_jaxcolors(jnp.array(wavelengths), jnp.array(spectrum)))

    onp.testing.assert_allclose(sRGB_jaxcolors, sRGB_colour, atol=1e-4, rtol=0)

def test_spectrum_to_sRGB():
    num_tests = 100
    random_state = onp.random.RandomState(42)

    for _ in range(0, num_tests):
        sRGB = random_state.uniform(low=0.0, high=1.0, size=(3, ))
        _test_one_sRGB(sRGB)
