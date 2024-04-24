import jax.numpy as jnp
import scipy.constants as scic


def convert_frequencies_to_wavelengths(f):
    """
    Convert frequency to wavelength in a JAX-compatible manner.

    Parameters:
    - f: Frequency in Hertz. Can be a float or a JAX array of floats.

    Returns:
    - Wavelength in meters. Has the same shape as input f.
    """

    wvl = scic.c / f
    return wvl


def convert_wavelengths_to_frequencies(wavelengths):
    """
    Convert wavelength to frequency in a JAX-compatible manner.

    Parameters:
    - wavelengths: Wavelength in meters. Can be a float or a JAX array of floats.

    Returns:
    - Frequency in Hertz. Has the same shape as input wavelengths.
    """
    frequencies = scic.c / wavelengths
    return frequencies


def get_frequencies_visible_light(num_wavelengths=1001):
    wavelengths = jnp.linspace(380 * scic.nano, 780 * scic.nano, num_wavelengths)
    frequencies = convert_wavelengths_to_frequencies(wavelengths)

    return frequencies

def get_frequencies_wide_visible_light(num_wavelengths=1001):
    wavelengths = jnp.linspace(300 * scic.nano, 900 * scic.nano, num_wavelengths)
    frequencies = convert_wavelengths_to_frequencies(wavelengths)

    return frequencies
