import jax.numpy as jnp
import scipy.constants as scic


def convert_frequencies_to_wavelengths(frequencies):
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    wavelengths = scic.c / frequencies
    return wavelengths


def convert_wavelengths_to_frequencies(wavelengths):
    assert isinstance(wavelengths, jnp.ndarray)
    assert wavelengths.ndim == 1

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
