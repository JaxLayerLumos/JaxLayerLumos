import jax.numpy as jnp

from jaxlayerlumos import utils_units


def convert_frequencies_to_wavelengths(frequencies):
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    wavelengths = utils_units.get_light_speed() / frequencies
    return wavelengths


def convert_wavelengths_to_energy(wavelengths):
    # energy is in eV
    assert isinstance(wavelengths, jnp.ndarray)
    assert wavelengths.ndim == 1

    energy = (
        utils_units.get_light_speed()
        * utils_units.get_planck_constant()
        / wavelengths
        / utils_units.get_elementary_charge()
    )
    return energy


def convert_wavelengths_to_frequencies(wavelengths):
    assert isinstance(wavelengths, jnp.ndarray)
    assert wavelengths.ndim == 1

    frequencies = utils_units.get_light_speed() / wavelengths
    return frequencies


def get_frequencies_visible_light(num_wavelengths=1001):
    wavelengths = jnp.linspace(
        380 * utils_units.get_nano(), 780 * utils_units.get_nano(), num_wavelengths
    )
    frequencies = convert_wavelengths_to_frequencies(wavelengths)

    return frequencies


def get_frequencies_wide_visible_light(num_wavelengths=1001):
    wavelengths = jnp.linspace(
        300 * utils_units.get_nano(), 900 * utils_units.get_nano(), num_wavelengths
    )
    frequencies = convert_wavelengths_to_frequencies(wavelengths)

    return frequencies
