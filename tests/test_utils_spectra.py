import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import utils_spectra


def test_convert_frequencies_to_wavelengths():
    num_frequencies = 101

    frequencies = jnp.linspace(1e10, 100000e10, num_frequencies)
    wavelengths = utils_spectra.convert_frequencies_to_wavelengths(frequencies)
    frequencies_ = utils_spectra.convert_wavelengths_to_frequencies(wavelengths)

    assert wavelengths.ndim == 1
    assert wavelengths.shape[0] == num_frequencies
    assert frequencies_.ndim == 1
    assert frequencies_.shape[0] == num_frequencies

    onp.testing.assert_allclose(frequencies_, frequencies)


def test_convert_wavelengths_to_frequencies():
    num_wavelengths = 101

    wavelengths = jnp.linspace(1e-9, 100000e-9, num_wavelengths)
    frequencies = utils_spectra.convert_wavelengths_to_frequencies(wavelengths)
    wavelengths_ = utils_spectra.convert_frequencies_to_wavelengths(frequencies)

    assert frequencies.ndim == 1
    assert frequencies.shape[0] == num_wavelengths
    assert wavelengths_.ndim == 1
    assert wavelengths_.shape[0] == num_wavelengths

    onp.testing.assert_allclose(wavelengths_, wavelengths)


def test_get_frequencies_visible_light():
    num_wavelengths = 1234

    frequencies = utils_spectra.get_frequencies_visible_light(
        num_wavelengths=num_wavelengths
    )

    assert frequencies.ndim == 1
    assert frequencies.shape[0] == num_wavelengths
    onp.testing.assert_allclose(onp.min(frequencies), 3.84349305e14)
    onp.testing.assert_allclose(onp.max(frequencies), 7.88927521e14)


def test_get_frequencies_wide_visible_light():
    num_wavelengths = 1234

    frequencies = utils_spectra.get_frequencies_wide_visible_light(
        num_wavelengths=num_wavelengths
    )

    assert frequencies.ndim == 1
    assert frequencies.shape[0] == num_wavelengths
    onp.testing.assert_allclose(onp.min(frequencies), 3.331027e14)
    onp.testing.assert_allclose(onp.max(frequencies), 9.993082e14)
