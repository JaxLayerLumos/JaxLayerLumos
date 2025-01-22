import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import wrappers
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_spectra
from jaxlayerlumos import utils_layers


def test_stackrt_n_k_0_sizes():
    num_wavelengths = 123
    materials = onp.array(["Air", "TiO2", "Air"])

    frequencies = utils_spectra.get_frequencies_wide_visible_light(
        num_wavelengths=num_wavelengths
    )
    n_stack = utils_materials.get_n_k(materials, frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(jnp.array([2e-8]))

    R_TE, T_TE, R_TM, T_TM = wrappers.stackrt_n_k_0(n_stack, d_stack, frequencies)

    assert isinstance(R_TE, jnp.ndarray)
    assert isinstance(R_TM, jnp.ndarray)
    assert isinstance(T_TE, jnp.ndarray)
    assert isinstance(T_TM, jnp.ndarray)
    assert R_TE.ndim == 2
    assert R_TM.ndim == 2
    assert T_TE.ndim == 2
    assert T_TM.ndim == 2
    assert R_TE.shape[0] == R_TM.shape[0] == 1
    assert T_TE.shape[0] == T_TM.shape[0] == 1
    assert R_TE.shape[1] == R_TM.shape[1] == num_wavelengths
    assert T_TE.shape[1] == T_TM.shape[1] == num_wavelengths


def test_stackrt_n_k_45_sizes():
    num_wavelengths = 123
    materials = onp.array(["Air", "TiO2", "Air"])

    frequencies = utils_spectra.get_frequencies_wide_visible_light(
        num_wavelengths=num_wavelengths
    )
    n_stack = utils_materials.get_n_k(materials, frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(jnp.array([2e-8]))

    R_TE, T_TE, R_TM, T_TM = wrappers.stackrt_n_k_45(n_stack, d_stack, frequencies)

    assert isinstance(R_TE, jnp.ndarray)
    assert isinstance(R_TM, jnp.ndarray)
    assert isinstance(T_TE, jnp.ndarray)
    assert isinstance(T_TM, jnp.ndarray)
    assert R_TE.ndim == 2
    assert R_TM.ndim == 2
    assert T_TE.ndim == 2
    assert T_TM.ndim == 2
    assert R_TE.shape[0] == R_TM.shape[0] == 1
    assert T_TE.shape[0] == T_TM.shape[0] == 1
    assert R_TE.shape[1] == R_TM.shape[1] == num_wavelengths
    assert T_TE.shape[1] == T_TM.shape[1] == num_wavelengths
