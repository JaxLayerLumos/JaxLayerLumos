import jax.numpy as jnp
import numpy as np

from jaxlayerlumos import jaxlayerlumos as jll
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_spectra
from jaxlayerlumos import utils_layers


def test_stackrt_base_sizes():
    num_wavelengths = 123

    frequencies = utils_spectra.get_frequencies_wide_visible_light(
        num_wavelengths=num_wavelengths
    )
    wavelengths = utils_spectra.convert_frequencies_to_wavelengths(frequencies)

    n_stack = utils_materials.get_n_k_surrounded_by_air(["TiO2"], frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(jnp.array([2e-8]))

    index = 1
    r_TE, t_TE, r_TM, t_TM, theta, cos_theta = jll.stackrt_base(
        n_stack[index], d_stack, wavelengths[index], 37.2
    )

    assert isinstance(r_TE, jnp.ndarray)
    assert isinstance(r_TM, jnp.ndarray)
    assert isinstance(t_TE, jnp.ndarray)
    assert isinstance(t_TM, jnp.ndarray)
    assert isinstance(theta, jnp.ndarray)
    assert isinstance(cos_theta, jnp.ndarray)
    assert r_TE.ndim == 0
    assert r_TM.ndim == 0
    assert t_TE.ndim == 0
    assert t_TM.ndim == 0
    assert theta.ndim == 0
    assert cos_theta.ndim == 0


def test_stackrt_theta_sizes():
    num_wavelengths = 123

    frequencies = utils_spectra.get_frequencies_wide_visible_light(
        num_wavelengths=num_wavelengths
    )
    n_stack = utils_materials.get_n_k_surrounded_by_air(["TiO2"], frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(jnp.array([2e-8]))

    R_TE, T_TE, R_TM, T_TM = jll.stackrt_theta(n_stack, d_stack, frequencies, 37.2)

    assert isinstance(R_TE, jnp.ndarray)
    assert isinstance(R_TM, jnp.ndarray)
    assert isinstance(T_TE, jnp.ndarray)
    assert isinstance(T_TM, jnp.ndarray)
    assert R_TE.ndim == 1
    assert R_TM.ndim == 1
    assert T_TE.ndim == 1
    assert T_TM.ndim == 1
    assert R_TE.shape[0] == R_TM.shape[0] == num_wavelengths
    assert T_TE.shape[0] == T_TM.shape[0] == num_wavelengths


def test_stackrt_sizes():
    num_wavelengths = 123
    num_angles = 4

    frequencies = utils_spectra.get_frequencies_wide_visible_light(
        num_wavelengths=num_wavelengths
    )
    n_stack = utils_materials.get_n_k_surrounded_by_air(["TiO2"], frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(jnp.array([2e-8]))
    thetas = jnp.linspace(0, 89, num_angles)

    R_TE, T_TE, R_TM, T_TM = jll.stackrt(n_stack, d_stack, frequencies, thetas)

    assert isinstance(R_TE, jnp.ndarray)
    assert isinstance(R_TM, jnp.ndarray)
    assert isinstance(T_TE, jnp.ndarray)
    assert isinstance(T_TM, jnp.ndarray)
    assert R_TE.ndim == 2
    assert R_TM.ndim == 2
    assert T_TE.ndim == 2
    assert T_TM.ndim == 2
    assert R_TE.shape[0] == R_TM.shape[0] == num_angles
    assert T_TE.shape[0] == T_TM.shape[0] == num_angles
    assert R_TE.shape[1] == R_TM.shape[1] == num_wavelengths
    assert T_TE.shape[1] == T_TM.shape[1] == num_wavelengths


def test_stackrt0_sizes():
    num_wavelengths = 123

    frequencies = utils_spectra.get_frequencies_wide_visible_light(
        num_wavelengths=num_wavelengths
    )
    n_stack = utils_materials.get_n_k_surrounded_by_air(["TiO2"], frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(jnp.array([2e-8]))

    R_TE, T_TE, R_TM, T_TM = jll.stackrt0(n_stack, d_stack, frequencies)

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


def test_stackrt45_sizes():
    num_wavelengths = 123

    frequencies = utils_spectra.get_frequencies_wide_visible_light(
        num_wavelengths=num_wavelengths
    )
    n_stack = utils_materials.get_n_k_surrounded_by_air(["TiO2"], frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(jnp.array([2e-8]))

    R_TE, T_TE, R_TM, T_TM = jll.stackrt45(n_stack, d_stack, frequencies)

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
