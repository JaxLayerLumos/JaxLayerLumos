import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos import stackrt
from jaxlayerlumos.utils_materials import (
    interpolate_material,
    get_n_k_surrounded_by_air,
)
from jaxlayerlumos.utils_spectra import get_frequencies_wide_visible_light
from jaxlayerlumos.utils_layers import get_thicknesses_surrounded_by_air


def test_sizes():
    num_wavelengths = 123
    num_angles = 4

    frequencies = get_frequencies_wide_visible_light(num_wavelengths=num_wavelengths)
    n_stack = get_n_k_surrounded_by_air(["TiO2"], frequencies)
    d_stack = get_thicknesses_surrounded_by_air(jnp.array([2e-8]))
    thetas = jnp.linspace(0, 89, num_angles)

    R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, thetas)

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


def test_angles():
    wavelengths = jnp.linspace(300e-9, 900e-9, 3)
    frequencies = scic.c / wavelengths

    n_TiO2, k_TiO2 = interpolate_material("TiO2", frequencies)
    n_k_TiO2 = n_TiO2 + 1j * k_TiO2

    n_air = jnp.ones_like(wavelengths)
    n_stack = jnp.vstack([n_air, n_k_TiO2, n_air]).T
    d_stack = jnp.array([0, 2e-8, 0])
    thetas = jnp.linspace(0, 89, 3)

    R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, thetas)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    expected_R_avg = jnp.array(
        [
            [
                0.5075084159634257,
                0.18726655434023687,
                0.08370762967721315,
            ],
            [
                0.4877839691834523,
                0.1945178061281731,
                0.09157449689223793,
            ],
            [
                0.9498492283810344,
                0.9783936773043997,
                0.9540558921256859,
            ],
        ]
    )

    expected_T_avg = jnp.array(
        [
            [
                0.17839133609006402,
                0.8127334404810548,
                0.916292370322787,
            ],
            [
                0.18801137946236623,
                0.8054821885944445,
                0.9084255031077624,
            ],
            [
                0.006495561010086865,
                0.021606322148813736,
                0.04594410787425793,
            ],
        ]
    )

    print("R_avg")
    for elem_1 in R_avg:
        for elem_2 in elem_1:
            print(elem_2)

    print("T_avg")
    for elem_1 in T_avg:
        for elem_2 in elem_1:
            print(elem_2)

    np.testing.assert_allclose(R_avg, expected_R_avg)
    np.testing.assert_allclose(T_avg, expected_T_avg)
