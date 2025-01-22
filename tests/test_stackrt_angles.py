import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import stackrt
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_spectra
from jaxlayerlumos import utils_layers
from jaxlayerlumos import utils_units


def test_sizes():
    num_wavelengths = 123
    num_angles = 4

    frequencies = utils_spectra.get_frequencies_wide_visible_light(
        num_wavelengths=num_wavelengths
    )
    materials = ["TiO2"]
    n_stack = utils_materials.get_n_k_surrounded_by_air(materials, frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(jnp.array([2e-8]))
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


def test_angles_1():
    num_wavelengths = 5
    frequencies = utils_spectra.get_frequencies_visible_light(
        num_wavelengths=num_wavelengths
    )

    materials = ["Air", "TiO2", "Air"]
    d_stack = jnp.array([0, 2e-8, 0])
    thetas = jnp.linspace(0, 89, 3)
    n_k = utils_materials.get_n_k(materials, frequencies)

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, d_stack, frequencies, thetas)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    expected_R_avg = jnp.array(
        [
            [
                0.4770790218584379,
                0.2918301244526284,
                0.20026632179451573,
                0.14616448215891184,
                0.11132762140037065,
            ],
            [
                0.4558617206305765,
                0.2910941832925791,
                0.20684237462156152,
                0.15476340159980778,
                0.11998132563573674,
            ],
            [
                0.9903190571800133,
                0.9857622632656011,
                0.979727833282493,
                0.9726668981700657,
                0.9646925772238752,
            ],
        ]
    )

    expected_T_avg = jnp.array(
        [
            [
                0.5198973861437665,
                0.7081681748619181,
                0.7997336666045016,
                0.8538355175673582,
                0.8886723785996291,
            ],
            [
                0.5410727876996986,
                0.7089041021153872,
                0.793157613579433,
                0.8452365981191836,
                0.8800186743642635,
            ],
            [
                0.009481699666494146,
                0.014237605935075181,
                0.020272165555583664,
                0.027333101794498202,
                0.035307422776128854,
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

    onp.testing.assert_allclose(R_avg, expected_R_avg)
    onp.testing.assert_allclose(T_avg, expected_T_avg)


def test_angles_2():
    wavelengths = jnp.array([300e-9])
    frequencies = utils_units.get_light_speed() / wavelengths

    materials = onp.array(["Air", "FusedSilica", "Si3N4"])
    thickness_materials = [0, 2.91937911, 0]
    theta = 47.1756

    thicknesses = jnp.array(thickness_materials)
    n_k = utils_materials.get_n_k(materials, frequencies)

    thicknesses *= utils_units.get_nano()

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies, theta)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    print("R_TE")
    for elem_1 in R_TE:
        for elem_2 in elem_1:
            print(elem_2)

    print("T_TE")
    for elem_1 in T_TE:
        for elem_2 in elem_1:
            print(elem_2)

    print("R_TM")
    for elem_1 in R_TM:
        for elem_2 in elem_1:
            print(elem_2)

    print("T_TM")
    for elem_1 in T_TM:
        for elem_2 in elem_1:
            print(elem_2)

    expected_R_avg = jnp.array([[0.24877806962502294 + 0.048295322372087564]]) / 2
    expected_T_avg = jnp.array([[0.7512219303749771 + 0.9517046776279122]]) / 2

    onp.testing.assert_allclose(R_avg, expected_R_avg)
    onp.testing.assert_allclose(T_avg, expected_T_avg)


def test_angles_3():
    wavelengths = jnp.array([300e-9])
    frequencies = utils_units.get_light_speed() / wavelengths

    materials = onp.array(["Air", "Ag", "Cr"])
    thickness_materials = [0, 9.36793259, 0.0]
    theta = 34.767507632418315

    thicknesses = jnp.array(thickness_materials)
    n_k = utils_materials.get_n_k(materials, frequencies)

    thicknesses *= utils_units.get_nano()
    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies, theta)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    print("R_TE")
    for elem_1 in R_TE:
        for elem_2 in elem_1:
            print(elem_2)

    print("T_TE")
    for elem_1 in T_TE:
        for elem_2 in elem_1:
            print(elem_2)

    print("R_TM")
    for elem_1 in R_TM:
        for elem_2 in elem_1:
            print(elem_2)

    print("T_TM")
    for elem_1 in T_TM:
        for elem_2 in elem_1:
            print(elem_2)

    expected_R_avg = jnp.array([[0.48574877]])
    expected_T_avg = jnp.array([[0.32087105]])

    onp.testing.assert_allclose(R_avg, expected_R_avg)
    onp.testing.assert_allclose(T_avg, expected_T_avg)
