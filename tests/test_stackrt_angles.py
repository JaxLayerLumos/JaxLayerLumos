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
    materials = onp.array(["Air", "TiO2", "Air"])
    n_stack = utils_materials.get_n_k(materials, frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(jnp.array([2e-8]))
    thetas = jnp.linspace(0, 89, num_angles)

    R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, thetas, materials)

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

    materials = onp.array(["Air", "TiO2", "Air"])
    d_stack = jnp.array([0, 2e-8, 0])
    thetas = jnp.linspace(0, 89, 3)

    n_k = []

    for material in materials:
        n_material, k_material = utils_materials.interpolate_material_n_k(
            material, frequencies
        )
        n_k_material = n_material + 1j * k_material

        n_k.append(n_k_material)

    n_k = jnp.array(n_k).T

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, d_stack, frequencies, thetas, materials)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    expected_R_avg = jnp.array(
        [
            [
                0.4770790218584376,
                0.2918301244526286,
                0.20026632179451592,
                0.1461644821589118,
                0.11132762140037072,
            ],
            [
                0.45586172063057634,
                0.29109418329257924,
                0.20684237462156152,
                0.15476340159980753,
                0.11998132563573674,
            ],
            [
                0.9903190571800176,
                0.9857622632656107,
                0.97972783328248,
                0.9726668981700777,
                0.9646925772238668,
            ],
        ]
    )

    expected_T_avg = jnp.array(
        [
            [
                0.5198973861437662,
                0.7081681748619179,
                0.7997336666045015,
                0.8538355175673582,
                0.8886723785996298,
            ],
            [
                0.5410727876996988,
                0.7089041021153875,
                0.7931576135794325,
                0.8452365981191826,
                0.8800186743642632,
            ],
            [
                0.009481699666492172,
                0.01423760593507062,
                0.02027216555558674,
                0.027333101794482728,
                0.03530742277613292,
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

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies, theta, materials)

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

    expected_R_avg = jnp.array([[0.24877806962502294 + 0.04829532237208759]]) / 2
    expected_T_avg = jnp.array([[0.751221930374977 + 0.9517046776279124]]) / 2

    onp.testing.assert_allclose(R_avg, expected_R_avg)
    onp.testing.assert_allclose(T_avg, expected_T_avg)


def test_angles_3():
    wavelengths = jnp.array([300e-9])
    frequencies = utils_units.get_light_speed() / wavelengths

    materials = onp.array(["Air", "Ag", "Cr"])
    thickness_materials = [0, 9.36793259, 0]
    theta = 34.767507632418315

    thicknesses = jnp.array(thickness_materials)
    n_k = utils_materials.get_n_k(materials, frequencies)

    thicknesses *= utils_units.get_nano()

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies, theta, materials)

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

    expected_R_avg = jnp.array([[0.5620107127607985 + 0.4094868270876877]]) / 2
    expected_T_avg = jnp.array([[0.2842889562588487 + 0.357453134709888]]) / 2

    onp.testing.assert_allclose(R_avg, expected_R_avg)
    onp.testing.assert_allclose(T_avg, expected_T_avg)
