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
                0.4768954231115574,
                0.2916708090357754,
                0.2001408141358968,
                0.14606583520597474,
                0.11124900565130748,
            ],
            [
                0.45569211601895465,
                0.29094419187661313,
                0.20671836742576427,
                0.15466186661258008,
                0.11989784671887059,
            ],
            [
                0.9903227871042097,
                0.9857668752176341,
                0.9797337186490993,
                0.972674261445052,
                0.9647015513636168,
            ],
        ]
    )

    expected_T_avg = jnp.array(
        [
            [
                0.520080710529681,
                0.708327490349119,
                0.7998591742644134,
                0.8539341645203364,
                0.8887509943486925,
            ],
            [
                0.5412420637055398,
                0.7090540935362966,
                0.7932816207760296,
                0.8453381331064422,
                0.8801021532811291,
            ],
            [
                0.009477949565895032,
                0.014232993943880392,
                0.02026628018852014,
                0.02732573851949386,
                0.03529844863638335,
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

    expected_R_avg = jnp.array([[0.24865506714761443 + 0.048242539134388605]]) / 2
    expected_T_avg = jnp.array([[0.7513449328523855 + 0.9517574608656109]]) / 2

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

    expected_R_avg = jnp.array([[0.5619582074445789 + 0.40941209817941343]]) / 2
    expected_T_avg = jnp.array([[0.2843212501139465 + 0.3574826780072898]]) / 2

    onp.testing.assert_allclose(R_avg, expected_R_avg)
    onp.testing.assert_allclose(T_avg, expected_T_avg)
