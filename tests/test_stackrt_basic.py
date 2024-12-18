import jax.numpy as jnp
import numpy as onp
import scipy.constants as scic

from jaxlayerlumos import stackrt
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units


def test_stackrt_1():
    wavelengths = jnp.linspace(300e-9, 900e-9, 3)
    frequencies = scic.c / wavelengths

    materials = onp.array(['Air', 'Ag', 'Air'])
    thickness_materials = [0, 2e-6, 0]

    thicknesses = jnp.array(thickness_materials)
    n_k = utils_materials.get_n_k(materials, frequencies)

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies, 0.0, materials)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    expected_R_avg = onp.array(
        [
            [
                0.09520024993673426,
                0.9601030713794845,
                0.9772141391075291,
            ],
        ]
    )
    expected_T_avg = onp.array(
        [
            [
                7.77150513637392e-25,
                1.7617872265745188e-65,
                1.4508201489462378e-70,
            ],
        ]
    )

    print("R_avg")
    for elem in R_avg:
        for elem2 in elem:
            print(elem2)
    print("T_avg")
    for elem in T_avg:
        for elem2 in elem:
            print(elem2)

    onp.testing.assert_allclose(R_avg, expected_R_avg)
    onp.testing.assert_allclose(T_avg, expected_T_avg)


def test_stackrt_2():
    wavelengths = jnp.linspace(300e-9, 900e-9, 3)
    frequencies = utils_units.get_light_speed() / wavelengths

    materials = onp.array(['Air', 'FusedSilica', 'Si3N4'])
    thickness_materials = [0, 2.91937911, 0]

    n_k = utils_materials.get_n_k(materials, frequencies)

    thicknesses = jnp.array(thickness_materials)

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies, 0.0, materials)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    expected_R_avg = onp.array(
        [
            [
                0.00013388240774118648,
                0.0957792837472504,
                0.0671234102574696,
            ],
        ]
    )
    expected_T_avg = onp.array(
        [
            [
                0.9998661175922586,
                0.90422071625275,
                0.9328765897425305,
            ],
        ]
    )

    print("R_avg")
    for elem in R_avg:
        for elem2 in elem:
            print(elem2)
    print("T_avg")
    for elem in T_avg:
        for elem2 in elem:
            print(elem2)

    onp.testing.assert_allclose(R_avg, expected_R_avg)
    onp.testing.assert_allclose(T_avg, expected_T_avg)
