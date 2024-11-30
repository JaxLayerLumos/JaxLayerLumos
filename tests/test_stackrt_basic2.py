import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos import stackrt

# from jaxlayerlumos.jaxlayerlumos_old2 import stackrt_n_k as stackrt
from jaxlayerlumos import utils_materials


def test_stackrt():
    wavelengths = jnp.linspace(300e-9, 900e-9, 3)
    frequencies = scic.c / wavelengths

    materials = ['FusedSilica', 'Si3N4']
    thickness_materials = [2.91937911, 6.12241842]

    n_k_air = jnp.ones_like(frequencies)
    thickness_air = 0.0

    n_k = [n_k_air]
    thicknesses = [thickness_air]
    thicknesses.extend(thickness_materials)

    for material in materials:
        n_material, k_material = utils_materials.interpolate_material_n_k(
            material, frequencies
        )
        n_k_material = n_material + 1j * k_material

        n_k.append(n_k_material)

    n_k = jnp.array(n_k).T
    thicknesses = jnp.array(thicknesses)

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies, 0.0)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    expected_R_avg = np.array(
        [
            [
                0.000137,
                0.095852,
                0.067178
            ],
        ]
    )
    expected_T_avg = np.array(
        [
            [
                0.999744,
                0.904846,
                0.93344
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

    np.testing.assert_allclose(R_avg, expected_R_avg)
#    np.testing.assert_allclose(T_avg, expected_T_avg)
