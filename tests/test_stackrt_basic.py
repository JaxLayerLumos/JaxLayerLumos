import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos import stackrt
from jaxlayerlumos import utils_materials
# from jaxlayerlumos.utils_materials import interpolate_multiple_materials_n_k

# from jaxlayerlumos.jaxlayerlumos_old import stackrt

# from jaxlayerlumos.jaxlayerlumos_old2 import stackrt_n_k as stackrt



def test_stackrt():
    wavelengths = jnp.linspace(300e-9, 900e-9, 3)
    frequencies = scic.c / wavelengths

    # n_Ag, k_Ag = utils_materials.interpolate_material_n_k("Ag", frequencies)
    # n_k_Ag = n_Ag + 1j * k_Ag
    #
    # n_air = jnp.ones_like(frequencies)
    # d_air = jnp.array([0])
    # d_Ag = jnp.array([2e-6])
    #
    # n_stack = jnp.vstack([n_air, n_k_Ag, n_air]).T
    # d_stack = jnp.hstack([d_air, d_Ag, d_air]).squeeze()
    #
    # R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, 0.0)


    materials = ['Air', 'Ag', 'Air']
    thickness_materials = [0, 2e-6, 0] # in m

    thicknesses = jnp.array(thickness_materials)
    n_k = utils_materials.interpolate_multiple_materials_n_k(materials, frequencies)

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies, 0.0, materials)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    expected_R_avg = np.array(
        [
            [
                0.09525774381746192,
                0.9601123170389492,
                0.9772199561956645,
            ],
        ]
    )
    expected_T_avg = np.array(
        [
            [
                7.770517514983977e-25,
                1.760970772881048e-65,
                1.4500794728372322e-70,
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
