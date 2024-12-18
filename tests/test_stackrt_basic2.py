import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos import stackrt
# from jaxlayerlumos.jaxlayerlumos_old import stackrt
# from jaxlayerlumos.jaxlayerlumos_old2 import stackrt_n_k as stackrt
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units

def test_stackrt():
    wavelengths = jnp.linspace(300e-9, 900e-9, 3)
    #wavelengths = jnp.array([300e-9])
    frequencies = utils_units.get_light_speed() / wavelengths

    materials = np.array(['Air', 'FusedSilica', 'Si3N4'])
    thickness_materials = [0, 2.91937911, 0]

    n_k = utils_materials.get_n_k(materials, frequencies)
#    n_k = utils_materials.interpolate_multiple_materials_n_k(materials, frequencies)

    thicknesses = jnp.array(thickness_materials)

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies, 0.0, materials)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    expected_R_avg = np.array(
        [
            [
                0.00013699554464010507,
                0.09585152519397383,
                0.06717789506951467
            ],
        ]
    )
    expected_T_avg = np.array(
        [
            [
                0.9998630044553601,
                0.9041484748060266,
                0.9328221049304859
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

    np.testing.assert_allclose(R_avg, expected_R_avg, rtol=1e-6)
    np.testing.assert_allclose(T_avg, expected_T_avg)
