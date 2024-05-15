import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos import stackrt
from jaxlayerlumos import utils_materials


def test_stackrt():
    wavelengths = jnp.linspace(300e-9, 900e-9, 3)
    frequencies = scic.c / wavelengths

    n_Ag, k_Ag = utils_materials.interpolate_material("Ag", frequencies)
    n_k_Ag = n_Ag + 1j * k_Ag

    n_air = jnp.ones_like(frequencies)
    d_air = jnp.array([0])
    d_Ag = jnp.array([2e-6])

    n_stack = jnp.vstack([n_air, n_k_Ag, n_air]).T
    d_stack = jnp.hstack([d_air, d_Ag, d_air]).squeeze()

    R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, 0.0)

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
        print(elem)
    print("T_avg")
    for elem in T_avg:
        print(elem)

    np.testing.assert_allclose(R_avg, expected_R_avg)
    np.testing.assert_allclose(T_avg, expected_T_avg)
