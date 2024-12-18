import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import stackrt_eps_mu
from jaxlayerlumos import utils_materials


def test_stackrt_radar():
    frequencies = jnp.linspace(0.1e9, 1e9, 3)

    materials = onp.array(["Air", "11", "16", "7", "4", "4", "PEC"])
    eps_stack, mu_stack = utils_materials.get_eps_mu(materials, frequencies)

    d_stack = jnp.array([0, 0.7742, 0.8485, 1.4878, 1.9883, 1.9863, 0]) * 1e-3
    R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(
        eps_stack, mu_stack, d_stack, frequencies, 0.0, materials
    )

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    R_db = 10 * jnp.log10(R_avg).squeeze()

    expected_R_db = onp.array(
        [
            -33.872113339369314,
            -39.141066380663602,
            -38.794125046637177,
        ]
    )

    print("R_db")
    for elem in R_db:
        print(elem)

    print("T_avg")
    for elem in T_avg:
        for elem2 in elem:
            print(elem2)

    onp.testing.assert_allclose(R_db, expected_R_db)
