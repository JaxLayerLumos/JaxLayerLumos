import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import stackrt_eps_mu
from jaxlayerlumos import utils_materials


def test_stackrt_radar_1():
    frequencies = jnp.linspace(0.1e9, 1e9, 3)
    # frequencies = jnp.array([1e9])
    materials = onp.array(["Air", "11", "16", "7", "4", "4", "PEC"])
    eps_stack, mu_stack = utils_materials.get_eps_mu(materials, frequencies)

    d_stack = jnp.array([0, 0.7742, 0.8485, 1.4878, 1.9883, 1.9863, 0]) * 1e-3
    R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(
        eps_stack, mu_stack, d_stack, frequencies, 0.0
    )

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    R_db = 10 * jnp.log10(R_avg).squeeze()
    T_db = 10 * jnp.log10(T_avg).squeeze()

    expected_R_db = onp.array(
        [
            -33.872113339369314,
            -39.141066380663602,
            -38.794125046637177,
        ]
    )

    expected_T_db = onp.array(
        [
            -onp.inf,
            -onp.inf,
            -onp.inf,
        ]
    )

    print("R_db")
    for elem in R_db:
        print(elem)

    print("T_db")
    for elem in T_db:
        print(elem)

    onp.testing.assert_allclose(R_db, expected_R_db)
    onp.testing.assert_allclose(T_db, expected_T_db)


def test_stackrt_radar_2():
    frequencies = jnp.linspace(0.1e9, 1e9, 5)

    materials = onp.array(["Air", "15", "12", "9", "PEC"])
    eps_stack, mu_stack = utils_materials.get_eps_mu(materials, frequencies)

    d_stack = jnp.array([0, 0.43, 0.21, 0.05, 0]) * 1e-3
    R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(
        eps_stack, mu_stack, d_stack, frequencies, 0.0
    )

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    R_db = 10 * jnp.log10(R_avg).squeeze()
    T_db = 10 * jnp.log10(T_avg).squeeze()

    expected_R_db = onp.array(
        [
            -0.057683530802195224,
            -0.5144977473090127,
            -1.1825636270341442,
            -1.9171267047938356,
            -2.673613156694648,
        ]
    )

    expected_T_db = onp.array(
        [
            -onp.inf,
            -onp.inf,
            -onp.inf,
            -onp.inf,
            -onp.inf,
        ]
    )

    print("R_db")
    for elem in R_db:
        print(elem)

    print("T_db")
    for elem in T_db:
        print(elem)

    onp.testing.assert_allclose(R_db, expected_R_db)
    onp.testing.assert_allclose(T_db, expected_T_db)
