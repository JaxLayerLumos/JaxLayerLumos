import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos import utils_materials
from jaxlayerlumos import stackrt_eps_mu

def test_stackrt_radar():
    frequencies = jnp.linspace(0.1e9, 1e9, 100) # in GHz
    material_stack = jnp.array([11, 16, 7, 4, 4])

    eps_stack, mu_stack = utils_materials.get_eps_mu_Michielssen(material_stack, frequencies)
    d_stack = jnp.array([1.9863, 1.9883, 1.4878, 0.8485, 0.7742]) * 1e-3

    eps_air = jnp.ones_like(frequencies)
    mu_air = jnp.ones_like(frequencies)
    d_air = jnp.array([0])

    eps_PEC = jnp.ones_like(frequencies) * (jnp.inf * 1j)
    mu_PEC = jnp.ones_like(frequencies)

    eps_stack = jnp.vstack([eps_air, eps_stack, eps_PEC]).T
    mu_stack = jnp.vstack([mu_air, mu_stack, mu_PEC]).T
    d_stack = jnp.hstack([d_air, d_stack, d_air]).squeeze()

    R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(eps_stack, mu_stack, d_stack, frequencies, 0.0)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    print("R_avg")
    for elem in R_avg:
        for elem2 in elem:
            print(elem2)
    print("T_avg")
    for elem in T_avg:
        for elem2 in elem:
            print(elem2)

    np.testing.assert_allclose(R_avg, expected_R_avg)
    np.testing.assert_allclose(T_avg, expected_T_avg)
