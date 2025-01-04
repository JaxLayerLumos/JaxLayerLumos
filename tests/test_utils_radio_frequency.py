import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import stackrt
from jaxlayerlumos import utils_radio_frequency


def test_stackrt_radio_frequency():
    frequencies = jnp.linspace(8e9, 18e9, 100)
    materials = ["Air", "Ag", "Air"]

    n_k_ag = utils_radio_frequency.load_material_RF("Ag", frequencies)
    n_ag = (
        n_k_ag[:, 1] + 1j * n_k_ag[:, 2]
    )  # Combine n and k into a complex refractive index

    n_air = jnp.ones_like(frequencies)  # Refractive index of air
    d_air = jnp.array([0])  # Air layer thickness
    d_ag = jnp.array([5e-8])  # Ag layer thickness

    n_stack = jnp.vstack([n_air, n_ag, n_air]).T  # Transpose to match expected shape
    d_stack = jnp.hstack([d_air, d_ag, d_air])  # Stack thickness

    R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, jnp.array([0]))

    SE_TE = -10 * jnp.log10(T_TE)
    SE_TM = -10 * jnp.log10(T_TM)
    SE = (SE_TE + SE_TM) / 2

    expected_mean_SE = 55.47150237872992

    onp.testing.assert_allclose(jnp.mean(SE), expected_mean_SE)
