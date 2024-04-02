import jax.numpy as jnp
import numpy as np
from jaxlayerlumos.utils_materials import load_material_RF
from jaxlayerlumos.jaxlayerlumos import stackrt  # Assuming stackrt includes the functionality of stackrt0

def test_stackrt_RF():
    # Define frequency range
    frequencies = jnp.linspace(8e9, 18e9, 100)  # Frequency range from 8GHz to 18GHz

    # Load material data for 'Ag' over the specified frequency range
    n_k_ag = load_material_RF('Ag', frequencies)
    n_ag = n_k_ag[:, 1] + 1j*n_k_ag[:, 2]  # Combine n and k into a complex refractive index

    # Define stack configuration
    n_air = jnp.ones_like(frequencies)  # Refractive index of air
    d_air = jnp.array([0])  # Air layer thickness
    d_ag = jnp.array([5e-8])  # Ag layer thickness

    # Stack refractive indices and thicknesses for air-Ag-air
    n_stack = jnp.vstack([n_air, n_ag, n_air]).T  # Transpose to match expected shape
    d_stack = jnp.hstack([d_air, d_ag, d_air])  # Stack thickness

    # Calculate R and T over the frequency range
    R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, theta=jnp.array([0]))

    # Calculate average shielding effectiveness
    SE_TE = -10 * jnp.log10(T_TE)
    SE_TM = -10 * jnp.log10(T_TM)
    SE = (SE_TE + SE_TM) / 2

    expected_mean_SE = 55.47150237872992
    # Assert that the actual mean SE is close to the expected value
    np.testing.assert_allclose(jnp.mean(SE), expected_mean_SE, rtol=1e-5)
