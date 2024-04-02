import unittest
import jax.numpy as jnp
import scipy.constants as scic
import numpy as np
from jaxlayerlumos.utils_materials import load_material, interpolate_material
from jaxlayerlumos.jaxlayerlumos import stackrt, stackrt_theta  # Assuming stackrt can handle the case previously covered by stackrt0

class TestJaxLayerLumos(unittest.TestCase):
    def test_stackrt(self):
        # Load material data for SiO2 (example previously used 'Ag', ensure you replace with 'SiO2' or the correct material if available)
        # This assumes 'load_material' and 'interpolate_material' are adapted for JAX
        Ag_data = load_material('Ag')  # Make sure your data includes SiO2 or adjust accordingly

        # Define a small wavelength range for testing
        wavelengths = jnp.linspace(300e-9, 900e-9, 3)  # from 300nm to 900nm
        frequencies = scic.c / wavelengths  # Convert wavelengths to frequencies

        # Interpolate n and k values for SiO2 over the specified frequency range
        n_k_Ag = interpolate_material(Ag_data, frequencies)
        n_Ag = n_k_Ag[:, 0] + 1j * n_k_Ag[:, 1]

        # Stack configuration
        n_air = jnp.ones_like(frequencies)
        d_air = jnp.array([0])
        d_Ag = jnp.array([2e-6])

        n_stack = jnp.vstack([n_air, n_Ag, n_air]).T
        d_stack = jnp.hstack([d_air, d_Ag, d_air]).squeeze()

        # Assuming stackrt can now handle theta=0 as a special case, eliminating the need for stackrt0
        R_TE, T_TE, R_TM, T_TM = stackrt_theta(n_stack, d_stack, frequencies)

        # Calculate average R and T
        R_avg = (R_TE + R_TM) / 2
        T_avg = (T_TE + T_TM) / 2
        # print(R_avg)
        # print(T_avg)
        # Expected output needs to be defined based on your new calculations and expectations
        # This part remains as an exercise since the actual expected values depend on your implementation and material data
        expected_R_avg = np.array([0.15756072, 0.98613162, 0.99031732])
        expected_T_avg = np.array([5.87146690e-34, 1.58748575e-71, 5.13012036e-76])

        # Validate the results with a tolerance for floating-point arithmetic
        np.testing.assert_allclose(R_avg, expected_R_avg, rtol=1e-2, atol=0)
        np.testing.assert_allclose(T_avg, expected_T_avg, rtol=1e-2, atol=0)

if __name__ == '__main__':
    unittest.main()
