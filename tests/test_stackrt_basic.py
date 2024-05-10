import unittest
import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos.jaxlayerlumos import stackrt_theta
from jaxlayerlumos.utils_materials import load_material, interpolate_material


class TestJaxLayerLumos(unittest.TestCase):
    def test_stackrt(self):
        # Load material data for SiO2 (example previously used 'Ag', ensure you replace with 'SiO2' or the correct material if available)
        # This assumes 'load_material' and 'interpolate_material' are adapted for JAX
        Ag_data = load_material(
            "Ag"
        )  # Make sure your data includes SiO2 or adjust accordingly

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

        R_TE, T_TE, R_TM, T_TM = stackrt_theta(n_stack, d_stack, frequencies, 0.0)

        R_avg = (R_TE + R_TM) / 2
        T_avg = (T_TE + T_TM) / 2

        expected_R_avg = np.array(
            [0.09525774381746192, 0.9601123170389492, 0.9772199561956645,]
        )
        expected_T_avg = np.array(
            [7.770517514983977e-25, 1.760970772881048e-65, 1.4500794728372322e-70,]
        )

        print("R_avg")
        for elem in R_avg:
            print(elem)
        print("T_avg")
        for elem in T_avg:
            print(elem)

        np.testing.assert_allclose(R_avg, expected_R_avg)
        np.testing.assert_allclose(T_avg, expected_T_avg)


if __name__ == "__main__":
    unittest.main()
