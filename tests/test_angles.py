import unittest
import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos.utils_materials import load_material, interpolate_material
from jaxlayerlumos.jaxlayerlumos import stackrt


class TestJaxLayerLumosStackrt(unittest.TestCase):
    def test_stackrt_with_angles(self):
        # Load material data for TiO2
        TiO2_data = load_material("TiO2")

        # Define wavelength range (in meters)
        wavelengths = jnp.linspace(300e-9, 900e-9, 3)  # 3 points from 300nm to 900nm
        frequencies = scic.c / wavelengths  # Convert wavelengths to frequencies

        # Interpolate n and k values for TiO2 over the specified frequency range
        n_k_TiO2 = interpolate_material(TiO2_data, frequencies)
        n_TiO2 = (
            n_k_TiO2[:, 0] + 1j * n_k_TiO2[:, 1]
        )  # Combine n and k into a complex refractive index

        # Define stack configuration
        n_air = jnp.ones_like(wavelengths)  # Refractive index of air is approximately 1
        n_stack = jnp.vstack(
            [n_air, n_TiO2, n_air]
        ).T  # Transpose to match expected shape (Nlayers x Nfreq)
        d_stack = jnp.array([0, 2e-8, 0])  # Stack thickness for air-TiO2-air
        thetas = jnp.linspace(0, 89, 3)  # Incident angles from 0 to 89 degrees

        # Calculate R and T over the frequency range for different incident angles
        R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, thetas)

        # Calculate average R and T
        R_avg = (R_TE + R_TM) / 2
        T_avg = (T_TE + T_TM) / 2
        # print(R_avg)
        # print(T_avg)

        expected_R_avg = jnp.array(
            [
                [
                    0.5075084159634257,
                    0.18726655434023687,
                    0.08370762967721315,
                ],
                [
                    0.4877839691834523,
                    0.1945178061281731,
                    0.09157449689223793,
                ],
                [
                    0.9498492283810344,
                    0.9783936773043997,
                    0.9540558921256859,
                ],
            ]
        )

        expected_T_avg = jnp.array(
            [
                [
                    0.17839133609006402,
                    0.8127334404810548,
                    0.916292370322787,
                ],
                [
                    0.18801137946236623,
                    0.8054821885944445,
                    0.9084255031077624,
                ],
                [
                    0.006495561010086865,
                    0.021606322148813736,
                    0.04594410787425793,
                ],
            ]
        )

        print('R_avg')
        for elem_1 in R_avg:
            for elem_2 in elem_1:
                print(elem_2)

        print('T_avg')
        for elem_1 in T_avg:
            for elem_2 in elem_1:
                print(elem_2)

        np.testing.assert_allclose(R_avg, expected_R_avg)
        np.testing.assert_allclose(T_avg, expected_T_avg)


if __name__ == "__main__":
    unittest.main()
