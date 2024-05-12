import unittest
import jax.numpy as jnp
import jax
import numpy as np
import scipy.constants as scic

from jaxlayerlumos.utils_materials import load_material, interpolate_material
from jaxlayerlumos.jaxlayerlumos import stackrt_theta


class TestJaxLayerLumos(unittest.TestCase):
    def test_gradient_stackrt_theta_thickness(self):
        data_n, data_k = load_material("Ag")

        # Define a small wavelength range for testing
        wavelengths = jnp.linspace(300e-9, 900e-9, 3)  # from 300nm to 900nm
        frequencies = scic.c / wavelengths  # Convert wavelengths to frequencies

        n_Ag = interpolate_material(data_n, frequencies)
        k_Ag = interpolate_material(data_k, frequencies)
        n_k_Ag = n_Ag + 1j * k_Ag

        n_air = jnp.ones_like(frequencies)
        d_air = jnp.array([0])
        d_Ag = jnp.array([2e-6])

        n_stack = jnp.vstack([n_air, n_k_Ag, n_air]).T
        d_stack = jnp.hstack([d_air, d_Ag, d_air]).squeeze()

        # Function to compute the first element of R_TE given the thickness
        def compute_R_TE_first_element(d_stack):
            R_TE, _, _, _ = stackrt_theta(n_stack, d_stack, frequencies, 0.0)
            return R_TE[0]  # Focusing on the first element for simplification

        # Compute the gradient of R_TE with respect to the layer's thickness
        grad_R_TE = jax.grad(compute_R_TE_first_element)(d_stack)
        # print(grad_R_TE)
        # Asserts to ensure the gradient computation is successful and results are sensible
        assert grad_R_TE is not None, "Gradient computation failed, returned None."
        assert isinstance(grad_R_TE, jnp.ndarray), "Gradient should be a JAX ndarray."
        assert (
            grad_R_TE.shape == d_stack.shape
        ), "Gradient shape mismatch with the input thickness shape."

        expected_grad_R_TE = jnp.array(
            [
                0.0,
                -1.8639916450783514e-10,
                4.2091929253306585e-10,
            ]
        )

        for elem in grad_R_TE:
            print(elem)

        np.testing.assert_allclose(
            grad_R_TE,
            expected_grad_R_TE,
            err_msg="Computed gradient does not match expected values.",
        )

    # Note: Since we're not comparing against a specific expected value here,
    # the asserts mainly ensure that the computation does not error out and returns
    # results of the correct type and shape. For more rigorous testing, consider
    # adding comparisons against known values or behaviors under specific scenarios.


if __name__ == "__main__":
    unittest.main()
