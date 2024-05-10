import jax.numpy as jnp
import numpy as np

from jaxlayerlumos import utils_materials


def verify_values(frequencies, values):
    unique_frequencies, indices = jnp.unique(frequencies, return_index=True)
    unique_values = values[indices]

    np.testing.assert_allclose(unique_frequencies, frequencies)
    np.testing.assert_allclose(unique_values, values)

    indices = jnp.argsort(frequencies)
    sorted_frequencies = frequencies[indices]
    sorted_values = values[indices]

    np.testing.assert_allclose(sorted_frequencies, frequencies)
    np.testing.assert_allclose(sorted_values, values)


def test_material_values():
    all_materials = utils_materials.get_all_materials()

    for material in all_materials:
        n_material, k_material = utils_materials.load_material(material)

        frequencies_n, values_n = n_material.T
        verify_values(frequencies_n, values_n)

        frequencies_k, values_k = k_material.T
        verify_values(frequencies_k, values_k)
