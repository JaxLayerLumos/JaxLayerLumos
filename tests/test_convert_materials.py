import jax.numpy as jnp
import numpy as np

from jaxlayerlumos import utils_materials


def verify_values(frequencies, values):
    unique_frequencies, indices = jnp.unique(frequencies, return_index=True)
    unique_values = values[indices]

    print(frequencies[0], frequencies[-1])
    print(unique_frequencies[0], unique_frequencies[-1])
    print(values[0], values[-1])
    print(unique_values[0], unique_values[-1])

    np.testing.assert_allclose(unique_frequencies, frequencies)
    np.testing.assert_allclose(unique_values, values)

    indices = jnp.argsort(frequencies)
    sorted_frequencies = frequencies[indices]
    sorted_values = values[indices]

    print(frequencies[0], frequencies[-1])
    print(sorted_frequencies[0], sorted_frequencies[-1])
    print(values[0], values[-1])
    print(sorted_values[0], sorted_values[-1])

    np.testing.assert_allclose(sorted_frequencies, frequencies)
    np.testing.assert_allclose(sorted_values, values)


def test_material_values():
    all_materials = utils_materials.get_all_materials()

    for material in all_materials:
        n_material, k_material = utils_materials.load_material(material)

        print(f"{material} n")
        frequencies_n, values_n = n_material.T
        verify_values(frequencies_n, values_n)

        print(f"{material} k")
        frequencies_k, values_k = k_material.T
        verify_values(frequencies_k, values_k)
