import jax.numpy as jnp
import csv

from jaxlayerlumos import utils_materials


def convert_values(frequencies, values):
    unique_frequencies, indices = jnp.unique(frequencies, return_index=True)
    unique_values = values[indices]

    indices = jnp.argsort(unique_frequencies)
    sorted_frequencies = unique_frequencies[indices]
    sorted_values = unique_values[indices]

    return sorted_frequencies, sorted_values

def convert_material(material):
    n_material, k_material = utils_materials.load_material(material)

    frequencies_n, values_n = n_material.T
    frequencies_k, values_k = k_material.T

    n_material_new = jnp.concatenate([frequencies_n[..., jnp.newaxis], values_n[..., jnp.newaxis]], axis=1)
    k_material_new = jnp.concatenate([frequencies_k[..., jnp.newaxis], values_k[..., jnp.newaxis]], axis=1)

    return n_material_new, k_material_new

def save_material(str_path, n_material, k_material):
    writer = csv.writer(open(str_path, 'w'), delimiter = ",")
    writer.writerow(['wl', 'n', 'k'])

    for ind in range(0, n_material.shape[0]):
        writer.writerow(list(n_material[ind]) + [k_material[ind, 1]])


if __name__ == '__main__':
    all_materials = utils_materials.get_all_materials()
    all_materials = ['Ag']
    material_indices, str_directory = utils_materials.load_material_json()

    for material in all_materials:
        str_path_material = material_indices.get(material)
        str_path = str_directory / str_path_material

        n_material, k_material = convert_material(material)
        save_material(str_path, n_material, k_material)
