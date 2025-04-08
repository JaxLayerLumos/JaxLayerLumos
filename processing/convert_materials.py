import jax.numpy as jnp
import csv
import os

from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_spectra


def convert_values(wavelenths, values):
    unique_wavelenths, indices = jnp.unique(wavelenths, return_index=True)
    unique_values = values[indices]

    indices = jnp.argsort(unique_wavelenths, descending=True)
    sorted_wavelenths = unique_wavelenths[indices]
    sorted_values = unique_values[indices]

    return sorted_wavelenths, sorted_values


def convert_material(material):
    n_material, k_material = utils_materials.load_material_wavelength_um(material)

    wavelengths_um_n, values_n = n_material.T
    wavelengths_um_k, values_k = k_material.T

    wavelengths_um_n, values_n = convert_values(wavelengths_um_n, values_n)
    wavelengths_um_k, values_k = convert_values(wavelengths_um_k, values_k)

    n_material_new = jnp.concatenate(
        [wavelengths_um_n[..., jnp.newaxis], values_n[..., jnp.newaxis]], axis=1
    )
    k_material_new = jnp.concatenate(
        [wavelengths_um_k[..., jnp.newaxis], values_k[..., jnp.newaxis]], axis=1
    )

    return n_material_new, k_material_new


def save_material(str_path, n_material, k_material):
    writer = csv.writer(open(str_path, "w"), delimiter=",")

    writer.writerow(["wl", "n"])
    for row in n_material:
        writer.writerow(row)

    writer.writerow([])
    writer.writerow(["wl", "k"])
    for row in k_material:
        writer.writerow(row)


if __name__ == "__main__":
    all_materials = utils_materials.get_all_materials()
    material_indices, str_directory = utils_materials.load_json()

    for material in all_materials:
        str_path_material = material_indices.get(material)
        str_path = os.path.join(str_directory, str_path_material)

        n_material, k_material = convert_material(material)
        save_material(str_path, n_material, k_material)

        print(material)
        print(n_material.shape, k_material.shape)
        print("")
