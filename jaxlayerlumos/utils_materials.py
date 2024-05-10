import jax.numpy as jnp
import csv
import json
from pathlib import Path

from jaxlayerlumos.utils_spectra import convert_wavelengths_to_frequencies


def load_json():
    current_dir = Path(__file__).parent
    materials_file = current_dir / "materials.json"

    with open(materials_file, "r") as file_json:
        material_indices = json.load(file_json)

    return material_indices, current_dir


def get_all_materials():
    material_indices, _ = load_json()
    return list(material_indices.keys())


def load_material_wavelength_um(material):
    material_indices, current_dir = load_json()
    str_file = material_indices.get(material)

    if not str_file:
        raise ValueError(f"Material {material} not found in JaxLayerLumos.")

    str_csv = current_dir / str_file
    data = []

    with open(str_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        for row in csvreader:
            try:
                wavelength_um, n, k = map(float, row)
                data.append((wavelength_um, n, k))
            except ValueError:
                continue

    data = jnp.array(data)
    data = data[data[:, 0].argsort()]

    data_n = data[:, [0, 1]]
    data_k = data[:, [0, 2]]

    return data_n, data_k


def load_material_wavelength(material):
    data_n, data_k = load_material_wavelength_um(material)

    data_n[:, 0] = data_n[:, 0] * 1e-6
    data_k[:, 0] = data_k[:, 0] * 1e-6

    return data_n, data_k

def load_material(material):
    data_n, data_k = load_material_wavelength(material)

    data_n[:, 0] = convert_wavelengths_to_frequencies(data_n[:, 0])
    data_k[:, 0] = convert_wavelengths_to_frequencies(data_k[:, 0])

    return data_n, data_k


def interpolate_material(material_info, frequencies):
    """
    Interpolate n and k values for the specified frequencies using JAX.
    Supports linear interpolation and extrapolation.

    Parameters:
    - material_data: The data for the material, as returned by load_material. Expected to be a JAX array.
    - frequencies: A list or JAX array of frequencies to interpolate n and k for.

    Returns:
    - Interpolated values of n and k as a JAX array.

    """

    assert isinstance(material_info, jnp.ndarray)
    assert isinstance(frequencies, jnp.ndarray)
    assert material_info.ndim == 2
    assert frequencies.ndim == 1

    freqs, values = material_info.T

    assert jnp.min(freqs) <= jnp.min(frequencies)
    assert jnp.max(frequencies) <= jnp.max(freqs)

    values_interpolated = jnp.interp(
        frequencies, freqs, values, left="extrapolate", right="extrapolate",
    )

    return values_interpolated


def get_n_k_surrounded_by_air(materials, frequencies):
    assert isinstance(materials, list)
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    num_layers = len(materials) + 2
    num_frequencies = frequencies.shape[0]

    n_k = jnp.ones((num_layers, num_frequencies), dtype=jnp.complex128)

    for ind, material in enumerate(materials):
        data_n, data_k = load_material(material)
        n_material = interpolate_material(data_n, frequencies)
        k_material = interpolate_material(data_k, frequencies)

        n_k = n_k.at[ind + 1, :].set(n_material + 1j * k_material)

    assert jnp.all(jnp.real(n_k[0]) == 1)
    assert jnp.all(jnp.imag(n_k[0]) == 0)
    assert jnp.all(jnp.real(n_k[-1]) == 1)
    assert jnp.all(jnp.imag(n_k[-1]) == 0)

    n_k = n_k.T
    return n_k
