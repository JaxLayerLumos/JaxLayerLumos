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
    material_indices, str_directory = load_json()
    str_file = material_indices.get(material)

    if not str_file:
        raise ValueError(f"Material {material} not found in JaxLayerLumos.")

    str_csv = str_directory / str_file
    data_n = []
    data_k = []

    with open(str_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)

        start_n = False
        start_k = False

        for row in csvreader:
            if len(row) == 2:
                if row[0] == "wl" and row[1] == "n":
                    start_n = True
                    start_k = False
                elif row[0] == "wl" and row[1] == "k":
                    start_n = False
                    start_k = True
                else:
                    wavelength_um, value = map(float, row)

                    if start_n and not start_k:
                        data_n.append([wavelength_um, value])
                    elif not start_n and start_k:
                        data_k.append([wavelength_um, value])
                    else:
                        raise ValueError
            elif len(row) == 0:
                pass
            else:
                raise ValueError

    data_n = jnp.array(data_n)
    data_k = jnp.array(data_k)
    assert data_n.shape[0] > 0 or data_k.shape[0] > 0

    if data_n.shape[0] == 0:
        data_n = jnp.concatenate(
            [data_k[:, 0][..., jnp.newaxis], jnp.zeros((data_k.shape[0], 1))], axis=1
        )
    if data_k.shape[0] == 0:
        data_k = jnp.concatenate(
            [data_n[:, 0][..., jnp.newaxis], jnp.zeros((data_n.shape[0], 1))], axis=1
        )

    return data_n, data_k


def load_material_wavelength(material):
    data_n, data_k = load_material_wavelength_um(material)

    data_n = data_n.at[:, 0].set(data_n[:, 0] * 1e-6)
    data_k = data_k.at[:, 0].set(data_k[:, 0] * 1e-6)

    return data_n, data_k


def load_material(material):
    data_n, data_k = load_material_wavelength(material)

    data_n = data_n.at[:, 0].set(convert_wavelengths_to_frequencies(data_n[:, 0]))
    data_k = data_k.at[:, 0].set(convert_wavelengths_to_frequencies(data_k[:, 0]))

    return data_n, data_k


def interpolate(freqs_values, frequencies):
    assert isinstance(freqs_values, jnp.ndarray)
    assert isinstance(frequencies, jnp.ndarray)
    assert freqs_values.ndim == 2
    assert frequencies.ndim == 1

    freqs, values = freqs_values.T

    assert jnp.min(freqs) <= jnp.min(frequencies)
    assert jnp.max(frequencies) <= jnp.max(freqs)

    values_interpolated = jnp.interp(
        frequencies,
        freqs,
        values,
        left="extrapolate",
        right="extrapolate",
    )

    return values_interpolated


def interpolate_material(material, frequencies):
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    data_n, data_k = load_material(material)

    n_material = interpolate(data_n, frequencies)
    k_material = interpolate(data_k, frequencies)

    return n_material, k_material


def get_n_k_surrounded_by_air(materials, frequencies):
    assert isinstance(materials, list)
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    num_layers = len(materials) + 2
    num_frequencies = frequencies.shape[0]

    n_k = jnp.ones((num_layers, num_frequencies), dtype=jnp.complex128)

    for ind, material in enumerate(materials):
        n_material, k_material = interpolate_material(material, frequencies)
        n_k = n_k.at[ind + 1, :].set(n_material + 1j * k_material)

    assert jnp.all(jnp.real(n_k[0]) == 1)
    assert jnp.all(jnp.imag(n_k[0]) == 0)
    assert jnp.all(jnp.real(n_k[-1]) == 1)
    assert jnp.all(jnp.imag(n_k[-1]) == 0)

    n_k = n_k.T
    return n_k
