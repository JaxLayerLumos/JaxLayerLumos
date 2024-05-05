import jax.numpy as jnp
import csv
import json
from pathlib import Path

from .utils_spectra import convert_wavelengths_to_frequencies


def load_material_json():
    current_dir = Path(__file__).parent
    materials_file = current_dir / "materials.json"

    with open(materials_file, "r") as file_json:
        material_indices = json.load(file_json)

    return material_indices, current_dir


def get_all_materials():
    material_indices, _ = load_material_json()
    return list(material_indices.keys())


def load_material(material_name):
    """
    Load material data from its CSV file, converting wavelength to frequency. Adapted for JAX.

    Parameters:
    - material_name: The name of the material to load.

    Returns:
    - A JAX array with columns for frequency (converted from wavelength), n, and k.

    """

    material_indices, current_dir = load_material_json()
    relative_file_path = material_indices.get(material_name)

    if not relative_file_path:
        raise ValueError(f"Material {material_name} not found in JaxLayerLumos.")

    csv_file_path = current_dir / relative_file_path
    data = []

    with open(csv_file_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            try:
                wavelength_um, n, k = map(float, row)
                frequency = convert_wavelengths_to_frequencies(
                    wavelength_um * 1e-6
                )  # Convert um to meters
                data.append((frequency, n, k))
            except ValueError:
                continue

    data = jnp.array(data)
    data = data[data[:, 0].argsort()]

    return data


def interpolate_material(material_data, frequencies):
    """
    Interpolate n and k values for the specified frequencies using JAX.
    Supports linear interpolation and extrapolation.

    Parameters:
    - material_data: The data for the material, as returned by load_material. Expected to be a JAX array.
    - frequencies: A list or JAX array of frequencies to interpolate n and k for.

    Returns:
    - Interpolated values of n and k as a JAX array.

    """

    # Extract frequency, n, and k columns
    freqs, n_values, k_values = material_data.T

    # Remove duplicates (if any) while preserving order
    unique_freqs, indices = jnp.unique(freqs, return_index=True)
    unique_n_values = n_values[indices]
    unique_k_values = k_values[indices]

    # Ensure frequencies are sorted (usually they should be, but just in case)
    sorted_indices = jnp.argsort(unique_freqs)
    sorted_freqs = unique_freqs[sorted_indices]
    sorted_n_values = unique_n_values[sorted_indices]
    sorted_k_values = unique_k_values[sorted_indices]

    # Interpolate n and k for the given frequencies using JAX's interp function
    n_interp_values = jnp.interp(
        frequencies,
        sorted_freqs,
        sorted_n_values,
        left="extrapolate",
        right="extrapolate",
    )

    k_interp_values = jnp.interp(
        frequencies,
        sorted_freqs,
        sorted_k_values,
        left="extrapolate",
        right="extrapolate",
    )

    return jnp.vstack((n_interp_values, k_interp_values)).T


def get_n_k_surrounded_by_air(materials, frequencies):
    assert isinstance(materials, list)
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    num_layers = len(materials) + 2
    num_frequencies = frequencies.shape[0]

    n_k = jnp.ones((num_layers, num_frequencies), dtype=jnp.complex128)

    for ind, material in enumerate(materials):
        data_material = load_material(material)
        n_k_material = interpolate_material(data_material, frequencies)

        n_k = n_k.at[ind + 1, :].set(n_k_material[:, 0] + 1j * n_k_material[:, 1])

    assert jnp.all(jnp.real(n_k[0]) == 1)
    assert jnp.all(jnp.imag(n_k[0]) == 0)
    assert jnp.all(jnp.real(n_k[-1]) == 1)
    assert jnp.all(jnp.imag(n_k[-1]) == 0)

    n_k = n_k.T
    return n_k
