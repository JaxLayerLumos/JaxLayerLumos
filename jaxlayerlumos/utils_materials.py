import jax.numpy as jnp
import csv
import json
from pathlib import Path
from .utils_spectra import (
    convert_wavelengths_to_frequencies,
)  # Ensure this function exists

# from scipy.interpolate import interp1d  # Note: Consider JAX-compatible interpolation if needed

# JAX doesn't have direct equivalents for all SciPy functions, so you might still need SciPy for some tasks,
# especially those not directly related to numerical arrays, such as file I/O or interpolation.
# For purely numerical tasks, seek JAX or JAX-compatible libraries.

from .utils_spectra import convert_frequencies_to_wavelengths

Metals_sigma = {
    "Cu": 5.96e7,
    "Cr": 7.74e6,
    "Ag": 6.3e7,
    "Al": 3.77e7,
    "Ni": 1.43e7,
    "W": 1.79e7,
    "Ti": 2.38e6,
    "Pd": 9.52e6,
}

Metals_nk_updated_specific_sigma = {}
mu0 = 4 * jnp.pi * 1e-7  # H/m
Z_0 = 377  # Ohms, impedance of free space
# Frequency range
nu = jnp.linspace(8e9, 18e9, 11)  # From 8 GHz to 18 GHz
omega = 2 * jnp.pi * nu  # Angular frequency

for metal, sigma in Metals_sigma.items():
    Z = jnp.sqrt(
        1j * omega * mu0 / sigma
    )  # Impedance of the material using specific sigma
    n_complex = Z_0 / Z  # Complex refractive index

    # Extract real and imaginary parts of the refractive index
    n_real = jnp.real(n_complex)
    k_imag = jnp.imag(n_complex)

    # Update the dictionary with the new values
    Metals_nk_updated_specific_sigma[metal] = {
        "freq_data": nu.tolist(),  # Converting JAX arrays to lists for JSON serialization
        "n_data": n_real.tolist(),
        "k_data": k_imag.tolist(),
    }


def load_material_RF(material_name, frequencies):
    """
    Load material RF data for a given material and frequencies. Adapted for JAX.

    Parameters:
    - material_name: The name of the material to load.
    - frequencies: Array of frequencies for which data is requested.

    Returns:
    - A JAX array with columns for frequency, n, and k.
    """

    frequencies = jnp.array(frequencies)  # Ensure input frequencies are JAX arrays

    if material_name not in Metals_nk_updated_specific_sigma:
        n_default = jnp.ones_like(frequencies)
        k_default = jnp.zeros_like(frequencies)
        data = jnp.column_stack((frequencies, n_default, k_default))
    else:
        material_data = Metals_nk_updated_specific_sigma[material_name]
        freq_data = jnp.array(material_data["freq_data"])
        n_data = jnp.array(material_data["n_data"])
        k_data = jnp.array(material_data["k_data"])

        n_interpolated = jnp.interp(
            frequencies, freq_data, n_data, left="extrapolate", right="extrapolate"
        )
        k_interpolated = jnp.interp(
            frequencies, freq_data, k_data, left="extrapolate", right="extrapolate"
        )

        data = jnp.column_stack((frequencies, n_interpolated, k_interpolated))

    return data


def load_material(material_name):
    """
    Load material data from its CSV file, converting wavelength to frequency. Adapted for JAX.

    Parameters:
    - material_name: The name of the material to load.

    Returns:
    - A JAX array with columns for frequency (converted from wavelength), n, and k.
    """
    current_dir = Path(__file__).parent
    materials_file = current_dir / "materials.json"

    with open(materials_file, "r") as file:
        material_index = json.load(file)

    relative_file_path = material_index.get(material_name)
    if not relative_file_path:
        raise ValueError(f"Material {material_name} not found in the index.")

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
    data = data[data[:, 0].argsort()]  # Ensure sorted by frequency

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
