"""
Material property utilities for optical calculations.

This module provides functions for loading, interpolating, and converting material
properties including refractive indices (n, k), permittivity (ε), and permeability (μ).
It supports both optical materials (from CSV files) and radar materials, with automatic
interpolation to desired frequency ranges.

The module handles:
- Loading material data from CSV files
- Interpolation of material properties to specific frequencies
- Conversion between refractive indices and permittivity/permeability
- Special materials like Air and PEC (Perfect Electric Conductor)
- Integration with radar material databases
"""

import jax.numpy as jnp
import numpy as onp
import os
import csv
import json
from pathlib import Path
import warnings


from jaxlayerlumos import utils_spectra
from jaxlayerlumos import utils_radar_materials


def load_json():
    """
    Load the materials index file that maps material names to CSV files.
    
    Returns:
        tuple: (material_indices, current_dir) where material_indices is a dictionary
               mapping material names to CSV filenames, and current_dir is the
               directory containing the materials data.
    """
    current_dir = str(Path(__file__).parent)
    materials_file = os.path.join(current_dir, "materials.json")

    with open(materials_file, "r") as file_json:
        material_indices = json.load(file_json)

    return material_indices, current_dir


def get_all_materials():
    """
    Get a list of all available material names.
    
    Returns:
        list: Names of all materials available in the database.
    """
    material_indices, _ = load_json()
    return list(material_indices.keys())


def load_material_wavelength_um(material):
    """
    Load refractive index data for a material with wavelengths in micrometers.
    
    This function reads the CSV file for the specified material and extracts
    the wavelength-dependent refractive index (n) and extinction coefficient (k).
    The data is returned with wavelengths in micrometers.
    
    Args:
        material (str): Name of the material to load.
    
    Returns:
        tuple: (data_n, data_k) where each is a jnp.ndarray with shape (n_points, 2)
               containing [wavelength_um, value] pairs.
    
    Raises:
        ValueError: If the material is not found or the CSV format is invalid.
    """
    material_indices, str_directory = load_json()
    str_file = material_indices.get(material)

    if not str_file:
        raise ValueError(f"Material {material} not found in JaxLayerLumos.")

    str_csv = os.path.join(str_directory, str_file)
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
    """
    Load refractive index data for a material with wavelengths in meters.
    
    This function loads material data and converts wavelengths from micrometers to meters.
    
    Args:
        material (str): Name of the material to load.
    
    Returns:
        tuple: (data_n, data_k) where each is a jnp.ndarray with shape (n_points, 2)
               containing [wavelength_m, value] pairs.
    """
    data_n, data_k = load_material_wavelength_um(material)

    data_n = data_n.at[:, 0].set(data_n[:, 0] * 1e-6)
    data_k = data_k.at[:, 0].set(data_k[:, 0] * 1e-6)

    return data_n, data_k


def load_material(material):
    """
    Load refractive index data for a material with frequencies in Hz.
    
    This function loads material data and converts wavelengths to frequencies.
    
    Args:
        material (str): Name of the material to load.
    
    Returns:
        tuple: (data_n, data_k) where each is a jnp.ndarray with shape (n_points, 2)
               containing [frequency_Hz, value] pairs.
    """
    data_n, data_k = load_material_wavelength(material)

    data_n = data_n.at[:, 0].set(
        utils_spectra.convert_wavelengths_to_frequencies(data_n[:, 0])
    )
    data_k = data_k.at[:, 0].set(
        utils_spectra.convert_wavelengths_to_frequencies(data_k[:, 0])
    )

    return data_n, data_k


def interpolate(freqs_values, frequencies):
    """
    Interpolate frequency-dependent values to target frequencies.
    
    This function performs linear interpolation of frequency-dependent data to
    the specified target frequencies. It includes bounds checking and warnings
    for extrapolation.
    
    Args:
        freqs_values (jnp.ndarray): Array with shape (n_points, 2) containing
                                   [frequency, value] pairs.
        frequencies (jnp.ndarray): Target frequencies for interpolation.
    
    Returns:
        jnp.ndarray: Interpolated values at the target frequencies.
    
    Note:
        Extrapolation is allowed but will generate a warning if frequencies
        are outside the data range.
    """
    assert isinstance(freqs_values, jnp.ndarray)
    assert isinstance(frequencies, jnp.ndarray)
    assert freqs_values.ndim == 2
    assert frequencies.ndim == 1

    freqs, values = freqs_values.T

    assert jnp.min(freqs) * 0.40 <= jnp.min(frequencies)
    assert jnp.max(frequencies) <= jnp.max(freqs) * 1.30

    if jnp.any(frequencies < jnp.min(freqs)) or jnp.any(frequencies > jnp.max(freqs)):
        warnings.warn(
            "Extrapolation detected: Some frequencies are outside the given data range.",
            UserWarning,
        )

    values_interpolated = jnp.interp(
        frequencies,
        freqs,
        values,
        left="extrapolate",
        right="extrapolate",
    )

    return values_interpolated


def interpolate_material_n_k(material, frequencies):
    """
    Interpolate refractive indices for a material to target frequencies.
    
    This function handles special materials (Air, PEC) and interpolates
    refractive index data for other materials to the specified frequencies.
    
    Args:
        material (str): Name of the material.
        frequencies (jnp.ndarray): Target frequencies in Hz.
    
    Returns:
        tuple: (n_material, k_material) - Real and imaginary parts of the
               refractive index at the target frequencies.
    """
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    if material == "Air":
        n_material = jnp.ones_like(frequencies)
        k_material = jnp.zeros_like(frequencies)
    elif material == "PEC":
        n_material = jnp.zeros_like(frequencies) + jnp.inf
        k_material = jnp.zeros_like(frequencies)
    else:
        data_n, data_k = load_material(material)
        n_material = interpolate(data_n, frequencies)
        k_material = interpolate(data_k, frequencies)

    return n_material, k_material


def get_eps_mu(materials, frequencies):
    """
    Get permittivity and permeability for a list of materials.
    
    This function handles radar materials using the Michielssen database
    and special materials like Air and PEC. It returns permittivity and
    permeability arrays suitable for multilayer calculations.
    
    Args:
        materials (list or onp.ndarray): List of material names or indices.
                                        First material must be "Air".
        frequencies (jnp.ndarray): Frequencies in Hz.
    
    Returns:
        tuple: (eps_r, mu_r) - Relative permittivity and permeability arrays
               with shape (n_freq, n_layers).
    
    Raises:
        NotImplementedError: If the last material is not supported.
    """
    assert isinstance(materials, (list, onp.ndarray))
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1
    assert materials[0] == "Air"

    materials = onp.array(materials)
    eps_r, mu_r = utils_radar_materials.get_eps_mu_Michielssen(
        materials[1:-1].astype(int), frequencies
    )

    n_k_air = get_n_k(materials[:1], frequencies)
    n_k_air = n_k_air.T
    eps_air, mu_air = convert_n_k_to_eps_mu_for_non_magnetic_materials(n_k_air)

    if materials[-1] == "PEC":
        eps_last = jnp.zeros_like(eps_air) + jnp.inf
        mu_last = jnp.ones_like(eps_air)
    else:
        try:
            eps_last, mu_last = utils_radar_materials.get_eps_mu_Michielssen(
                materials[-1:].astype(int), frequencies
            )
        except:
            raise NotImplementedError("This condition is not implemented yet.")

    eps_r = jnp.concatenate([eps_air, eps_r, eps_last], axis=0)
    mu_r = jnp.concatenate([mu_air, mu_r, mu_last], axis=0)

    eps_r = eps_r.T
    mu_r = mu_r.T

    return eps_r, mu_r


def interpolate_material_eps_mu(material, frequencies):
    """
    Interpolate permittivity and permeability for a material.
    
    This function loads frequency-dependent permittivity and permeability data
    and interpolates to the target frequencies.
    
    Args:
        material (str): Name of the material.
        frequencies (jnp.ndarray): Target frequencies in Hz.
    
    Returns:
        tuple: (eps_r_real, eps_r_imag, mu_r_real, mu_r_imag) - Real and
               imaginary parts of permittivity and permeability.
    
    Note:
        This function requires the material to have frequency-dependent
        permittivity and permeability data available.
    """
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    data_eps_r, data_mu_r = load_material_f_ghz(material)

    eps_r_real = interpolate(jnp.real(data_eps_r), frequencies)
    eps_r_imag = interpolate(jnp.imag(data_eps_r), frequencies)

    mu_r_real = interpolate(jnp.real(data_eps_r), frequencies)
    mu_r_imag = interpolate(jnp.imag(data_eps_r), frequencies)

    return eps_r_real, eps_r_imag, mu_r_real, mu_r_imag


def get_n_k(materials, frequencies):
    """
    Get refractive indices for a list of materials.
    
    This function interpolates refractive index data for each material to
    the target frequencies and returns a complex refractive index array.
    
    Args:
        materials (list or onp.ndarray): List of material names.
        frequencies (jnp.ndarray): Target frequencies in Hz.
    
    Returns:
        jnp.ndarray: Complex refractive indices with shape (n_freq, n_layers).
    """
    assert isinstance(materials, (list, onp.ndarray))
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    num_layers = len(materials)
    num_frequencies = frequencies.shape[0]

    n_k = jnp.ones((num_layers, num_frequencies), dtype=jnp.complex128)

    for ind, material in enumerate(materials):
        n_material, k_material = interpolate_material_n_k(material, frequencies)
        n_k = n_k.at[ind, :].set(n_material + 1j * k_material)

    n_k = n_k.T
    return n_k


def get_n_k_surrounded_by_air(materials, frequencies):
    """
    Get refractive indices for materials surrounded by air layers.
    
    This function adds air layers at the beginning and end of the material stack,
    which is a common requirement for multilayer calculations.
    
    Args:
        materials (list or onp.ndarray): List of material names.
        frequencies (jnp.ndarray): Target frequencies in Hz.
    
    Returns:
        jnp.ndarray: Complex refractive indices with shape (n_freq, n_layers + 2).
    """
    assert isinstance(materials, (list, onp.ndarray))
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    n_k = get_n_k(onp.concatenate([["Air"], materials, ["Air"]], axis=0), frequencies)
    return n_k


def convert_n_k_to_eps_mu_for_non_magnetic_materials(n_k):
    """
    Convert refractive indices to permittivity and permeability for non-magnetic materials.
    
    This function assumes μ = 1 (non-magnetic materials) and calculates
    ε = n² using the relationship between refractive index and material properties.
    
    Args:
        n_k (jnp.ndarray): Complex refractive indices.
    
    Returns:
        tuple: (eps, mu) - Relative permittivity and permeability arrays.
               mu will be all ones for non-magnetic materials.
    """
    eps = jnp.conj(n_k**2)
    mu = jnp.ones_like(eps)

    return eps, mu
