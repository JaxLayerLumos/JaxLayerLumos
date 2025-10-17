"""
Light source utilities for optical calculations.

This module provides functions for loading and interpolating light source spectra,
including standard solar spectra (AM0, AM1.5G, AM1.5D) and other light sources.
It handles wavelength-dependent irradiance data and provides interpolation to
arbitrary wavelength ranges.
"""

import jax.numpy as jnp
import os
import csv
from pathlib import Path
import json

from jaxlayerlumos import utils_units


def load_light_source_json():
    """
    Load the light source index file that maps light source names to CSV files.
    
    Returns:
        tuple: (light_source_indices, current_dir) where light_source_indices is a
               dictionary mapping light source names to CSV filenames, and current_dir
               is the directory containing the light source data.
    """
    current_dir = str(Path(__file__).parent)
    light_source_file = os.path.join(current_dir, "light_sources.json")

    with open(light_source_file, "r") as file_json:
        light_source_indices = json.load(file_json)

    return light_source_indices, current_dir


def interpolate(wavelength, data_wavelength, data_irradiance):
    """
    Interpolate irradiance data to target wavelengths.
    
    Args:
        wavelength (jnp.ndarray): Target wavelengths for interpolation.
        data_wavelength (jnp.ndarray): Wavelengths of the source data.
        data_irradiance (jnp.ndarray): Irradiance values of the source data.
    
    Returns:
        jnp.ndarray: Interpolated irradiance values at the target wavelengths.
    """
    values_interpolated = jnp.interp(wavelength, data_wavelength, data_irradiance)
    return values_interpolated


def interpolate_light_source_irradiance(light_source, wavelength):
    """
    Interpolate a light source's irradiance spectrum to target wavelengths.
    
    Args:
        light_source (str): Name of the light source.
        wavelength (jnp.ndarray): Target wavelengths in meters.
    
    Returns:
        jnp.ndarray: Interpolated irradiance values at the target wavelengths.
    """
    data_wavelength, data_irradiance = load_light_source_wavelength(light_source)
    irradiance = interpolate(wavelength, data_wavelength, data_irradiance)
    return irradiance


def load_light_source_wavelength(light_source):
    """
    Load light source data with wavelengths in meters.
    
    This function loads light source data and converts wavelengths from nanometers
    to meters.
    
    Args:
        light_source (str): Name of the light source.
    
    Returns:
        tuple: (data_wavelength, data_irradiance) - Wavelengths in meters and
               corresponding irradiance values.
    """
    data_wavelength, data_irradiance = load_light_source_wavelength_nm(light_source)
    return data_wavelength * utils_units.get_nano(), data_irradiance


def get_irradiance(light_source, wavelength):
    """
    Get irradiance values for a light source at specified wavelengths.
    
    This is a convenience function that loads and interpolates light source data
    to the target wavelengths.
    
    Args:
        light_source (str): Name of the light source.
        wavelength (jnp.ndarray): Target wavelengths in meters.
    
    Returns:
        jnp.ndarray: Irradiance values at the target wavelengths.
    """
    irradiance = interpolate_light_source_irradiance(light_source, wavelength)
    return irradiance


def load_light_source_wavelength_nm(light_source):
    """
    Load light source data with wavelengths in nanometers.
    
    This function reads the CSV file for the specified light source and extracts
    the wavelength-dependent irradiance data.
    
    Args:
        light_source (str): Name of the light source.
    
    Returns:
        tuple: (wavelength, irradiance) - Wavelengths in nanometers and
               corresponding irradiance values.
    
    Raises:
        ValueError: If the light source is not found in the database.
    """
    light_source_indices, str_directory = load_light_source_json()
    str_file = light_source_indices.get(light_source)

    if not str_file:
        raise ValueError(f"Light Source {light_source} not found in JaxLayerLumos.")

    str_csv = os.path.join(str_directory, str_file)

    wavelength = []
    irradiance = []

    with open(str_csv, mode="r") as file_csv:
        reader = csv.reader(file_csv)

        for line in reader:
            wavelength.append(float(line[0]))
            irradiance.append(float(line[1]))

    wavelength = jnp.array(wavelength)
    irradiance = jnp.array(irradiance)

    return wavelength, irradiance
