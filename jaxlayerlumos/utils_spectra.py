import jax.numpy as jnp
import os
import csv
from pathlib import Path
import json

from jaxlayerlumos import utils_units


def load_light_source_json():
    current_dir = str(Path(__file__).parent)
    light_source_file = os.path.join(current_dir, "light_sources.json")

    with open(light_source_file, "r") as file_json:
        light_source_indices = json.load(file_json)

    return light_source_indices, current_dir


def interpolate(wavelength, data_wavelength, data_irradiance):
    values_interpolated = jnp.interp(wavelength, data_wavelength, data_irradiance)
    return values_interpolated


def interpolate_light_source_irradiance(light_source, wavelength):
    data_wavelength, data_irradiance = load_light_source_wavelength(light_source)
    irradiance = interpolate(wavelength, data_wavelength, data_irradiance)
    return irradiance


def load_light_source_wavelength(light_source):
    data_wavelength, data_irradiance = load_light_source_wavelength_nm(light_source)
    return data_wavelength * 1e-9, data_irradiance


def get_irradiance(light_source, wavelength):
    irradiance = interpolate_light_source_irradiance(light_source, wavelength)
    return irradiance


def load_light_source_wavelength_nm(light_source):
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


def convert_frequencies_to_wavelengths(frequencies):
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    wavelengths = utils_units.get_light_speed() / frequencies
    return wavelengths


def convert_wavelengths_to_energy(wavelengths):
    # energy is in eV
    assert isinstance(wavelengths, jnp.ndarray)
    assert wavelengths.ndim == 1

    energy = (
        utils_units.get_light_speed()
        * utils_units.get_planck_constant()
        / wavelengths
        / utils_units.get_elementary_charge()
    )
    return energy


def convert_wavelengths_to_frequencies(wavelengths):
    assert isinstance(wavelengths, jnp.ndarray)
    assert wavelengths.ndim == 1

    frequencies = utils_units.get_light_speed() / wavelengths
    return frequencies


def get_frequencies_visible_light(num_wavelengths=1001):
    wavelengths = jnp.linspace(
        380 * utils_units.get_nano(), 780 * utils_units.get_nano(), num_wavelengths
    )
    frequencies = convert_wavelengths_to_frequencies(wavelengths)

    return frequencies


def get_frequencies_wide_visible_light(num_wavelengths=1001):
    wavelengths = jnp.linspace(
        300 * utils_units.get_nano(), 900 * utils_units.get_nano(), num_wavelengths
    )
    frequencies = convert_wavelengths_to_frequencies(wavelengths)

    return frequencies
