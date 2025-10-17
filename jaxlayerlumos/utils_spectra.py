"""
Spectral conversion utilities for optical calculations.

This module provides functions for converting between different spectral units
commonly used in optical calculations, including frequencies, wavelengths, and
energies. It also includes predefined frequency ranges for visible light.
"""

import jax.numpy as jnp

from jaxlayerlumos import utils_units


def convert_frequencies_to_wavelengths(frequencies):
    """
    Convert frequencies to wavelengths using the speed of light.
    
    Args:
        frequencies (jnp.ndarray): Frequencies in Hz.
    
    Returns:
        jnp.ndarray: Wavelengths in meters.
    """
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    wavelengths = utils_units.get_light_speed() / frequencies
    return wavelengths


def convert_wavelengths_to_energy(wavelengths):
    """
    Convert wavelengths to photon energies in electron volts (eV).
    
    This function uses the relationship E = hc/λ to convert wavelengths
    to photon energies.
    
    Args:
        wavelengths (jnp.ndarray): Wavelengths in meters.
    
    Returns:
        jnp.ndarray: Photon energies in eV.
    """
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
    """
    Convert wavelengths to frequencies using the speed of light.
    
    Args:
        wavelengths (jnp.ndarray): Wavelengths in meters.
    
    Returns:
        jnp.ndarray: Frequencies in Hz.
    """
    assert isinstance(wavelengths, jnp.ndarray)
    assert wavelengths.ndim == 1

    frequencies = utils_units.get_light_speed() / wavelengths
    return frequencies


def get_frequencies_visible_light(num_wavelengths=1001):
    """
    Generate frequency array covering the visible light spectrum.
    
    This function creates a frequency array corresponding to wavelengths
    from 380 nm to 780 nm, which covers the standard visible light range.
    
    Args:
        num_wavelengths (int, optional): Number of wavelength points.
                                        Defaults to 1001.
    
    Returns:
        jnp.ndarray: Frequencies in Hz covering the visible spectrum.
    """
    wavelengths = jnp.linspace(
        380 * utils_units.get_nano(), 780 * utils_units.get_nano(), num_wavelengths
    )
    frequencies = convert_wavelengths_to_frequencies(wavelengths)

    return frequencies


def get_frequencies_wide_visible_light(num_wavelengths=1001):
    """
    Generate frequency array covering an extended visible light spectrum.
    
    This function creates a frequency array corresponding to wavelengths
    from 300 nm to 900 nm, which includes near-UV and near-IR regions
    in addition to the standard visible range.
    
    Args:
        num_wavelengths (int, optional): Number of wavelength points.
                                        Defaults to 1001.
    
    Returns:
        jnp.ndarray: Frequencies in Hz covering the extended visible spectrum.
    """
    wavelengths = jnp.linspace(
        300 * utils_units.get_nano(), 900 * utils_units.get_nano(), num_wavelengths
    )
    frequencies = convert_wavelengths_to_frequencies(wavelengths)

    return frequencies
