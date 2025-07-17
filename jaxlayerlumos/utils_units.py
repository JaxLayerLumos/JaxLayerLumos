"""
Unit conversion utilities and physical constants.

This module provides access to fundamental physical constants and unit conversion
functions commonly used in optical calculations. All constants are sourced from
scipy.constants for accuracy and consistency.
"""

import scipy.constants as scic


def get_light_speed():
    """
    Get the speed of light in vacuum.
    
    Returns:
        float: Speed of light in m/s (approximately 299,792,458 m/s).
    """
    return scic.c


def get_nano():
    """
    Get the nano prefix multiplier.
    
    Returns:
        float: 1e-9 (nano prefix).
    """
    return scic.nano


def get_micro():
    """
    Get the micro prefix multiplier.
    
    Returns:
        float: 1e-6 (micro prefix).
    """
    return scic.micro


def get_milli():
    """
    Get the milli prefix multiplier.
    
    Returns:
        float: 1e-3 (milli prefix).
    """
    return scic.milli


def get_centi():
    """
    Get the centi prefix multiplier.
    
    Returns:
        float: 1e-2 (centi prefix).
    """
    return scic.centi


def get_giga():
    """
    Get the giga prefix multiplier.
    
    Returns:
        float: 1e9 (giga prefix).
    """
    return scic.giga


def get_planck_constant():
    """
    Get Planck's constant.
    
    Returns:
        float: Planck's constant in J⋅s.
    """
    return scic.h


def get_elementary_charge():
    """
    Get the elementary charge.
    
    Returns:
        float: Elementary charge in C.
    """
    return scic.e


def convert_nm_to_m(thicknesses):
    """
    Convert thicknesses from nanometers to meters.
    
    Args:
        thicknesses: Thickness values in nanometers.
    
    Returns:
        Thickness values converted to meters.
    """
    return thicknesses * get_nano()


def convert_m_to_nm(thicknesses):
    """
    Convert thicknesses from meters to nanometers.
    
    Args:
        thicknesses: Thickness values in meters.
    
    Returns:
        Thickness values converted to nanometers.
    """
    return thicknesses / get_nano()


def convert_m_to_um(thicknesses):
    """
    Convert thicknesses from meters to micrometers.
    
    Args:
        thicknesses: Thickness values in meters.
    
    Returns:
        Thickness values converted to micrometers.
    """
    return thicknesses / get_micro()


def convert_mm_to_m(thicknesses):
    """
    Convert thicknesses from millimeters to meters.
    
    Args:
        thicknesses: Thickness values in millimeters.
    
    Returns:
        Thickness values converted to meters.
    """
    return thicknesses * get_milli()


def convert_cm_to_m(thicknesses):
    """
    Convert thicknesses from centimeters to meters.
    
    Args:
        thicknesses: Thickness values in centimeters.
    
    Returns:
        Thickness values converted to meters.
    """
    return thicknesses * get_centi()
