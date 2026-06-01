"""
Composite color transformation functions for spectral data.

This module provides high-level functions that combine spectral-to-XYZ conversion
with color space transformations to directly convert spectral data to various
color spaces in a single function call.
"""

from jaxlayerlumos.colors import transform


def spectrum_to_xyY(
    wavelengths, values, str_color_space="cie1931", str_illuminant="d65"
):
    """
    Convert spectral data directly to xyY chromaticity coordinates.
    
    This is a convenience function that combines spectrum_to_XYZ and XYZ_to_xyY
    to directly convert spectral data to xyY chromaticity coordinates.
    
    Args:
        wavelengths (jnp.ndarray): Wavelengths in meters.
        values (jnp.ndarray): Spectral values (reflectance, transmittance, etc.).
        str_color_space (str, optional): Color space for color matching functions.
                                        Defaults to "cie1931".
        str_illuminant (str, optional): Illuminant name. Defaults to "d65".
    
    Returns:
        jnp.ndarray: xyY chromaticity coordinates [x, y, Y].
    """
    XYZ = transform.spectrum_to_XYZ(
        wavelengths,
        values,
        str_color_space=str_color_space,
        str_illuminant=str_illuminant,
    )
    xyY = transform.XYZ_to_xyY(XYZ)

    return xyY


def spectrum_to_sRGB(
    wavelengths,
    values,
    str_color_space="cie1931",
    str_illuminant="d65",
    use_clipping=False,
):
    """
    Convert spectral data directly to sRGB color space.
    
    This is a convenience function that combines spectrum_to_XYZ and XYZ_to_sRGB
    to directly convert spectral data to sRGB color space.
    
    Args:
        wavelengths (jnp.ndarray): Wavelengths in meters.
        values (jnp.ndarray): Spectral values (reflectance, transmittance, etc.).
        str_color_space (str, optional): Color space for color matching functions.
                                        Defaults to "cie1931".
        str_illuminant (str, optional): Illuminant name. Defaults to "d65".
        use_clipping (bool, optional): Whether to clip RGB values to [0, 1].
                                      Defaults to False.
    
    Returns:
        jnp.ndarray: sRGB values [R, G, B] in the range [0, 1].
    """
    XYZ = transform.spectrum_to_XYZ(
        wavelengths,
        values,
        str_color_space=str_color_space,
        str_illuminant=str_illuminant,
    )
    sRGB = transform.XYZ_to_sRGB(XYZ, use_clipping=use_clipping)

    return sRGB


def spectrum_to_Lab(
    wavelengths, values, str_color_space="cie1931", str_illuminant="d65"
):
    """
    Convert spectral data directly to CIE Lab color space.
    
    This is a convenience function that combines spectrum_to_XYZ and XYZ_to_Lab
    to directly convert spectral data to CIE Lab color space.
    
    Args:
        wavelengths (jnp.ndarray): Wavelengths in meters.
        values (jnp.ndarray): Spectral values (reflectance, transmittance, etc.).
        str_color_space (str, optional): Color space for color matching functions.
                                        Defaults to "cie1931".
        str_illuminant (str, optional): Illuminant name. Defaults to "d65".
    
    Returns:
        jnp.ndarray: CIE Lab values [L, a, b].
    """
    XYZ = transform.spectrum_to_XYZ(
        wavelengths,
        values,
        str_color_space=str_color_space,
        str_illuminant=str_illuminant,
    )
    Lab = transform.XYZ_to_Lab(XYZ, str_illuminant=str_illuminant)

    return Lab
