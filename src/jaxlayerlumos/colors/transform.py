"""
Color space transformation functions for spectral data.

This module provides functions for converting spectral data to various color spaces
including CIE XYZ, xyY chromaticity coordinates, sRGB, and CIE Lab. It implements
standard color space transformations used in color science and display applications.
"""

import jax.numpy as jnp

from jaxlayerlumos.colors import utils


def spectrum_to_XYZ(
    wavelengths, values, str_color_space="cie1931", str_illuminant="d65"
):
    """
    Convert spectral data to CIE XYZ tristimulus values.
    
    This function calculates the CIE XYZ tristimulus values from spectral reflectance
    or transmittance data using color matching functions and an illuminant.
    
    Args:
        wavelengths (jnp.ndarray): Wavelengths in meters.
        values (jnp.ndarray): Spectral values (reflectance, transmittance, etc.).
        str_color_space (str, optional): Color space for color matching functions.
                                        Defaults to "cie1931".
        str_illuminant (str, optional): Illuminant name. Defaults to "d65".
    
    Returns:
        jnp.ndarray: CIE XYZ tristimulus values [X, Y, Z].
    
    Note:
        The calculation uses trapezoidal integration over the spectral range.
        Y represents luminance and is normalized to the illuminant.
    """
    assert isinstance(wavelengths, jnp.ndarray)
    assert isinstance(values, jnp.ndarray)
    assert wavelengths.ndim == 1
    assert values.ndim == 1
    assert wavelengths.shape[0] == values.shape[0]

    cmfs = utils.get_cmfs(wavelengths, str_color_space=str_color_space)
    illuminant = utils.get_illuminant(wavelengths, str_illuminant=str_illuminant)

    assert cmfs.ndim == 2
    assert illuminant.ndim == 2
    assert wavelengths.shape[0] == cmfs.shape[0] == illuminant.shape[0]
    assert cmfs.shape[1] == 4
    assert illuminant.shape[1] == 2
    assert jnp.all(cmfs[:, 0] == wavelengths)
    assert jnp.all(wavelengths == cmfs[:, 0])
    assert jnp.all(wavelengths == illuminant[:, 0])

    x = cmfs[:, 1]
    y = cmfs[:, 2]
    z = cmfs[:, 3]
    I = illuminant[:, 1]

    delta_wavelengths = wavelengths[1:] - wavelengths[:-1]
    Iy = I * y
    SIx = values * I * x
    SIy = values * I * y
    SIz = values * I * z

    denominator = jnp.sum(((Iy[1:] + Iy[:-1]) / 2) * delta_wavelengths)

    numerator_X = jnp.sum(((SIx[1:] + SIx[:-1]) / 2) * delta_wavelengths)
    numerator_Y = jnp.sum(((SIy[1:] + SIy[:-1]) / 2) * delta_wavelengths)
    numerator_Z = jnp.sum(((SIz[1:] + SIz[:-1]) / 2) * delta_wavelengths)

    X = numerator_X / denominator
    Y = numerator_Y / denominator
    Z = numerator_Z / denominator

    return jnp.array([X, Y, Z])


def XYZ_to_xyY(XYZ):
    """
    Convert CIE XYZ to xyY chromaticity coordinates.
    
    This function converts CIE XYZ tristimulus values to xyY chromaticity coordinates,
    where x and y represent the chromaticity and Y represents the luminance.
    
    Args:
        XYZ (jnp.ndarray): CIE XYZ tristimulus values [X, Y, Z].
    
    Returns:
        jnp.ndarray: xyY chromaticity coordinates [x, y, Y].
    
    Note:
        The chromaticity coordinates x and y are normalized so that x + y + z = 1,
        where z = 1 - x - y.
    """
    assert isinstance(XYZ, jnp.ndarray)
    assert XYZ.ndim == 1
    assert XYZ.shape[0] == 3

    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]

    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    return jnp.array([x, y, Y])


def XYZ_to_sRGB(XYZ, use_clipping=False):
    """
    Convert CIE XYZ to sRGB color space.
    
    This function converts CIE XYZ tristimulus values to sRGB color space using
    the standard transformation matrix and gamma correction.
    
    Args:
        XYZ (jnp.ndarray): CIE XYZ tristimulus values [X, Y, Z].
        use_clipping (bool, optional): Whether to clip RGB values to [0, 1].
                                      Defaults to False.
    
    Returns:
        jnp.ndarray: sRGB values [R, G, B] in the range [0, 1].
    
    Note:
        The transformation includes gamma correction (2.4 power law) and
        the standard sRGB transformation matrix.
    """
    assert isinstance(XYZ, jnp.ndarray)
    assert isinstance(use_clipping, bool)
    assert XYZ.ndim == 1
    assert XYZ.shape[0] == 3

    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]

    R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def transform_nonlinear(C):
        if C <= 0.0031308:
            C *= 12.92
        else:
            C = 1.055 * (C ** (1 / 2.4)) - 0.055

        return C

    R = transform_nonlinear(R)
    G = transform_nonlinear(G)
    B = transform_nonlinear(B)

    if use_clipping:
        R = jnp.clip(R, 0, 1)
        G = jnp.clip(G, 0, 1)
        B = jnp.clip(B, 0, 1)

    return jnp.array([R, G, B])


def XYZ_to_Lab(XYZ, str_illuminant="d65"):
    """
    Convert CIE XYZ to CIE Lab color space.
    
    This function converts CIE XYZ tristimulus values to CIE Lab color space,
    which is perceptually uniform and widely used in color science.
    
    Args:
        XYZ (jnp.ndarray): CIE XYZ tristimulus values [X, Y, Z].
        str_illuminant (str, optional): Reference illuminant for white point.
                                       Currently only supports "d65". Defaults to "d65".
    
    Returns:
        jnp.ndarray: CIE Lab values [L, a, b] where L is lightness and a, b are
                    chromaticity coordinates.
    
    Raises:
        ValueError: If the illuminant is not supported.
    
    Note:
        - L represents lightness (0 = black, 1 = white)
        - a represents red-green chromaticity (positive = red, negative = green)
        - b represents yellow-blue chromaticity (positive = yellow, negative = blue)
        - The transformation uses the D65 white point reference values
    """
    assert isinstance(XYZ, jnp.ndarray)
    assert XYZ.ndim == 1
    assert XYZ.shape[0] == 3

    if str_illuminant == "d65":
        Xn = 95.0489
        Yn = 100
        Zn = 108.8840
    else:
        raise ValueError

    XYZ *= 100

    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]

    def transform_nonlinear(t):
        delta = 6 / 29

        if t > delta**3:
            output = t ** (1 / 3)
        else:
            output = 1 / 3 * t * delta ** (-2) + 4 / 29

        return output

    L = 116 * transform_nonlinear(Y / Yn) - 16
    a = 500 * (transform_nonlinear(X / Xn) - transform_nonlinear(Y / Yn))
    b = 200 * (transform_nonlinear(Y / Yn) - transform_nonlinear(Z / Zn))

    L /= 100
    a /= 100
    b /= 100

    return jnp.array([L, a, b])
