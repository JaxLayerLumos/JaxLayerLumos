"""
Color science utilities for optical calculations.

This module provides functions for color calculations including:
- Spectral to color space conversions (XYZ, xyY, sRGB, Lab)
- Color matching functions and illuminants
- Color transformations and composite calculations

The module supports standard color spaces and illuminants commonly used
in optical and display applications.
"""

import jax

jax.config.update("jax_enable_x64", True)

from jaxlayerlumos.colors.transform import spectrum_to_XYZ
from jaxlayerlumos.colors.transform import XYZ_to_xyY
from jaxlayerlumos.colors.transform import XYZ_to_sRGB
from jaxlayerlumos.colors.transform import XYZ_to_Lab

from jaxlayerlumos.colors.composite import spectrum_to_xyY
from jaxlayerlumos.colors.composite import spectrum_to_sRGB
from jaxlayerlumos.colors.composite import spectrum_to_Lab
