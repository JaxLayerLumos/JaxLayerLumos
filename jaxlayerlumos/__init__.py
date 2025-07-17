"""
JaxLayerLumos: A JAX-based library for optical multilayer stack calculations.

This package provides tools for calculating reflection and transmission coefficients
for multilayer optical stacks using JAX for efficient computation and automatic differentiation.
It supports both TE and TM polarizations, various materials, and different light sources.

Main Functions:
    - stackrt_n_k: Calculate reflection/transmission for refractive index stacks
    - stackrt_eps_mu: Calculate reflection/transmission for permittivity/permeability stacks
    - stackrt_n_k_0: Wrapper for normal incidence (0 degrees)
    - stackrt_n_k_45: Wrapper for 45-degree incidence

The package includes utilities for:
    - Material properties and conversions
    - Light source spectra (AM0, AM1.5G, AM1.5D)
    - Color calculations and transformations
    - Unit conversions
    - Position and geometry calculations
    - Radio frequency and radar materials
"""

import jax

jax.config.update("jax_enable_x64", True)

from jaxlayerlumos.jaxlayerlumos import stackrt_n_k
from jaxlayerlumos.jaxlayerlumos import stackrt_eps_mu

from jaxlayerlumos.wrappers import stackrt_n_k_0
from jaxlayerlumos.wrappers import stackrt_n_k_45


stackrt = stackrt_n_k
stackrt0 = stackrt_n_k_0
stackrt45 = stackrt_n_k_45
