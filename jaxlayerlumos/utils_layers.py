"""
Layer thickness utilities for multilayer stack calculations.

This module provides helper functions for manipulating layer thicknesses in
multilayer optical stack calculations, particularly for handling boundary conditions.
"""

import jax.numpy as jnp


def get_thicknesses_surrounded_by_air(thicknesses):
    """
    Add zero-thickness air layers at the beginning and end of a layer stack.
    
    This function is used to prepare thickness arrays for multilayer calculations
    where the first and last layers are assumed to be semi-infinite media (air).
    The zero thicknesses indicate that these are boundary layers, not physical layers.
    
    Args:
        thicknesses (jnp.ndarray): Thickness values for the physical layers.
                                  Shape (n_layers,).
    
    Returns:
        jnp.ndarray: Thickness array with zeros prepended and appended.
                     Shape (n_layers + 2,).
    
    Example:
        >>> thicknesses = jnp.array([100e-9, 200e-9])  # 100nm, 200nm layers
        >>> result = get_thicknesses_surrounded_by_air(thicknesses)
        >>> print(result)  # [0.0, 100e-9, 200e-9, 0.0]
    """
    assert isinstance(thicknesses, jnp.ndarray)
    assert thicknesses.ndim == 1

    return jnp.concatenate([jnp.array([0.0]), thicknesses, jnp.array([0.0])], axis=0)
