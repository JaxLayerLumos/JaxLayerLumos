"""
Convenience wrapper functions for common incidence angles.

This module provides simplified interfaces for calculating reflection and transmission
coefficients at specific incidence angles (0° and 45°) that are commonly used in
optical applications.
"""

import jax.numpy as jnp

from jaxlayerlumos.jaxlayerlumos import stackrt_n_k


def stackrt_n_k_0(n, d, f):
    """
    Calculate reflection and transmission coefficients at normal incidence (0°).
    
    This is a convenience wrapper around stackrt_n_k for normal incidence calculations,
    which are commonly used in optical design and analysis.
    
    Args:
        n (jnp.ndarray): Refractive indices for each frequency and layer.
                         Shape (n_freq, n_layers).
        d (jnp.ndarray): Thickness of each layer in meters. Shape (n_layers,).
        f (jnp.ndarray): Frequencies in Hz. Shape (n_freq,).
    
    Returns:
        tuple: (R_TE, T_TE, R_TM, T_TM) - Reflection and transmission coefficients
               for TE and TM polarizations at normal incidence.
    """
    return stackrt_n_k(n, d, f, jnp.array([0]))


def stackrt_n_k_45(n, d, f):
    """
    Calculate reflection and transmission coefficients at 45° incidence.
    
    This is a convenience wrapper around stackrt_n_k for 45° incidence calculations,
    which are useful for analyzing angular-dependent optical properties.
    
    Args:
        n (jnp.ndarray): Refractive indices for each frequency and layer.
                         Shape (n_freq, n_layers).
        d (jnp.ndarray): Thickness of each layer in meters. Shape (n_layers,).
        f (jnp.ndarray): Frequencies in Hz. Shape (n_freq,).
    
    Returns:
        tuple: (R_TE, T_TE, R_TM, T_TM) - Reflection and transmission coefficients
               for TE and TM polarizations at 45° incidence.
    """
    return stackrt_n_k(n, d, f, jnp.array([45]))
