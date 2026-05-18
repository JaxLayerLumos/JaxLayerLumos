"""
Effective medium approximation utilities for two-material mixtures.
"""

import jax.numpy as jnp

from jaxlayerlumos import utils_graded_layers
from jaxlayerlumos import utils_materials


SUPPORTED_METHODS = ("linear", "Maxwell-Garnett", "Bruggeman")


def get_effective_medium(material1, material2, f1, v, method="linear"):
    """
    Return effective permittivity and permeability for a two-material mixture.

    Args:
        material1 (str): Name of material 1 in the JaxLayerLumos material database.
        material2 (str): Name of material 2 in the JaxLayerLumos material database.
        f1: Volume fraction of material1. May be scalar or broadcastable to v.
        v (jnp.ndarray): Frequency vector in Hz with shape (n_frequencies,).
        method (str): One of "linear", "Maxwell-Garnett", or "Bruggeman".

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Effective relative permittivity and
        permeability arrays with shape matching v.
    """
    assert isinstance(material1, str)
    assert isinstance(material2, str)
    assert isinstance(v, jnp.ndarray)
    assert v.ndim == 1

    _validate_effective_medium_method(method)

    f1 = _get_fraction_broadcast_to_v(f1, v)
    eps1, eps2 = _get_material_eps(material1, material2, v)
    eps_eff, mu_eff = _get_effective_eps_mu(eps1, eps2, f1, method)

    return eps_eff, mu_eff


def get_graded_effective_medium(
    material1,
    material2,
    v,
    num_graded_layers=10,
    method="linear",
    graded_method="linear",
    beta=3.0,
):
    """
    Return EMA layers whose fractions come from the graded layer utility.

    The graded layer utility returns material2 fractions, so this function uses
    f2 = ratio and f1 = 1 - ratio for a material1-to-material2 transition.

    Args:
        material1 (str): Name of material 1 in the JaxLayerLumos material database.
        material2 (str): Name of material 2 in the JaxLayerLumos material database.
        v (jnp.ndarray): Frequency vector in Hz with shape (n_frequencies,).
        num_graded_layers (int): Number of true intermediate EMA layers.
        method (str): EMA method. One of "linear", "Maxwell-Garnett", or
            "Bruggeman".
        graded_method (str): Fraction profile from utils_graded_layers. One of
            "linear", "sine", or "exponential".
        beta (float): Exponential steepness for graded_method="exponential".

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Effective relative permittivity and
        permeability arrays with shape (n_frequencies, num_graded_layers).
    """
    assert isinstance(material1, str)
    assert isinstance(material2, str)
    assert isinstance(v, jnp.ndarray)
    assert v.ndim == 1

    _validate_effective_medium_method(method)

    f2 = utils_graded_layers.get_mixing_ratios(
        num_graded_layers, method=graded_method, beta=beta
    )
    f1 = 1 - f2

    eps1, eps2 = _get_material_eps(material1, material2, v)
    eps_eff, mu_eff = _get_effective_eps_mu(
        eps1[:, jnp.newaxis],
        eps2[:, jnp.newaxis],
        f1[jnp.newaxis, :],
        method,
    )

    return eps_eff, mu_eff


def _get_fraction_broadcast_to_v(f1, v):
    f1 = jnp.asarray(f1)

    try:
        f1 = jnp.broadcast_to(f1, v.shape)
    except ValueError as exc:
        raise ValueError("f1 must be scalar or broadcast-compatible with v.") from exc

    if jnp.any(f1 < 0) or jnp.any(f1 > 1):
        raise ValueError("f1 must be between 0 and 1.")

    return f1


def _validate_effective_medium_method(method):
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            "Unsupported effective medium method. Use 'linear', "
            "'Maxwell-Garnett', or 'Bruggeman'."
        )


def _get_material_eps(material1, material2, v):
    n_k = utils_materials.get_n_k([material1, material2], v)
    eps_r, _ = utils_materials.convert_n_k_to_eps_mu_for_non_magnetic_materials(n_k)
    return eps_r[:, 0], eps_r[:, 1]


def _get_effective_eps_mu(eps1, eps2, f1, method):
    _validate_effective_medium_method(method)

    f2 = 1 - f1

    if method == "linear":
        eps_eff = f1 * eps1 + f2 * eps2
    elif method == "Maxwell-Garnett":
        eps_eff = _get_maxwell_garnett_eps(eps1, eps2, f1)
    else:
        eps_eff = _get_bruggeman_eps(eps1, eps2, f1, f2)

    mu_eff = jnp.ones_like(eps_eff)
    return eps_eff, mu_eff


def _get_maxwell_garnett_eps(eps1, eps2, f1):
    numerator = eps1 + 2 * eps2 + 2 * f1 * (eps1 - eps2)
    denominator = eps1 + 2 * eps2 - f1 * (eps1 - eps2)
    return eps2 * numerator / denominator


def _get_bruggeman_eps(eps1, eps2, f1, f2):
    b = f1 * (2 * eps1 - eps2) + f2 * (2 * eps2 - eps1)
    discriminant = b**2 + 8 * eps1 * eps2

    root_plus = (b + jnp.sqrt(discriminant)) / 4
    root_minus = (b - jnp.sqrt(discriminant)) / 4
    linear_eps = f1 * eps1 + f2 * eps2

    use_plus = jnp.abs(root_plus - linear_eps) <= jnp.abs(root_minus - linear_eps)
    return jnp.where(use_plus, root_plus, root_minus)
