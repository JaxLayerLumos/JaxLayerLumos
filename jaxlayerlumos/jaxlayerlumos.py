"""
Core optical multilayer stack calculation functions using JAX.

This module provides the fundamental functions for calculating reflection and transmission
coefficients for multilayer optical stacks. It implements the transfer matrix method
for both TE and TM polarizations, supporting complex refractive indices and arbitrary
incidence angles.

The main functions handle:
- Single frequency/angle calculations (stackrt_eps_mu_base)
- Multiple frequencies/angles (stackrt_eps_mu_theta, stackrt_eps_mu)
- Refractive index input (stackrt_n_k)

All functions are JAX-compatible for automatic differentiation and GPU acceleration.
"""

import jax
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units


def true_fun(r_jk_TE, t_jk_TE, r_jk_TM, t_jk_TM):
    """
    Set reflection and transmission coefficients for perfect conductor boundary.
    
    This function is used when the last layer is a perfect conductor (infinite permittivity),
    setting the reflection coefficient to -1 and transmission coefficient to 1.
    
    Args:
        r_jk_TE: TE reflection coefficients
        t_jk_TE: TE transmission coefficients  
        r_jk_TM: TM reflection coefficients
        t_jk_TM: TM transmission coefficients
        
    Returns:
        Tuple of modified reflection and transmission coefficients
    """
    r_jk_TE = r_jk_TE.at[-1].set(-1.0)
    t_jk_TE = t_jk_TE.at[-1].set(1.0)
    r_jk_TM = r_jk_TM.at[-1].set(-1.0)
    t_jk_TM = t_jk_TM.at[-1].set(1.0)

    return r_jk_TE, t_jk_TE, r_jk_TM, t_jk_TM


def false_fun(r_jk_TE, t_jk_TE, r_jk_TM, t_jk_TM):
    """
    Return reflection and transmission coefficients unchanged.
    
    This function is used when the last layer is not a perfect conductor,
    returning the coefficients as calculated.
    
    Args:
        r_jk_TE: TE reflection coefficients
        t_jk_TE: TE transmission coefficients
        r_jk_TM: TM reflection coefficients  
        t_jk_TM: TM transmission coefficients
        
    Returns:
        Tuple of reflection and transmission coefficients unchanged
    """
    return r_jk_TE, t_jk_TE, r_jk_TM, t_jk_TM


def stackrt_eps_mu_base(eps_r, mu_r, thicknesses, f_i, thetas_k, return_coeffs=False):
    """
    Calculate reflection and transmission coefficients for a multilayer stack.
    
    This is the core function that implements the transfer matrix method for calculating
    reflection and transmission coefficients of a multilayer optical stack. It handles
    both TE and TM polarizations and supports complex permittivity and permeability.
    
    Args:
        eps_r (jnp.ndarray): Relative permittivity of each layer. Shape (n_layers,).
        mu_r (jnp.ndarray): Relative permeability of each layer. Shape (n_layers,).
        thicknesses (jnp.ndarray): Thickness of each layer in meters. Shape (n_layers,).
                                  First and last elements must be 0 (semi-infinite media).
        f_i (float): Frequency in Hz.
        thetas_k (jnp.ndarray): Incidence angles in radians. Shape (n_angles,).
        return_coeffs (bool, optional): If True, return additional coefficients and
                                       intermediate calculations. Defaults to False.
    
    Returns:
        If return_coeffs=False:
            tuple: (R_TE, T_TE, R_TM, T_TM) - Reflection and transmission coefficients
                  for TE and TM polarizations.
        If return_coeffs=True:
            tuple: (R_TE, T_TE, R_TM, T_TM, results_coeffs) - Coefficients plus
                  additional data including field coefficients, refractive indices,
                  and power calculations.
    
    Note:
        - The first and last layers are assumed to be semi-infinite media (thickness=0)
        - Perfect conductor boundaries are automatically handled
        - All calculations are performed using JAX for GPU acceleration and differentiation
    """
    assert isinstance(eps_r, jnp.ndarray)
    assert isinstance(mu_r, jnp.ndarray)
    assert isinstance(thicknesses, jnp.ndarray)

    assert eps_r.ndim == 1
    assert mu_r.ndim == 1
    assert thicknesses.ndim == 1

    assert thicknesses[0] == 0
    assert thicknesses[-1] == 0

    assert eps_r.shape[0] == thicknesses.shape[0]
    assert mu_r.shape[0] == thicknesses.shape[0]

    c = utils_units.get_light_speed()
    n = jnp.conj(jnp.sqrt(eps_r * mu_r))
    k = 2 * jnp.pi / c * f_i * n
    eta = jnp.conj(jnp.sqrt(mu_r / eps_r))

    sin_theta = jnp.expand_dims(jnp.sin(thetas_k), axis=0)
    sin_theta = sin_theta * n[0] / n
    cos_theta_t = jnp.sqrt(1 - sin_theta**2)
    kz = k * cos_theta_t

    upper_bound = 600.0
    delta = thicknesses * kz
    delta = jnp.real(delta) + 1j * jnp.clip(jnp.imag(delta), -upper_bound, upper_bound)

    Z_TE = eta / cos_theta_t
    Z_TM = eta * cos_theta_t

    r_jk_TE = (Z_TE[1:] - Z_TE[:-1]) / (Z_TE[1:] + Z_TE[:-1])
    t_jk_TE = (2 * Z_TE[1:]) / (Z_TE[1:] + Z_TE[:-1])

    r_jk_TM = (Z_TM[1:] - Z_TM[:-1]) / (Z_TM[1:] + Z_TM[:-1])
    t_jk_TM = (
        (2 * Z_TM[1:]) / (Z_TM[1:] + Z_TM[:-1]) * cos_theta_t[:-1] / cos_theta_t[1:]
    )

    r_jk_TE, t_jk_TE, r_jk_TM, t_jk_TM = jax.lax.cond(
        jnp.isinf(jnp.real(eps_r[-1]))
        # & jnp.isclose(jnp.imag(eps_r[-1]), 0)
        & jnp.isclose(jnp.real(mu_r[-1]), 1) & jnp.isclose(jnp.imag(mu_r[-1]), 0),
        true_fun,
        false_fun,
        r_jk_TE,
        t_jk_TE,
        r_jk_TM,
        t_jk_TM,
    )

    t_inv_TE = 1 / t_jk_TE
    r_over_t_TE = r_jk_TE / t_jk_TE

    D_TE = jnp.stack(
        [
            jnp.stack([t_inv_TE, r_over_t_TE], axis=-1),
            jnp.stack([r_over_t_TE, t_inv_TE], axis=-1),
        ],
        axis=-2,
    )

    t_inv_TM = 1 / t_jk_TM
    r_over_t_TM = r_jk_TM / t_jk_TM

    D_TM = jnp.stack(
        [
            jnp.stack([t_inv_TM, r_over_t_TM], axis=-1),
            jnp.stack([r_over_t_TM, t_inv_TM], axis=-1),
        ],
        axis=-2,
    )

    exp_neg_jdelta = jnp.exp(-1j * delta[0:-1])
    exp_pos_jdelta = jnp.exp(1j * delta[0:-1])

    zeros = jnp.zeros_like(exp_neg_jdelta)

    P = jnp.stack(
        [
            jnp.stack([exp_neg_jdelta, zeros], axis=-1),
            jnp.stack([zeros, exp_pos_jdelta], axis=-1),
        ],
        axis=-2,
    )

    DP_TE = jnp.matmul(P, D_TE)
    DP_TM = jnp.matmul(P, D_TM)

    if not return_coeffs:

        def matmul_scan(a, b):
            return jnp.matmul(a, b)

        M_TE = jax.lax.associative_scan(matmul_scan, DP_TE)[-1]
        M_TM = jax.lax.associative_scan(matmul_scan, DP_TM)[-1]
    else:

        def matmul_left(a, b):
            return jnp.matmul(b, a)

        DP_TE_rev = jnp.flip(DP_TE, axis=0)

        M_TE_all = jax.lax.associative_scan(matmul_left, DP_TE_rev)
        M_TE_flipped = jnp.flip(M_TE_all, axis=0)

        M_TE = M_TE_all[-1]

        coeff_TE = M_TE_flipped[:, :, 0]
        coeff_TE = jnp.concatenate((coeff_TE, jnp.array([[1, 0]])), axis=0)
        coeff_TE = coeff_TE / M_TE[0, 0]

        DP_TM_rev = jnp.flip(DP_TM, axis=0)

        M_TM_all = jax.lax.associative_scan(matmul_left, DP_TM_rev)
        M_TM_flipped = jnp.flip(M_TM_all, axis=0)
        M_TM = M_TM_all[-1]

        coeff_TM = M_TM_flipped[:, :, 0]
        coeff_TM = jnp.concatenate((coeff_TM, jnp.array([[1, 0]])), axis=0)
        coeff_TM = coeff_TM / M_TM[0, 0]

        results_coeffs = {
            "coeff_TE": coeff_TE,
            "coeff_TM": coeff_TM,
            "n": n,
            "k_z": kz,
            "cos_theta": cos_theta_t,
        }

    r_TE_i = M_TE[1, 0] / M_TE[0, 0]
    t_TE_i = 1 / M_TE[0, 0]

    r_TM_i = M_TM[1, 0] / M_TM[0, 0]
    t_TM_i = 1 / M_TM[0, 0]

    R_TE = jnp.abs(r_TE_i) ** 2
    R_TM = jnp.abs(r_TM_i) ** 2

    T_TE = (
        jnp.abs(t_TE_i) ** 2
        * jnp.real(n[-1] * cos_theta_t[-1])
        / jnp.real(n[0] * cos_theta_t[0])
    )
    T_TM = (
        jnp.abs(t_TM_i) ** 2
        * jnp.real(n[-1] * jnp.conj(cos_theta_t[-1]))
        / jnp.real(n[0] * jnp.conj(cos_theta_t[0]))
    )

    if return_coeffs:
        power_entering_TE = jnp.real(
            n[0] * cos_theta_t[0] * (1 + jnp.conj(r_TE_i)) * (1 - r_TE_i)
        ) / jnp.real(n[0] * cos_theta_t[0])

        power_entering_TM = jnp.real(
            n[0] * jnp.conj(cos_theta_t[0]) * (1 + r_TM_i) * (1 - jnp.conj(r_TM_i))
        ) / jnp.real(n[0] * jnp.conj(cos_theta_t[0]))

        results_coeffs.update(
            {
                "power_entering_TE": power_entering_TE,
                "power_entering_TM": power_entering_TM,
                "R_TE": R_TE,
                "T_TE": T_TE,
                "R_TM": R_TM,
                "T_TM": T_TM,
            }
        )
        return R_TE, T_TE, R_TM, T_TM, results_coeffs
    else:
        return R_TE, T_TE, R_TM, T_TM


def stackrt_eps_mu_theta(eps_r, mu_r, d, f, theta, return_coeffs=False):
    """
    Calculate reflection and transmission coefficients for multiple frequencies at a single angle.
    
    This function vectorizes the base calculation over multiple frequencies for a given
    incidence angle. It handles the conversion from degrees to radians and applies
    special handling for perfect conductor boundaries.
    
    Args:
        eps_r (jnp.ndarray): Relative permittivity for each frequency and layer.
                             Shape (n_freq, n_layers).
        mu_r (jnp.ndarray): Relative permeability for each frequency and layer.
                            Shape (n_freq, n_layers).
        d (jnp.ndarray): Thickness of each layer in meters. Shape (n_layers,).
        f (jnp.ndarray): Frequencies in Hz. Shape (n_freq,).
        theta (float): Incidence angle in degrees.
        return_coeffs (bool, optional): If True, return additional coefficients.
                                       Defaults to False.
    
    Returns:
        If return_coeffs=False:
            tuple: (R_TE, T_TE, R_TM, T_TM) - Arrays of reflection and transmission
                  coefficients for each frequency.
        If return_coeffs=True:
            tuple: (R_TE, T_TE, R_TM, T_TM, results_coeffs) - Coefficients plus
                  additional data for each frequency.
    """
    assert isinstance(eps_r, jnp.ndarray)
    assert isinstance(mu_r, jnp.ndarray)
    assert isinstance(d, jnp.ndarray)
    assert isinstance(f, jnp.ndarray)

    assert eps_r.ndim == 2
    assert mu_r.ndim == 2
    assert d.ndim == 1
    assert f.ndim == 1

    assert eps_r.shape[0] == f.shape[0]
    assert eps_r.shape[1] == d.shape[0]

    assert mu_r.shape[0] == f.shape[0]
    assert mu_r.shape[1] == d.shape[0]

    theta_rad = jnp.radians(theta)

    if not return_coeffs:
        fun_mapped = jax.vmap(stackrt_eps_mu_base, (0, 0, None, 0, None), (0, 0, 0, 0))

        R_TE, T_TE, R_TM, T_TM = fun_mapped(eps_r, mu_r, d, f, theta_rad)
    else:
        fun_mapped = jax.vmap(
            stackrt_eps_mu_base, (0, 0, None, 0, None, None), (0, 0, 0, 0, 0)
        )

        R_TE, T_TE, R_TM, T_TM, results_coeffs = fun_mapped(
            eps_r, mu_r, d, f, theta_rad, return_coeffs
        )

    if (
        jnp.all(jnp.isinf(jnp.real(eps_r[:, -1])))
        # and jnp.allclose(jnp.imag(eps_r[:, -1]), 0)
        and jnp.allclose(jnp.real(mu_r[:, -1]), 1)
        and jnp.allclose(jnp.imag(mu_r[:, -1]), 0)
    ):  # TODO: it is needed? It is just enforcing outputs. Outputs should be zeros by calculation.
        T_TE = jnp.zeros_like(R_TE)
        T_TM = jnp.zeros_like(R_TM)

    if not return_coeffs:
        return R_TE, T_TE, R_TM, T_TM
    else:
        return R_TE, T_TE, R_TM, T_TM, results_coeffs


def stackrt_eps_mu(eps_r, mu_r, d, f, thetas):
    """
    Calculate reflection and transmission coefficients for multiple frequencies and angles.
    
    This function vectorizes the calculation over both frequencies and incidence angles.
    It handles input validation and converts single angle inputs to arrays.
    
    Args:
        eps_r (jnp.ndarray): Relative permittivity for each frequency and layer.
                             Shape (n_freq, n_layers).
        mu_r (jnp.ndarray): Relative permeability for each frequency and layer.
                            Shape (n_freq, n_layers).
        d (jnp.ndarray): Thickness of each layer in meters. Shape (n_layers,).
        f (jnp.ndarray): Frequencies in Hz. Shape (n_freq,).
        thetas (jnp.ndarray, float, int, or None): Incidence angle(s) in degrees.
                                                  If None, defaults to [0].
    
    Returns:
        tuple: (R_TE, T_TE, R_TM, T_TM) - Arrays of reflection and transmission
               coefficients. Shape (n_freq, n_angles) for each output.
    """
    if thetas is None:
        thetas = jnp.array([0])
    elif isinstance(thetas, (float, int)):
        thetas = jnp.array([thetas])

    fun_mapped = jax.vmap(
        stackrt_eps_mu_theta, (None, None, None, None, 0), (0, 0, 0, 0)
    )
    R_TE, T_TE, R_TM, T_TM = fun_mapped(eps_r, mu_r, d, f, thetas)

    return R_TE, T_TE, R_TM, T_TM


def stackrt_n_k(
    refractive_indices, thicknesses, frequencies, thetas, return_coeffs=False
):
    """
    Calculate reflection and transmission coefficients for a multilayer stack at
    different frequencies and incidence angles.

    Parameters:
    - refractive_indices: The refractive indices of the layers for each frequency.
        Shape should be (Nfreq, Nlayers), where Nfreq is the number of frequencies
        and Nlayers is the number of layers.
    - thicknesses: The thicknesses of the layers. Shape should be (Nlayers, ).
    - frequencies: The frequencies at which to calculate the coefficients. Shape
        should be (Nfreq, ).
    - thetas: The incidence angle(s) in degrees. Can be a single value or an array
        of angles. If it is None, it will be [0].

    Returns:
    - A tuple containing:
      - R_TE (jax.numpy.ndarray): Reflectance for TE polarization.
            Shape is (Nfreq, ).
      - T_TE (jax.numpy.ndarray): Transmittance for TE polarization.
            Shape is (Nfreq, ).
      - R_TM (jax.numpy.ndarray): Reflectance for TM polarization.
            Shape is (Nfreq, ).
      - T_TM (jax.numpy.ndarray): Transmittance for TM polarization.
            Shape is (Nfreq, ).

    """

    assert isinstance(refractive_indices, jnp.ndarray)
    assert isinstance(thicknesses, jnp.ndarray)
    assert isinstance(frequencies, jnp.ndarray)
    assert isinstance(thetas, (type(None), jnp.ndarray, float, int))

    assert refractive_indices.ndim == 2
    assert thicknesses.ndim == 1
    assert frequencies.ndim == 1

    assert refractive_indices.shape[0] == frequencies.shape[0]
    assert refractive_indices.shape[1] == thicknesses.shape[0]

    if thetas is None:
        thetas = jnp.array([0])
    elif isinstance(thetas, (float, int)):
        thetas = jnp.array([thetas])

    eps_r, mu_r = utils_materials.convert_n_k_to_eps_mu_for_non_magnetic_materials(
        refractive_indices
    )

    if not return_coeffs:
        fun_mapped = jax.vmap(
            stackrt_eps_mu_theta, (None, None, None, None, 0), (0, 0, 0, 0)
        )
        R_TE, T_TE, R_TM, T_TM = fun_mapped(
            eps_r, mu_r, thicknesses, frequencies, thetas
        )
        return R_TE, T_TE, R_TM, T_TM
    else:
        fun_mapped = jax.vmap(
            stackrt_eps_mu_theta, (None, None, None, None, 0, None), (0, 0, 0, 0, 0)
        )
        R_TE, T_TE, R_TM, T_TM, results_coeffs = fun_mapped(
            eps_r, mu_r, thicknesses, frequencies, thetas, return_coeffs
        )
        return R_TE, T_TE, R_TM, T_TM, results_coeffs
