#
# This file will be deleted after testing our new implementation.
# Keep this file for a while.
#

import jax
import jax.numpy as jnp

from jaxlayerlumos import utils_spectra
from jaxlayerlumos import utils_units


def stackrt_eps_mu_base(
    eps_r, mu_r, thicknesses, f_i, thetas_k, is_back_layer_PEC=False
):
    assert isinstance(eps_r, jnp.ndarray)
    assert isinstance(mu_r, jnp.ndarray)
    assert isinstance(thicknesses, jnp.ndarray)

    assert eps_r.ndim == 1
    assert mu_r.ndim == 1
    assert thicknesses.ndim == 1

    assert eps_r.shape[0] == thicknesses.shape[0]
    assert mu_r.shape[0] == thicknesses.shape[0]

    num_layers = thicknesses.shape[0]

    c = utils_units.get_light_speed()
    k = 2 * jnp.pi / c * f_i * jnp.conj(jnp.sqrt(eps_r * mu_r))
    eta = jnp.conj(jnp.sqrt(mu_r / eps_r))

    sin_theta = jnp.expand_dims(jnp.sin(thetas_k), axis=0)
    sin_theta = sin_theta * k[0] / k
    cos_theta_t = jnp.sqrt(1 - sin_theta**2)
    kz = k * cos_theta_t

    upper_bound = 200.0
    delta = thicknesses * kz
    delta = jnp.real(delta) + 1j * jnp.clip(jnp.imag(delta), -upper_bound, upper_bound)

    Z_TE = eta / cos_theta_t
    Z_TM = eta * cos_theta_t

    M_TE = jnp.eye(2, dtype=jnp.complex128)
    M_TM = jnp.eye(2, dtype=jnp.complex128)

    for j in range(0, num_layers - 1):
        r_jk_TE = (Z_TE[j + 1] - Z_TE[j]) / (Z_TE[j + 1] + Z_TE[j])
        t_jk_TE = (2 * Z_TE[j + 1]) / (Z_TE[j + 1] + Z_TE[j])

        r_jk_TM = (Z_TM[j + 1] - Z_TM[j]) / (Z_TM[j + 1] + Z_TM[j])
        t_jk_TM = (2 * Z_TM[j + 1]) / (Z_TM[j + 1] + Z_TM[j])

        if j == num_layers - 2 and is_back_layer_PEC:
            # r_jk_TE = -jnp.ones_like(r_jk_TE)
            r_jk_TE, r_jk_TM = -1, -1
            t_jk_TE, t_jk_TM = 1, 1
            # jnp.ones_like(t_jk_TE)
            # r_jk_TM = -jnp.ones_like(r_jk_TM)
            # t_jk_TM = jnp.ones_like(t_jk_TM)

        D_jk_TE = jnp.array(
            [[1 / t_jk_TE, r_jk_TE / t_jk_TE], [r_jk_TE / t_jk_TE, 1 / t_jk_TE]],
            dtype=jnp.complex128,
        )

        D_jk_TM = jnp.array(
            [[1 / t_jk_TM, r_jk_TM / t_jk_TM], [r_jk_TM / t_jk_TM, 1 / t_jk_TM]],
            dtype=jnp.complex128,
        )

        P = jnp.array(
            [[jnp.exp(-1j * delta[j + 1]), 0], [0, jnp.exp(1j * delta[j + 1])]],
            dtype=jnp.complex128,
        )
        M_TE = jnp.dot(M_TE, jnp.dot(D_jk_TE, P))
        M_TM = jnp.dot(M_TM, jnp.dot(D_jk_TM, P))

    r_TE_i = M_TE[1, 0] / M_TE[0, 0]
    t_TE_i = 1 / M_TE[0, 0]

    r_TM_i = M_TM[1, 0] / M_TM[0, 0]
    t_TM_i = 1 / M_TM[0, 0]

    return r_TE_i, t_TE_i, r_TM_i, t_TM_i


def stackrt_eps_mu_theta(eps_r, mu_r, d, f, theta, is_back_layer_PEC=False):
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

    fun_mapped = jax.vmap(
        stackrt_eps_mu_base, (0, 0, None, 0, None, None), (0, 0, 0, 0)
    )

    r_TE, t_TE, r_TM, t_TM = fun_mapped(eps_r, mu_r, d, f, theta_rad, is_back_layer_PEC)

    #    n = jnp.conj(jnp.sqrt(eps_r * mu_r))

    R_TE = jnp.abs(r_TE) ** 2
    T_TE = jnp.abs(t_TE) ** 2
    R_TM = jnp.abs(r_TM) ** 2
    T_TM = jnp.abs(t_TM) ** 2

    if is_back_layer_PEC:
        T_TE = jnp.zeros_like(R_TE)
        T_TM = jnp.zeros_like(R_TM)

    return R_TE, T_TE, R_TM, T_TM


def stackrt_eps_mu(eps_r, mu_r, d, f, thetas, is_back_layer_PEC=False):
    if thetas is None:
        thetas = jnp.array([0])
    elif isinstance(thetas, (float, int)):
        thetas = jnp.array([thetas])

    fun_mapped = jax.vmap(
        stackrt_eps_mu_theta, (None, None, None, None, 0, None), (0, 0, 0, 0)
    )
    R_TE, T_TE, R_TM, T_TM = fun_mapped(eps_r, mu_r, d, f, thetas, is_back_layer_PEC)

    return R_TE, T_TE, R_TM, T_TM


def stackrt_n_k_base(refractive_indices_i, thicknesses, frequencies_i, thetas_k):
    assert isinstance(refractive_indices_i, jnp.ndarray)
    assert isinstance(thicknesses, jnp.ndarray)

    assert refractive_indices_i.ndim == 1
    assert thicknesses.ndim == 1
    assert refractive_indices_i.shape[0] == thicknesses.shape[0]

    eps_r = jnp.conj(refractive_indices_i**2)
    mu_r = jnp.ones_like(eps_r)

    r_TE_i, t_TE_i, r_TM_i, t_TM_i = stackrt_eps_mu_base(
        eps_r, mu_r, thicknesses, frequencies_i, thetas_k
    )

    return r_TE_i, t_TE_i, r_TM_i, t_TM_i


def stackrt_n_k_theta(refractive_indices, thicknesses, frequencies, theta):
    assert isinstance(refractive_indices, jnp.ndarray)
    assert isinstance(thicknesses, jnp.ndarray)
    assert isinstance(frequencies, jnp.ndarray)

    assert refractive_indices.ndim == 2
    assert thicknesses.ndim == 1
    assert frequencies.ndim == 1

    assert refractive_indices.shape[0] == frequencies.shape[0]
    assert refractive_indices.shape[1] == thicknesses.shape[0]

    eps_r = jnp.conj(refractive_indices ** 2)
    mu_r = jnp.ones_like(eps_r)
    theta_rad = jnp.radians(theta)

    fun_mapped = jax.vmap(stackrt_eps_mu_base, (0, 0, None, 0, None), (0, 0, 0, 0))
    r_TE, t_TE, r_TM, t_TM = fun_mapped(
        eps_r, mu_r, thicknesses, frequencies, theta_rad
    )

    # fun_mapped = jax.vmap(stackrt_n_k_base, (0, None, 0, None), (0, 0, 0, 0))
    # r_TE, t_TE, r_TM, t_TM = fun_mapped(
    #     refractive_indices, thicknesses, frequencies, theta_rad
    # )

    R_TE = jnp.abs(r_TE) ** 2
    T_TE = jnp.abs(t_TE) ** 2

    R_TM = jnp.abs(r_TM) ** 2
    T_TM = jnp.abs(t_TM) ** 2

    return R_TE, T_TE, R_TM, T_TM


def stackrt_n_k(refractive_indices, thicknesses, frequencies, thetas=None):
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

    fun_mapped = jax.vmap(stackrt_n_k_theta, (None, None, None, 0), (0, 0, 0, 0))
    R_TE, T_TE, R_TM, T_TM = fun_mapped(
        refractive_indices, thicknesses, frequencies, thetas
    )

    return R_TE, T_TE, R_TM, T_TM
