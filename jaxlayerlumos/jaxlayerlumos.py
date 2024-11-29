import jax
import jax.numpy as jnp
from functools import partial
from jax import lax, vmap

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

    r_jk_TE = (Z_TE[1:] - Z_TE[:-1]) / (Z_TE[1:] + Z_TE[:-1])
    t_jk_TE = (2 * Z_TE[1:]) / (Z_TE[1:] + Z_TE[:-1])

    r_jk_TM = (Z_TM[1:] - Z_TM[:-1]) / (Z_TM[1:] + Z_TM[:-1])
    t_jk_TM = (2 * Z_TM[1:]) / (Z_TM[1:] + Z_TM[:-1])

    if is_back_layer_PEC:
        r_jk_TE = r_jk_TE.at[-1].set(-1.0)
        t_jk_TE = t_jk_TE.at[-1].set(1.0)
        r_jk_TM = r_jk_TM.at[-1].set(-1.0)
        t_jk_TM = t_jk_TM.at[-1].set(1.0)

    t_inv_TE = 1 / t_jk_TE
    r_over_t_TE = r_jk_TE / t_jk_TE

    D_TE = jnp.stack(
        [
            jnp.stack([t_inv_TE, r_over_t_TE], axis=-1),
            jnp.stack([r_over_t_TE, t_inv_TE], axis=-1),
        ],
        axis=-2,
    )  # Shape: (num_layers - 1, 2, 2)

    t_inv_TM = 1 / t_jk_TM
    r_over_t_TM = r_jk_TM / t_jk_TM

    D_TM = jnp.stack(
        [
            jnp.stack([t_inv_TM, r_over_t_TM], axis=-1),
            jnp.stack([r_over_t_TM, t_inv_TM], axis=-1),
        ],
        axis=-2,
    )

    exp_neg_jdelta = jnp.exp(-1j * delta[1:])
    exp_pos_jdelta = jnp.exp(1j * delta[1:])

    zeros = jnp.zeros_like(exp_neg_jdelta)

    P = jnp.stack(
        [
            jnp.stack([exp_neg_jdelta, zeros], axis=-1),
            jnp.stack([zeros, exp_pos_jdelta], axis=-1),
        ],
        axis=-2,
    )

    DP_TE = D_TE @ P
    DP_TM = D_TM @ P

    def matmul_scan(a, b):
        return jnp.matmul(a, b)

    M_TE = lax.associative_scan(matmul_scan, DP_TE)[-1]
    M_TM = lax.associative_scan(matmul_scan, DP_TM)[-1]

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

    stackrt_eps_mu_base_partial = partial(
        stackrt_eps_mu_base, is_back_layer_PEC=is_back_layer_PEC
    )
    fun_mapped = jax.vmap(
        stackrt_eps_mu_base_partial, (0, 0, None, 0, None), (0, 0, 0, 0)
    )

    r_TE, t_TE, r_TM, t_TM = fun_mapped(eps_r, mu_r, d, f, theta_rad)

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


def stackrt_n_k_theta(
    refractive_indices, thicknesses, frequencies, theta, is_back_layer_PEC=False
):
    assert isinstance(refractive_indices, jnp.ndarray)
    assert isinstance(thicknesses, jnp.ndarray)
    assert isinstance(frequencies, jnp.ndarray)

    assert refractive_indices.ndim == 2
    assert thicknesses.ndim == 1
    assert frequencies.ndim == 1

    assert refractive_indices.shape[0] == frequencies.shape[0]
    assert refractive_indices.shape[1] == thicknesses.shape[0]

    eps_r = jnp.conj(refractive_indices**2)
    mu_r = jnp.ones_like(eps_r)
    theta_rad = jnp.radians(theta)

    stackrt_eps_mu_base_partial = partial(
        stackrt_eps_mu_base, is_back_layer_PEC=is_back_layer_PEC
    )
    fun_mapped = jax.vmap(
        stackrt_eps_mu_base_partial, (0, 0, None, 0, None), (0, 0, 0, 0)
    )
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

    if is_back_layer_PEC:
        T_TE = jnp.zeros_like(R_TE)
        T_TM = jnp.zeros_like(R_TM)

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
