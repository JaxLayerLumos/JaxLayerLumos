#
# This file will be deleted after testing our new implementation.
# Keep this file for a while.
#

import jax
import jax.numpy as jnp

from jaxlayerlumos import utils_spectra

jax.config.update("jax_enable_x64", True)


def stackrt_base(n_i, d, wvl_i, theta_k):
    assert isinstance(n_i, jnp.ndarray)
    assert isinstance(d, jnp.ndarray)
    assert n_i.ndim == 1
    assert d.ndim == 1
    assert n_i.shape[0] == d.shape[0]

    M_TE = jnp.eye(2, dtype=jnp.complex128)
    M_TM = jnp.eye(2, dtype=jnp.complex128)

    for j in range(0, n_i.shape[0] - 1):
        n_current = n_i[j]
        n_next = n_i[j + 1]
        d_next = d[j + 1]

        sin_theta_t = n_current * jnp.sin(theta_k) / n_next
        theta_t = jnp.arcsin(sin_theta_t)
        cos_theta_k = jnp.cos(theta_k)
        cos_theta_t = jnp.cos(theta_t)

        r_jk_TE = (n_current * cos_theta_k - n_next * cos_theta_t) / (
            n_current * cos_theta_k + n_next * cos_theta_t
        )
        t_jk_TE = (
            2
            * n_current
            * cos_theta_k
            / (n_current * cos_theta_k + n_next * cos_theta_t)
        )
        r_jk_TM = (n_next * cos_theta_k - n_current * cos_theta_t) / (
            n_next * cos_theta_k + n_current * cos_theta_t
        )
        t_jk_TM = (
            2
            * n_current
            * cos_theta_k
            / (n_next * cos_theta_k + n_current * cos_theta_t)
        )

        M_jk_TE = jnp.array(
            [[1 / t_jk_TE, r_jk_TE / t_jk_TE], [r_jk_TE / t_jk_TE, 1 / t_jk_TE]],
            dtype=jnp.complex128,
        )
        M_jk_TM = jnp.array(
            [[1 / t_jk_TM, r_jk_TM / t_jk_TM], [r_jk_TM / t_jk_TM, 1 / t_jk_TM]],
            dtype=jnp.complex128,
        )

        upper_bound = 600.0  # Jungtaek: I manually chose this number by testing a sufficient number of structures.  It might be fixed in future.

        delta = 2 * jnp.pi * n_next * d_next * cos_theta_t / wvl_i
        delta = jnp.real(delta) + 1j * jnp.clip(
            jnp.imag(delta), -upper_bound, upper_bound
        )
        P = jnp.array(
            [[jnp.exp(-1j * delta), 0], [0, jnp.exp(1j * delta)]],
            dtype=jnp.complex128,
        )
        M_TE = jnp.dot(M_TE, jnp.dot(M_jk_TE, P))
        M_TM = jnp.dot(M_TM, jnp.dot(M_jk_TM, P))

        theta_k = theta_t

    r_TE_i = M_TE[1, 0] / M_TE[0, 0]
    t_TE_i = 1 / M_TE[0, 0]
    r_TM_i = M_TM[1, 0] / M_TM[0, 0]
    t_TM_i = 1 / M_TM[0, 0]

    return r_TE_i, t_TE_i, r_TM_i, t_TM_i, theta_k, cos_theta_t


def stackrt_theta(n, d, f, theta):
    """
    Calculate the reflection and transmission coefficients for a multilayer stack
    at different frequencies under an arbitrary angle of incidence.

    :param n: The refractive indices of the layers for each frequency.
              Shape should be (Nfreq, Nlayers), where Nfreq is the number of
              frequencies and Nlayers is the number of layers.
    :param d: The thicknesses of the layers. Shape should be (Nlayers,).
    :param f: The frequencies at which to calculate the coefficients.
              Shape should be (Nfreq,).
    :param theta: The incident angle in degrees. Defaults to 0 for normal incidence.
    :returns: A tuple containing:
              - R_TE (numpy.ndarray): Reflectance for TE polarization. Shape is (Nfreq,).
              - T_TE (numpy.ndarray): Transmittance for TE polarization. Shape is (Nfreq,).
              - R_TM (numpy.ndarray): Reflectance for TM polarization. Shape is (Nfreq,).
              - T_TM (numpy.ndarray): Transmittance for TM polarization. Shape is (Nfreq,).

    """

    assert isinstance(n, jnp.ndarray)
    assert isinstance(d, jnp.ndarray)
    assert isinstance(f, jnp.ndarray)
    assert n.ndim == 2
    assert d.ndim == 1
    assert f.ndim == 1
    assert n.shape[0] == f.shape[0]
    assert n.shape[1] == d.shape[0]

    wvl = utils_spectra.convert_frequencies_to_wavelengths(f)
    theta_rad = jnp.radians(theta)

    fun_mapped = jax.vmap(stackrt_base, (0, None, 0, None), (0, 0, 0, 0, 0, 0))
    r_TE, t_TE, r_TM, t_TM, thetas_k, cos_thetas_t = fun_mapped(n, d, wvl, theta_rad)

    R_TE = jnp.abs(r_TE) ** 2
    T_TE = jnp.abs(t_TE) ** 2 * jnp.real(
        n[:, -1] * jnp.cos(thetas_k) / (n[:, 0] * cos_thetas_t)
    )
    R_TM = jnp.abs(r_TM) ** 2
    T_TM = jnp.abs(t_TM) ** 2 * jnp.real(
        n[:, -1] * jnp.cos(thetas_k) / (n[:, 0] * cos_thetas_t)
    )

    return R_TE, T_TE, R_TM, T_TM


def stackrt(n, d, f, thetas=None):
    """
    Calculate the reflection and transmission coefficients for a multilayer stack
    at different frequencies and incidence angles, adapted for JAX.

    Parameters:
    - n: The refractive indices of the layers for each frequency.
         Shape should be (Nfreq, Nlayers), where Nfreq is the number of frequencies and Nlayers is the number of layers.
    - d: The thicknesses of the layers. Shape should be (Nlayers,).
    - f: The frequencies at which to calculate the coefficients. Shape should be (Nfreq,).
    - thetas: The incidence angle(s) in degrees. Can be a single value or an array of angles. Defaults to [0].

    Returns:
    - A tuple containing:
      - R_TE (jax.numpy.ndarray): Reflectance for TE polarization. Shape is (Nfreq,).
      - T_TE (jax.numpy.ndarray): Transmittance for TE polarization. Shape is (Nfreq,).
      - R_TM (jax.numpy.ndarray): Reflectance for TM polarization. Shape is (Nfreq,).
      - T_TM (jax.numpy.ndarray): Transmittance for TM polarization. Shape is (Nfreq,).

    """

    if thetas is None:
        thetas = jnp.array([0])
    elif isinstance(thetas, (float, int)):
        thetas = jnp.array([thetas])

    fun_mapped = jax.vmap(stackrt_theta, (None, None, None, 0), (0, 0, 0, 0))
    R_TE, T_TE, R_TM, T_TM = fun_mapped(n, d, f, thetas)

    return R_TE, T_TE, R_TM, T_TM


def stackrt0(n, d, f):
    return stackrt(n, d, f, thetas=jnp.array([0]))


def stackrt45(n, d, f):
    return stackrt(n, d, f, thetas=jnp.array([45]))
