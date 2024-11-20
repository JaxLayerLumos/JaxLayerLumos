import jax
import jax.numpy as jnp

from jaxlayerlumos import utils_spectra

jax.config.update("jax_enable_x64", True)


def stackrt_eps_mu(eps_r, mu_r, d, f, theta):

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
        stackrt_eps_mu_base, (0, 0, None, 0, None), (0, 0, 0, 0, 0, 0)
    )
    r_TE, t_TE, r_TM, t_TM, thetas_k, cos_thetas_t = fun_mapped(
        eps_r, mu_r, d, f, theta_rad
    )

    n = jnp.conj(jnp.sqrt(eps_r * mu_r))

    R_TE = jnp.abs(r_TE) ** 2
    T_TE = jnp.abs(t_TE) ** 2 * jnp.real(
        n[:, -1] * jnp.cos(thetas_k) / (n[:, 0] * cos_thetas_t)
    )
    R_TM = jnp.abs(r_TM) ** 2
    T_TM = jnp.abs(t_TM) ** 2 * jnp.real(
        n[:, -1] * jnp.cos(thetas_k) / (n[:, 0] * cos_thetas_t)
    )

    return R_TE, T_TE, R_TM, T_TM


def stackrt_eps_mu_base(eps_r, mu_r, d, f_i, theta_k):
    assert isinstance(eps_r, jnp.ndarray)
    assert isinstance(mu_r, jnp.ndarray)
    assert isinstance(d, jnp.ndarray)
    assert eps_r.ndim == 1
    assert d.ndim == 1
    assert eps_r.shape[0] == d.shape[0]

    assert mu_r.ndim == 1
    assert d.ndim == 1
    assert mu_r.shape[0] == d.shape[0]

    # Convert frequency from GHz to Hz
    f = f_i * 1e9

    # Convert slab thickness from mm to m
    d = d * 1e-3

    num_layers = len(d)

    c = 299792458  # Speed of light in m/s
    k = 2 * jnp.pi / c * f * jnp.conj(jnp.sqrt(eps_r * mu_r))
    eta = jnp.conj(jnp.sqrt(mu_r / eps_r))

    sin_theta_layer = [jnp.sin(jnp.radians(theta_k))]
    sin_theta = [sin_theta_layer]
    for j in range(num_layers - 1):
        sin_theta_layer = (k[j] * sin_theta_layer) / k[j + 1]
        sin_theta.append(sin_theta_layer)

    cos_theta_t = jnp.sqrt(1 - sin_theta**2)
    kz = k * cos_theta_t

    upper_bound = 600.0
    delta = d[:, jnp.newaxis] * kz

    delta = jnp.real(delta) + 1j * jnp.clip(jnp.imag(delta), -upper_bound, upper_bound)

    Z_TE = eta / cos_theta_t
    Z_TM = eta * cos_theta_t

    M_TE = jnp.repeat(jnp.eye(2)[..., jnp.newaxis], len(f), axis=2)
    M_TM = jnp.repeat(jnp.eye(2)[..., jnp.newaxis], len(f), axis=2)

    for j in range(num_layers - 1):

        r_jk_TE = (Z_TE[j + 1, :] - Z_TE[j, :]) / (Z_TE[j + 1, :] + Z_TE[j, :])
        t_jk_TE = (2 * Z_TE[j + 1, :]) / (Z_TE[j + 1, :] + Z_TE[j, :])

        r_jk_TM = (Z_TM[j + 1, :] - Z_TM[j, :]) / (Z_TM[j + 1, :] + Z_TM[j, :])
        t_jk_TM = (2 * Z_TM[j + 1, :]) / (Z_TM[j + 1, :] + Z_TM[j, :])

        if j == num_layers - 2 and eps_r == jnp.inf * 1j:
            # check for PEC
            r_jk_TE = -jnp.ones(len(f))
            t_jk_TE = jnp.ones(len(f))
            r_jk_TM = -jnp.ones(len(f))
            t_jk_TM = jnp.ones(len(f))

        D_jk_TE = jnp.zeros((2, 2, len(f)), dtype=complex)
        D_jk_TE[0, 0, :] = 1
        D_jk_TE[1, 1, :] = 1
        D_jk_TE[0, 1, :] = r_jk_TE
        D_jk_TE[1, 0, :] = r_jk_TE
        D_jk_TE /= t_jk_TE[jnp.newaxis, jnp.newaxis, :]

        D_jk_TM = jnp.zeros((2, 2, len(f)), dtype=complex)
        D_jk_TM[0, 0, :] = 1
        D_jk_TM[1, 1, :] = 1
        D_jk_TM[0, 1, :] = r_jk_TM
        D_jk_TM[1, 0, :] = r_jk_TM
        D_jk_TM /= t_jk_TM[jnp.newaxis, jnp.newaxis, :]

        P = jnp.zeros((2, 2, len(f)), dtype=complex)
        P[0, 0, :] = jnp.exp(-1j * delta[j + 1, :])
        P[1, 1, :] = jnp.exp(1j * delta[j + 1, :])

        M_TE = jnp.einsum("ijk,jlk->ilk", M_TE, jnp.einsum("ijk,jlk->ilk", D_jk_TE, P))
        M_TM = jnp.einsum("ijk,jlk->ilk", M_TM, jnp.einsum("ijk,jlk->ilk", D_jk_TM, P))

    r_TE_i = M_TE[1, 0, :] / M_TE[0, 0, :]
    t_TE_i = 1 / M_TE[0, 0, :]

    r_TM_i = M_TM[1, 0, :] / M_TM[0, 0, :]
    t_TM_i = 1 / M_TM[0, 0, :]

    return r_TE_i, t_TE_i, r_TM_i, t_TM_i, theta_k, cos_theta_t


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
