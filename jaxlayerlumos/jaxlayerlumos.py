import jax
import jax.numpy as jnp
from .utils_spectra import convert_frequencies_to_wavelengths

jax.config.update("jax_enable_x64", True)


def stackrt_theta(n, d, f, theta=0):
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

    wvl = convert_frequencies_to_wavelengths(f)
    theta_rad = jnp.radians(theta)

    r_TE = jnp.zeros_like(f, dtype=jnp.complex128)
    r_TM = jnp.zeros_like(f, dtype=jnp.complex128)
    t_TE = jnp.zeros_like(f, dtype=jnp.complex128)
    t_TM = jnp.zeros_like(f, dtype=jnp.complex128)
    R_TE = jnp.zeros_like(f)
    T_TE = jnp.zeros_like(f)
    R_TM = jnp.zeros_like(f)
    T_TM = jnp.zeros_like(f)

    for i in range(len(f)):
        lambda_i = wvl[i]
        M_TE, M_TM = jnp.eye(2, dtype=jnp.complex128), jnp.eye(2, dtype=jnp.complex128)
        theta_i = theta_rad

        for j in range(n.shape[1] - 1):
            n_j, n_next, d_next = n[i, j], n[i, j + 1], d[j + 1]
            sin_theta_t = n_j * jnp.sin(theta_i) / n_next
            theta_t = jnp.arcsin(sin_theta_t)

            cos_theta_i, cos_theta_t = jnp.cos(theta_i), jnp.cos(theta_t)
            r_jk_TE = (n_j * cos_theta_i - n_next * cos_theta_t) / (
                n_j * cos_theta_i + n_next * cos_theta_t
            )
            t_jk_TE = 2 * n_j * cos_theta_i / (n_j * cos_theta_i + n_next * cos_theta_t)
            r_jk_TM = (n_next * cos_theta_i - n_j * cos_theta_t) / (
                n_next * cos_theta_i + n_j * cos_theta_t
            )
            t_jk_TM = 2 * n_j * cos_theta_i / (n_next * cos_theta_i + n_j * cos_theta_t)

            M_jk_TE = jnp.array(
                [[1 / t_jk_TE, r_jk_TE / t_jk_TE], [r_jk_TE / t_jk_TE, 1 / t_jk_TE]],
                dtype=jnp.complex128,
            )
            M_jk_TM = jnp.array(
                [[1 / t_jk_TM, r_jk_TM / t_jk_TM], [r_jk_TM / t_jk_TM, 1 / t_jk_TM]],
                dtype=jnp.complex128,
            )

            delta = 2 * jnp.pi * n_next * d_next * cos_theta_t/ lambda_i
            P = jnp.array(
                [[jnp.exp(-1j * delta), 0], [0, jnp.exp(1j * delta)]],
                dtype=jnp.complex128,
            )
            M_TE = jnp.dot(M_TE, jnp.dot(M_jk_TE, P))
            M_TM = jnp.dot(M_TM, jnp.dot(M_jk_TM, P))

            theta_i = theta_t

        # Use .at[].set() for updating JAX arrays
        r_TE = r_TE.at[i].set(M_TE[1, 0] / M_TE[0, 0])
        t_TE = t_TE.at[i].set(1 / M_TE[0, 0])
        r_TM = r_TM.at[i].set(M_TM[1, 0] / M_TM[0, 0])
        t_TM = t_TM.at[i].set(1 / M_TM[0, 0])

        R_TE = R_TE.at[i].set(jnp.abs(r_TE[i]) ** 2)
        T_TE = T_TE.at[i].set(
            jnp.abs(t_TE[i]) ** 2
            * jnp.real(n[i, -1] * jnp.cos(theta_rad) / (n[i, 0] * cos_theta_t))
        )
        R_TM = R_TM.at[i].set(jnp.abs(r_TM[i]) ** 2)
        T_TM = T_TM.at[i].set(
            jnp.abs(t_TM[i]) ** 2
            * jnp.real(n[i, -1] * jnp.cos(theta_rad) / (n[i, 0] * cos_theta_t))
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
    elif isinstance(
        thetas, (float, jnp.float32, jnp.float64, int, jnp.int32, jnp.int64)
    ):
        thetas = jnp.array([thetas])

    Nfreq, Ntheta = len(f), len(thetas)
    R_TE = jnp.zeros((Ntheta, Nfreq))
    T_TE = jnp.zeros((Ntheta, Nfreq))
    R_TM = jnp.zeros((Ntheta, Nfreq))
    T_TM = jnp.zeros((Ntheta, Nfreq))

    for i, angle in enumerate(thetas):
        R_TE_i, T_TE_i, R_TM_i, T_TM_i = stackrt_theta(n, d, f, angle)

        R_TE = R_TE.at[i, :].set(R_TE_i)
        T_TE = T_TE.at[i, :].set(T_TE_i)
        R_TM = R_TM.at[i, :].set(R_TM_i)
        T_TM = T_TM.at[i, :].set(T_TM_i)

    return R_TE, T_TE, R_TM, T_TM


def stackrt0(n, d, f):
    return stackrt(n, d, f, thetas=jnp.array([0]))


def stackrt45(n, d, f):
    return stackrt(n, d, f, thetas=jnp.array([45]))
