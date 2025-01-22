import numpy as onp
import jax.numpy as jnp

from jaxlayerlumos import jaxlayerlumos


def compute_properties_jaxlayerlumos(thicknesses, n_k, frequencies, angle_of_incidence):
    assert isinstance(thicknesses, onp.ndarray)
    assert isinstance(n_k, onp.ndarray)
    assert isinstance(frequencies, onp.ndarray)
    assert isinstance(angle_of_incidence, float)

    assert thicknesses.ndim == 1
    assert n_k.ndim == 2
    assert frequencies.ndim == 1

    assert thicknesses.shape[0] == n_k.shape[0]
    assert frequencies.shape[0] == n_k.shape[1]

    n_k = jnp.array(n_k).T
    thicknesses = jnp.array(thicknesses)
    frequencies = jnp.array(frequencies)

    R_TE, T_TE, R_TM, T_TM = jaxlayerlumos.stackrt_n_k(
        n_k, thicknesses, frequencies, jnp.array([angle_of_incidence])
    )

    R_TE = onp.array(R_TE)
    T_TE = onp.array(T_TE)
    R_TM = onp.array(R_TM)
    T_TM = onp.array(T_TM)

    R_TE = onp.squeeze(R_TE, axis=0)
    T_TE = onp.squeeze(T_TE, axis=0)
    R_TM = onp.squeeze(R_TM, axis=0)
    T_TM = onp.squeeze(T_TM, axis=0)

    return R_TE, R_TM, T_TE, T_TM
