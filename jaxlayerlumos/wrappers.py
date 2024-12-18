import jax.numpy as jnp

from jaxlayerlumos.jaxlayerlumos import stackrt_n_k


def stackrt_n_k_0(n, d, f, m):
    return stackrt_n_k(n, d, f, jnp.array([0]), m)


def stackrt_n_k_45(n, d, f, m):
    return stackrt_n_k(n, d, f, jnp.array([45]), m)
