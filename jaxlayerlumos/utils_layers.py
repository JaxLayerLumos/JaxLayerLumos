import jax.numpy as jnp


def get_thicknesses_surrounded_by_air(thicknesses):
    assert isinstance(thicknesses, jnp.ndarray)
    assert thicknesses.ndim == 1

    return jnp.concatenate([jnp.array([0.0]), thicknesses, jnp.array([0.0])], axis=0)
