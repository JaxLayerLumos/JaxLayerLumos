import jax.numpy as jnp
import scipy.constants as scic


def get_thicknesses_surrounded_by_air(thicknesses):
    assert isinstance(thicknesses, jnp.ndarray)
    assert thicknesses.ndim == 1

    return jnp.concatenate([jnp.array([0.0]), thicknesses, jnp.array([0.0])], axis=0)


def convert_nm_to_m(thicknesses):
    return thicknesses * scic.nano


def convert_mm_to_m(thicknesses):
    return thicknesses * scic.milli


def convert_cm_to_m(thicknesses):
    return thicknesses * scic.centi
