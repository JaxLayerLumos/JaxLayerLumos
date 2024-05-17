import pytest
import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos import utils_layers


def test_get_thicknesses_surrounded_by_air():
    with pytest.raises(AssertionError):
        utils_layers.get_thicknesses_surrounded_by_air("abc")
    with pytest.raises(AssertionError):
        utils_layers.get_thicknesses_surrounded_by_air(1234)
    with pytest.raises(AssertionError):
        utils_layers.get_thicknesses_surrounded_by_air(np.array([1, 2, 3]))
    with pytest.raises(AssertionError):
        utils_layers.get_thicknesses_surrounded_by_air(jnp.array(1.0))
    with pytest.raises(AssertionError):
        utils_layers.get_thicknesses_surrounded_by_air(
            jnp.array([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]])
        )

    thicknesses = jnp.array([10.0, 20.0, 4.0, 1.0, 2.0])
    thicknesses_surrounded_by_air = utils_layers.get_thicknesses_surrounded_by_air(
        thicknesses
    )

    assert thicknesses_surrounded_by_air.ndim == 1
    assert thicknesses_surrounded_by_air.shape[0] == thicknesses.shape[0] + 2
    assert thicknesses_surrounded_by_air[0] == thicknesses_surrounded_by_air[-1] == 0.0

    np.testing.assert_allclose(thicknesses_surrounded_by_air[1:-1], thicknesses)


def test_convert_nm_to_m():
    thicknesses = jnp.array([10.0, 20.0, 4.0, 1.0, 2.0])
    thicknesses_in_m = utils_layers.convert_nm_to_m(thicknesses)

    assert thicknesses_in_m.ndim == 1
    assert thicknesses_in_m.shape[0] == thicknesses.shape[0]

    np.testing.assert_allclose(thicknesses_in_m, thicknesses * scic.nano)

    thicknesses = jnp.array(1000.0)
    thicknesses_in_m = utils_layers.convert_nm_to_m(thicknesses)

    np.testing.assert_allclose(thicknesses_in_m, thicknesses * scic.nano)

    thickness = 1000.0
    thickness_in_m = utils_layers.convert_nm_to_m(thickness)

    np.testing.assert_allclose(thickness_in_m, thickness * scic.nano)
