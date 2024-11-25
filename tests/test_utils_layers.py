import pytest
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import utils_layers


def test_get_thicknesses_surrounded_by_air():
    with pytest.raises(AssertionError):
        utils_layers.get_thicknesses_surrounded_by_air("abc")
    with pytest.raises(AssertionError):
        utils_layers.get_thicknesses_surrounded_by_air(1234)
    with pytest.raises(AssertionError):
        utils_layers.get_thicknesses_surrounded_by_air(onp.array([1, 2, 3]))
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

    onp.testing.assert_allclose(thicknesses_surrounded_by_air[1:-1], thicknesses)
