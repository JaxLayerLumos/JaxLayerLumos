import jax.numpy as jnp
import numpy as onp

import colour

from jaxlayerlumos.colors import transform


def _test_one_XYZ(XYZ):
    xyY_colour = colour.XYZ_to_xyY(XYZ)
    xyY_jaxcolors = onp.array(transform.XYZ_to_xyY(jnp.array(XYZ)))

    onp.testing.assert_allclose(xyY_jaxcolors, xyY_colour)

def test_XYZ_to_xyY():
    num_tests = 10000
    random_state = onp.random.RandomState(42)

    for _ in range(0, num_tests):
        XYZ = random_state.uniform(low=0.0, high=1.0, size=(3, ))
        _test_one_XYZ(XYZ)
