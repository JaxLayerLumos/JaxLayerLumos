import jax.numpy as jnp

from jaxlayerlumos.colors import color_matching_functions
from jaxlayerlumos.colors import illuminants


def get_cmfs(bx, str_color_space="cie1931"):
    assert isinstance(bx, jnp.ndarray)

    if str_color_space == "cie1931":
        cmfs = color_matching_functions.cmfs_cie1931
    else:
        raise ValueError

    cmfs = jnp.array(cmfs)
    assert cmfs.shape[1] == 4
    assert jnp.min(cmfs[:, 0]) <= jnp.min(bx)
    assert jnp.max(bx) <= jnp.max(cmfs[:, 0])

    cmfs_interpolated = jnp.concatenate(
        [
            bx[..., jnp.newaxis],
            jnp.interp(bx, cmfs[:, 0], cmfs[:, 1])[..., jnp.newaxis],
            jnp.interp(bx, cmfs[:, 0], cmfs[:, 2])[..., jnp.newaxis],
            jnp.interp(bx, cmfs[:, 0], cmfs[:, 3])[..., jnp.newaxis],
        ],
        axis=1,
    )

    return cmfs_interpolated


def get_illuminant(bx, str_illuminant="d65"):
    assert isinstance(bx, jnp.ndarray)

    if str_illuminant == "d65":
        illuminant = illuminants.illuminant_d65
    else:
        raise ValueError

    illuminant = jnp.array(illuminant)
    assert illuminant.shape[1] == 2
    assert jnp.min(illuminant[:, 0]) <= jnp.min(bx)
    assert jnp.max(bx) <= jnp.max(illuminant[:, 0])

    illuminant_interpolated = jnp.concatenate(
        [
            bx[..., jnp.newaxis],
            jnp.interp(bx, illuminant[:, 0], illuminant[:, 1])[..., jnp.newaxis],
        ],
        axis=1,
    )

    return illuminant_interpolated
