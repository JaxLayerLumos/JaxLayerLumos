import numpy as np
import jax.numpy as jnp
import scipy.constants as scic

from jaxlayerlumos import stackrt
from jaxlayerlumos.utils_materials import get_n_k_surrounded_by_air
from jaxlayerlumos.utils_spectra import get_frequencies_visible_light
from jaxlayerlumos.utils_layers import get_thicknesses_surrounded_by_air, convert_nm_to_m


if __name__ == "__main__":
    frequencies = get_frequencies_visible_light()
    list_materials = [
        ['Ag'],
    ]
    list_thicknesses = [
        jnp.array([100.0]),
    ]

    for materials, thicknesses in zip(list_materials, list_thicknesses):
        assert len(materials) == thicknesses.shape[0]

        n_k = get_n_k_surrounded_by_air(materials, frequencies)
        thicknesses = convert_nm_to_m(get_thicknesses_surrounded_by_air(thicknesses))

        R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies)

        print(R_TE.shape)
        print(T_TE.shape)
        print(R_TM.shape)
        print(T_TM.shape)
