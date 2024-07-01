import numpy as np
import jax.numpy as jnp
from jaxlayerlumos import stackrt

from jaxlayerlumos.utils_materials import get_n_k_surrounded_by_air
from jaxlayerlumos.utils_spectra import get_frequencies_visible_light
from jaxlayerlumos.utils_layers import (
    get_thicknesses_surrounded_by_air,
    convert_nm_to_m,
)


if __name__ == "__main__":
    frequencies = get_frequencies_visible_light()
    list_materials = [
        ["Ag"],
        ["Ag"],
        ["Ag", "Al", "Ag"],
        ["TiO2", "Ag", "TiO2"],
    ]
    list_thicknesses = [
        jnp.array([100.0]),
        jnp.array([10.0]),
        jnp.array([10.0, 11.0, 12.0]),
        jnp.array([20.0, 5.0, 30.0]),
    ]
    angles = jnp.array([0.0, 45.0, 75.0, 89.0])

    for angle in angles:
        for materials, thicknesses in zip(list_materials, list_thicknesses):
            assert len(materials) == thicknesses.shape[0]

            n_k = get_n_k_surrounded_by_air(materials, frequencies)
            layers = convert_nm_to_m(get_thicknesses_surrounded_by_air(thicknesses))

            R_TE, T_TE, R_TM, T_TM = stackrt(
                n_k, layers, frequencies, thetas=jnp.array([angle])
            )

            R_TE_jll = np.squeeze(np.array(R_TE), axis=0)
            R_TM_jll = np.squeeze(np.array(R_TM), axis=0)
            T_TE_jll = np.squeeze(np.array(T_TE), axis=0)
            T_TM_jll = np.squeeze(np.array(T_TM), axis=0)

            print(R_TE_jll.shape)
            print(R_TM_jll.shape)
            print(T_TE_jll.shape)
            print(T_TM_jll.shape)
