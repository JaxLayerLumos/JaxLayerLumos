import numpy as np
import jax.numpy as jnp
import scipy.constants as scic
from jaxlayerlumos import stackrt

from jaxlayerlumos.utils_materials import get_n_k_surrounded_by_air
from jaxlayerlumos.utils_spectra import get_frequencies_visible_light
from jaxlayerlumos.utils_layers import (
    get_thicknesses_surrounded_by_air,
    convert_nm_to_m,
)

from lumerical_stackrt_multi_layers import compute_properties_via_stackrt
from plot_spectra import plot_spectra


if __name__ == "__main__":
    save_figure = True

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

            Rs, Rp, Ts, Tp = compute_properties_via_stackrt(
                np.array(layers),
                np.array(n_k).T,
                np.array(frequencies),
                angle_of_incidence=np.array([angle]),
            )

            R_TE_jll = np.squeeze(np.array(R_TE), axis=0)
            R_TM_jll = np.squeeze(np.array(R_TM), axis=0)
            T_TE_jll = np.squeeze(np.array(T_TE), axis=0)
            T_TM_jll = np.squeeze(np.array(T_TM), axis=0)

            R_TE_lum = np.squeeze(np.array(Rs), axis=1)
            R_TM_lum = np.squeeze(np.array(Rp), axis=1)
            T_TE_lum = np.squeeze(np.array(Ts), axis=1)
            T_TM_lum = np.squeeze(np.array(Tp), axis=1)

            assert np.allclose(R_TE_jll, R_TE_lum)
            assert np.allclose(R_TM_jll, R_TM_lum)
            assert np.allclose(T_TE_jll, T_TE_lum)
            assert np.allclose(T_TM_jll, T_TM_lum)

            if save_figure:
                str_file = f"tmm_{'_'.join(f'{mat}_{thick}nm' for mat, thick in zip(materials, thicknesses))}_angle_{angle}_deg"

                plot_spectra(
                    frequencies,
                    np.array(
                        [
                            R_TE_jll,
                            R_TE_lum,
                        ]
                    ),
                    np.array(
                        [
                            R_TM_jll,
                            R_TM_lum,
                        ]
                    ),
                    np.array(
                        [
                            T_TE_jll,
                            T_TE_lum,
                        ]
                    ),
                    np.array(
                        [
                            T_TM_jll,
                            T_TM_lum,
                        ]
                    ),
                    ["Lumerical", "JaxLayerLumos"],
                    ["-", "--"],
                    str_file,
                )
