import numpy as np
import jax.numpy as jnp
import scipy.constants as scic

from jaxlayerlumos import stackrt
from jaxlayerlumos.utils_materials import get_n_k_surrounded_by_air
from jaxlayerlumos.utils_spectra import get_frequencies_visible_light
from jaxlayerlumos.utils_layers import get_thicknesses_surrounded_by_air, convert_nm_to_m

from lumerical_stackrt_multi_layers import compute_properties_via_stackrt


if __name__ == "__main__":
    frequencies = get_frequencies_visible_light()
    list_materials = [
        ['Ag'],
        ['Ag'],
        ['Ag', 'Al', 'Ag'],
        ['TiO2', 'Ag', 'TiO2'],
    ]
    list_thicknesses = [
        jnp.array([100.0]),
        jnp.array([10.0]),
        jnp.array([10.0, 11.0, 12.0]),
        jnp.array([20.0, 5.0, 30.0]),
    ]
    angles = jnp.array([0.0, 45.0])
    angles = jnp.array([0.0])

    for materials, thicknesses in zip(list_materials, list_thicknesses):
        assert len(materials) == thicknesses.shape[0]

        n_k = get_n_k_surrounded_by_air(materials, frequencies)
        layers = convert_nm_to_m(get_thicknesses_surrounded_by_air(thicknesses))

        R_TE, T_TE, R_TM, T_TM = stackrt(n_k, layers, frequencies, thetas=angles)

        print(R_TE.shape)
        print(T_TE.shape)
        print(R_TM.shape)
        print(T_TM.shape)

        Rs, Rp, Ts, Tp = compute_properties_via_stackrt(
            np.array(layers),
            np.array(n_k).T,
            np.array(frequencies),
            angle_of_incidence=np.array(angles)
        )

        print(Rs.shape)
        print(Rp.shape)
        print(Ts.shape)
        print(Tp.shape)

        assert jnp.allclose(R_TE, Rs.T)
        assert jnp.allclose(R_TM, Rp.T)
        assert jnp.allclose(T_TE, Ts.T)
        assert jnp.allclose(T_TM, Tp.T)
