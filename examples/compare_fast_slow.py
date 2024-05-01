import numpy as np
import time
import jax.numpy as jnp
from jaxlayerlumos.jaxlayerlumos import stackrt as stackrt_slow
from jaxlayerlumos.jaxlayerlumos_fast import stackrt as stackrt_fast

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

    for materials, thicknesses in zip(list_materials, list_thicknesses):
        assert len(materials) == thicknesses.shape[0]

        n_k = get_n_k_surrounded_by_air(materials, frequencies)
        layers = convert_nm_to_m(get_thicknesses_surrounded_by_air(thicknesses))

        time_start_fast = time.monotonic()
        R_TE_fast, T_TE_fast, R_TM_fast, T_TM_fast = stackrt_fast(
            n_k, layers, frequencies, thetas=angles
        )
        time_end_fast = time.monotonic()

        time_start_slow = time.monotonic()
        R_TE_slow, T_TE_slow, R_TM_slow, T_TM_slow = stackrt_slow(
            n_k, layers, frequencies, thetas=angles
        )
        time_end_slow = time.monotonic()

        print(f'fast {time_end_fast - time_start_fast} slow {time_end_slow - time_start_slow}')

        np.testing.assert_allclose(R_TE_fast, R_TE_slow)
        np.testing.assert_allclose(R_TM_fast, R_TM_slow)
        np.testing.assert_allclose(T_TE_fast, T_TE_slow)
        np.testing.assert_allclose(T_TM_fast, T_TM_slow)
