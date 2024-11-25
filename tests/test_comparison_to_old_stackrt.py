import time
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos.jaxlayerlumos import stackrt_n_k as stackrt_new
from jaxlayerlumos.jaxlayerlumos_old import stackrt as stackrt_old
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units


def test_stackrt():
    wavelengths = jnp.linspace(300e-9, 900e-9, 3)
    frequencies = utils_units.get_light_speed() / wavelengths

    all_materials = utils_materials.get_all_materials()

    num_layerss = jnp.array([2, 4, 6, 8, 10])
    num_tests = 100

    random_state = onp.random.RandomState(42)

    for num_layers in num_layerss:
        for _ in range(0, num_tests):
            materials = random_state.choice(all_materials, num_layers)

            n_k_air = jnp.ones_like(frequencies)
            thickness_air = 0.0

            n_k = [n_k_air]
            thicknesses = [thickness_air]

            for material in materials:
                n_material, k_material = utils_materials.interpolate_material_n_k(material, frequencies)
                n_k_material = n_material + 1j * k_material
                thickness_material = random_state.uniform(0.01, 10.0)

                n_k.append(n_k_material)
                thicknesses.append(thickness_material)

            n_k = jnp.array(n_k).T
            thicknesses = jnp.array(thicknesses)

            R_TE_old, T_TE_old, R_TM_old, T_TM_old = stackrt_old(n_k, thicknesses, frequencies, 0.0)
            R_TE_new, T_TE_new, R_TM_new, T_TM_new = stackrt_new(n_k, thicknesses, frequencies, 0.0)

            onp.testing.assert_allclose(R_TE_old, R_TE_new, rtol=1e-5)
#            onp.testing.assert_allclose(T_TE_old, T_TE_new, rtol=1e-5)
            onp.testing.assert_allclose(R_TM_old, R_TM_new, rtol=1e-5)
#            onp.testing.assert_allclose(T_TM_old, T_TM_new, rtol=1e-5)
