import time
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos.jaxlayerlumos import stackrt_n_k as stackrt_new
from jaxlayerlumos.jaxlayerlumos_old import stackrt as stackrt_old
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units


def test_comparison_stackrt_old_new():
    wavelengths = jnp.linspace(300e-9, 900e-9, 3)
    frequencies = utils_units.get_light_speed() / wavelengths

    all_materials = utils_materials.get_all_materials()

    num_layerss = jnp.array([2, 4, 6, 8, 10])
    num_tests = 20

    random_state = onp.random.RandomState(42)

    times_new = []
    times_old = []

    for num_layers in num_layerss:
        for _ in range(0, num_tests):
            materials = random_state.choice(all_materials, num_layers)

            n_k_air = jnp.ones_like(frequencies)
            thickness_air = 0.0

            n_k = [n_k_air]
            thicknesses = [thickness_air]

            for material in materials:
                n_material, k_material = utils_materials.interpolate_material_n_k(
                    material, frequencies
                )
                n_k_material = n_material + 1j * k_material
                thickness_material = random_state.uniform(0.01, 10.0)

                n_k.append(n_k_material)
                thicknesses.append(thickness_material)

            n_k = jnp.array(n_k).T
            thicknesses = jnp.array(thicknesses)

            time_start_old = time.monotonic()
            R_TE_old, T_TE_old, R_TM_old, T_TM_old = stackrt_old(
                n_k, thicknesses, frequencies, 0.0
            )
            time_end_old = time.monotonic()

            time_start_new = time.monotonic()
            R_TE_new, T_TE_new, R_TM_new, T_TM_new = stackrt_new(
                n_k, thicknesses, frequencies, 0.0
            )
            time_end_new = time.monotonic()

            R_TE_old = onp.clip(R_TE_old, a_min=1e-8, a_max = None)
            T_TE_old = onp.clip(T_TE_old, a_min=1e-8, a_max = None)
            R_TM_old = onp.clip(R_TM_old, a_min=1e-8, a_max = None)
            T_TM_old = onp.clip(T_TM_old, a_min=1e-8, a_max = None)

            R_TE_new = onp.clip(R_TE_new, a_min=1e-8, a_max = None)
            T_TE_new = onp.clip(T_TE_new, a_min=1e-8, a_max = None)
            R_TM_new = onp.clip(R_TM_new, a_min=1e-8, a_max = None)
            T_TM_new = onp.clip(T_TM_new, a_min=1e-8, a_max = None)

            print(f":materials: {materials}")
            print(f":thicknesses: {thicknesses}")

            onp.testing.assert_allclose(R_TE_old, R_TE_new)
            onp.testing.assert_allclose(T_TE_old, T_TE_new)
            onp.testing.assert_allclose(R_TM_old, R_TM_new)
            onp.testing.assert_allclose(T_TM_old, T_TM_new)

            time_consumed_new = time_end_new - time_start_new
            time_consumed_old = time_end_old - time_start_old

            print(f":new: {time_consumed_new}")
            print(f":old: {time_consumed_old}")

            times_new.append(time_consumed_new)
            times_old.append(time_consumed_old)

    if onp.mean(times_new) > onp.mean(times_new):
        assert False
