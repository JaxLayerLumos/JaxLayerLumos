import time
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos.jaxlayerlumos import stackrt_n_k as stackrt_new
from jaxlayerlumos.jaxlayerlumos_old import stackrt as stackrt_old
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units


def compare_stackrt_old_new(use_random_angle):
    wavelengths = jnp.linspace(300e-9, 900e-9, 10)
    frequencies = utils_units.get_light_speed() / wavelengths

    all_materials = utils_materials.get_all_materials()

    num_layerss = jnp.array([2, 4, 6, 8, 10])
    num_tests = 20

    random_state = onp.random.RandomState(42)

    times_new = []
    times_old = []

    counts_R_TE = 0
    counts_T_TE = 0
    counts_R_TM = 0
    counts_T_TM = 0
    counts_all = 0

    for num_layers in num_layerss:
        for _ in range(0, num_tests):
            materials = random_state.choice(all_materials, num_layers)

            if use_random_angle:
                angle = random_state.uniform(0.0, 89.9)
            else:
                angle = 0.0

            n_k_air = jnp.ones_like(frequencies, dtype=jnp.complex128)
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
                n_k, thicknesses, frequencies, angle
            )
            time_end_old = time.monotonic()

            time_start_new = time.monotonic()
            R_TE_new, T_TE_new, R_TM_new, T_TM_new = stackrt_new(
                n_k, thicknesses, frequencies, angle
            )
            time_end_new = time.monotonic()

            try:
                R_TE_old = onp.clip(R_TE_old, min=1e-8)
                T_TE_old = onp.clip(T_TE_old, min=1e-8)
                R_TM_old = onp.clip(R_TM_old, min=1e-8)
                T_TM_old = onp.clip(T_TM_old, min=1e-8)

                R_TE_new = onp.clip(R_TE_new, min=1e-8)
                T_TE_new = onp.clip(T_TE_new, min=1e-8)
                R_TM_new = onp.clip(R_TM_new, min=1e-8)
                T_TM_new = onp.clip(T_TM_new, min=1e-8)
            except:
                R_TE_old = onp.clip(R_TE_old, a_min=1e-8)
                T_TE_old = onp.clip(T_TE_old, a_min=1e-8)
                R_TM_old = onp.clip(R_TM_old, a_min=1e-8)
                T_TM_old = onp.clip(T_TM_old, a_min=1e-8)

                R_TE_new = onp.clip(R_TE_new, a_min=1e-8)
                T_TE_new = onp.clip(T_TE_new, a_min=1e-8)
                R_TM_new = onp.clip(R_TM_new, a_min=1e-8)
                T_TM_new = onp.clip(T_TM_new, a_min=1e-8)

            str_thicknesses = [f"{thickness:.4f}" for thickness in thicknesses[1:]]
            print(f":materials: [{', '.join(materials)}]")
            print(f":thicknesses: [{', '.join(str_thicknesses)}]")
            print(f":angle: {angle:.4f}")

            is_close_R_TE = onp.allclose(R_TE_old, R_TE_new, rtol=1e-5)
            is_close_T_TE = onp.allclose(T_TE_old, T_TE_new, rtol=1e-5)
            is_close_R_TM = onp.allclose(R_TM_old, R_TM_new, rtol=1e-5)
            is_close_T_TM = onp.allclose(T_TM_old, T_TM_new, rtol=1e-5)

            if not is_close_R_TE:
                counts_R_TE += 1
            if not is_close_T_TE:
                counts_T_TE += 1
            if not is_close_R_TM:
                counts_R_TM += 1
            if not is_close_T_TM:
                counts_T_TM += 1
            counts_all += 1

            time_consumed_new = time_end_new - time_start_new
            time_consumed_old = time_end_old - time_start_old

            print(f":new: {time_consumed_new:.4f} sec.")
            print(f":old: {time_consumed_old:.4f} sec.")
            if not (is_close_R_TE and is_close_T_TE and is_close_R_TM and is_close_T_TM):
                print("=====NOT MATCHED=====")

            print("")

            times_new.append(time_consumed_new)
            times_old.append(time_consumed_old)

    if onp.mean(times_new) > onp.mean(times_new):
        assert False

    if counts_R_TE > 0 or counts_T_TE > 0 or counts_R_TM > 0 or counts_T_TM > 0:
        print('failure ratios')
        print(f'R_TE {counts_R_TE / counts_all:.4f} T_TE {counts_T_TE / counts_all:.4f}')
        print(f'R_TM {counts_R_TM / counts_all:.4f} T_TM {counts_T_TM / counts_all:.4f}')

        assert False


def test_comparison_stackrt_old_new_zero_angle():
    use_random_angle = False

    compare_stackrt_old_new(use_random_angle)


def test_comparison_stackrt_old_new_nonzero_angle():
    use_random_angle = True

    compare_stackrt_old_new(use_random_angle)
