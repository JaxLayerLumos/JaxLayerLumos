import time
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units

import method_ansys
import method_jaxlayerlumos_old
import method_jaxlayerlumos


def run_simulation(method, frequencies, thicknesses, n_k, angle):
    time_start = time.monotonic()

    if method == 'ansys':
        R_TE, R_TM, T_TE, T_TM = method_ansys.compute_properties_ansys(
            thicknesses, n_k, frequencies, angle)
    elif method == 'jaxlayerlumos_old':
        R_TE, R_TM, T_TE, T_TM = method_jaxlayerlumos_old.compute_properties_jaxlayerlumos_old(
            thicknesses, n_k, frequencies, angle)
    elif method == 'jaxlayerlumos':
        R_TE, R_TM, T_TE, T_TM = method_jaxlayerlumos.compute_properties_jaxlayerlumos(
            thicknesses, n_k, frequencies, angle)
    else:
        raise ValueError

    time_end = time.monotonic()
    time_consumed = time_end - time_start

    return R_TE, R_TM, T_TE, T_TM, time_consumed


def compare_simulation(frequencies, thicknesses, n_k, angle):
    methods = onp.array([
        'ansys',
        'jaxlayerlumos_old',
        'jaxlayerlumos',
    ])

    Rs_TE = []
    Rs_TM = []
    Ts_TE = []
    Ts_TM = []
    times_consumed = []

    for method in methods:
        R_TE, R_TM, T_TE, T_TM, time_consumed = run_simulation(method, frequencies, thicknesses, n_k, angle)

        Rs_TE.append(R_TE)
        Rs_TM.append(R_TM)
        Ts_TE.append(T_TE)
        Ts_TM.append(T_TM)
        times_consumed.append(time_consumed)

    Rs_TE = onp.array(Rs_TE)
    Rs_TM = onp.array(Rs_TM)
    Ts_TE = onp.array(Ts_TE)
    Ts_TM = onp.array(Ts_TM)
    times_consumed = onp.array(times_consumed)

    print(Rs_TE.shape, Rs_TM.shape, Ts_TE.shape, Ts_TM.shape, times_consumed.shape)


def compare_simulations(num_tests, use_zero_angle, use_thick_layers):
    wavelengths = onp.linspace(300e-9, 900e-9, 101)
    frequencies = utils_units.get_light_speed() / wavelengths

    all_materials = utils_materials.get_all_materials()

#    num_layerss = onp.array([2, 4, 6, 8, 10, 12, 14, 16])
    num_layerss = onp.array([2, 4])

    random_state = onp.random.RandomState(42)

    for num_layers in num_layerss:
        for _ in range(0, num_tests):
            materials = random_state.choice(all_materials, num_layers)

            angle = 0.0 if use_zero_angle else random_state.uniform(0.0, 89.9)

            n_k_air = onp.ones_like(frequencies, dtype=onp.complex128)
            thickness_air = 0.0

            n_k = [n_k_air]
            thicknesses = [thickness_air]

            for material in materials:
                n_material, k_material = utils_materials.interpolate_material_n_k(
                    material, jnp.array(frequencies)
                )
                n_k_material = n_material + 1j * k_material

                thickness_material = (
                    random_state.uniform(5.0, 100.0)
                    if use_thick_layers
                    else random_state.uniform(0.01, 10.0)
                )
                thickness_material *= utils_units.get_nano()

                n_k.append(n_k_material)
                thicknesses.append(thickness_material)

            thicknesses = onp.array(thicknesses)
            n_k = onp.array(n_k)

            thicknesses[-1] = 0.0

            compare_simulation(frequencies, thicknesses, n_k, angle)

            '''
            # Compare old and new
            is_close_R_TE = onp.allclose(R_TE_old, R_TE_new, rtol=1e-5)
            is_close_T_TE = onp.allclose(T_TE_old, T_TE_new, rtol=1e-5)
            is_close_R_TM = onp.allclose(R_TM_old, R_TM_new, rtol=1e-5)
            is_close_T_TM = onp.allclose(T_TM_old, T_TM_new, rtol=1e-5)

            if not (
                is_close_R_TE and is_close_T_TE and is_close_R_TM and is_close_T_TM
            ):
                print(f"Mismatch detected")
                print(f"materials {materials}")
                print(f"thicknesses {thicknesses}")
                print(f"angle {angle}")

                # Calculate Lumerical results
                Rs, Rp, Ts, Tp = compute_properties_via_stackrt(
                    onp.array(thicknesses),
                    onp.array(n_k).T,
                    onp.array(frequencies),
                    angle_of_incidence=onp.array([angle]),
                )

                R_TE_lum = onp.squeeze(onp.array(Rs), axis=1)
                R_TM_lum = onp.squeeze(onp.array(Rp), axis=1)
                T_TE_lum = onp.squeeze(onp.array(Ts), axis=1)
                T_TM_lum = onp.squeeze(onp.array(Tp), axis=1)

                if not is_close_R_TE:
                    print("R_TE NOT MATHCED")
                    print(f"Old: {R_TE_old}")
                    print(f"New: {R_TE_new}")
                    print(f"Lumerical: {R_TE_lum}\n")

                if not is_close_T_TE:
                    print("T_TE NOT MATHCED")
                    print(f"Old: {T_TE_old}")
                    print(f"New: {T_TE_new}")
                    print(f"Lumerical: {T_TE_lum}\n")

                if not is_close_R_TM:
                    print("R_TM NOT MATHCED")
                    print(f"Old: {R_TM_old}")
                    print(f"New: {R_TM_new}")
                    print(f"Lumerical: {R_TM_lum}\n")

                if not is_close_T_TM:
                    print("T_TM NOT MATHCED")
                    print(f"Old: {T_TM_old}")
                    print(f"New: {T_TM_new}")
                    print(f"Lumerical: {T_TM_lum}\n")
            '''


if __name__ == "__main__":
    num_tests = 1

    compare_simulations(num_tests=num_tests, use_zero_angle=True, use_thick_layers=False)
    compare_simulations(num_tests=num_tests, use_zero_angle=True, use_thick_layers=True)
    compare_simulations(num_tests=num_tests, use_zero_angle=False, use_thick_layers=False)
    compare_simulations(num_tests=num_tests, use_zero_angle=False, use_thick_layers=True)
