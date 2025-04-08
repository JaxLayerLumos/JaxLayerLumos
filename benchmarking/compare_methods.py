import time
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units

import method_ansys
import method_tmm
import method_tmm_fast
import method_jaxlayerlumos_old
import method_jaxlayerlumos

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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
    elif method == 'tmm':
        R_TE, R_TM, T_TE, T_TM = method_tmm.compute_properties_tmm(
            thicknesses, n_k, frequencies, angle)
    elif method == 'tmm_fast':
        R_TE, R_TM, T_TE, T_TM = method_tmm_fast.compute_properties_tmm_fast(
            thicknesses, n_k, frequencies, angle)
    else:
        raise ValueError

    time_end = time.monotonic()
    time_consumed = time_end - time_start

    return R_TE, R_TM, T_TE, T_TM, time_consumed


def compare_simulation(methods, frequencies, thicknesses, n_k, angle):
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

    for ind_1 in range(0, methods.shape[0]):
        for ind_2 in range(ind_1 + 1, methods.shape[0]):
            rtol = 1e-5
            atol = 0.0

            try:
                onp.testing.assert_allclose(Rs_TE[ind_1], Rs_TE[ind_2], rtol=rtol, atol=atol)
            except:
                print('R_TE not matched:', methods[ind_1], '<->', methods[ind_2], flush=True)

            try:
                onp.testing.assert_allclose(Rs_TM[ind_1], Rs_TM[ind_2], rtol=rtol, atol=atol)
            except:
                print('R_TM not matched:', methods[ind_1], '<->', methods[ind_2], flush=True)

            try:
                onp.testing.assert_allclose(Ts_TE[ind_1], Ts_TE[ind_2], rtol=rtol, atol=atol)
            except:
                print('T_TE not matched:', methods[ind_1], '<->', methods[ind_2], flush=True)

            try:
                onp.testing.assert_allclose(Ts_TM[ind_1], Ts_TM[ind_2], rtol=rtol, atol=atol)
            except:
                print('T_TM not matched:', methods[ind_1], '<->', methods[ind_2], flush=True)

    return Rs_TE, Rs_TM, Ts_TE, Ts_TM, times_consumed


def compare_simulations_layer(methods, num_layers, num_tests, use_zero_angle, use_thick_layers):
    wavelengths = onp.linspace(300e-9, 900e-9, 10001)
    frequencies = utils_units.get_light_speed() / wavelengths

    all_materials = utils_materials.get_all_materials()

    random_state = onp.random.RandomState(42)

    materials_layer = []
    thicknesses_layer = onp.zeros((0, num_layers + 1))
    angles_layer = onp.zeros((0, ))
    Rs_TE_layer = onp.zeros((methods.shape[0], 0, wavelengths.shape[0]))
    Rs_TM_layer = onp.zeros((methods.shape[0], 0, wavelengths.shape[0]))
    Ts_TE_layer = onp.zeros((methods.shape[0], 0, wavelengths.shape[0]))
    Ts_TM_layer = onp.zeros((methods.shape[0], 0, wavelengths.shape[0]))
    times_consumed_layer = onp.zeros((methods.shape[0], 0))

    for _ in range(0, num_tests):
        materials = random_state.choice(all_materials, num_layers)
        materials = onp.concatenate([['Air'], materials], axis=0)

        angle = 0.0 if use_zero_angle else random_state.uniform(0.0, 89.9)

        n_k = []
        thicknesses = []

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

        thicknesses[0] = 0.0
        thicknesses[-1] = 0.0

        Rs_TE, Rs_TM, Ts_TE, Ts_TM, times_consumed = compare_simulation(
            methods, frequencies, thicknesses, n_k, angle)

        thicknesses = onp.expand_dims(thicknesses, axis=0)
        angle = onp.array([angle])
        Rs_TE = onp.expand_dims(Rs_TE, axis=1)
        Rs_TM = onp.expand_dims(Rs_TM, axis=1)
        Ts_TE = onp.expand_dims(Ts_TE, axis=1)
        Ts_TM = onp.expand_dims(Ts_TM, axis=1)
        times_consumed = onp.expand_dims(times_consumed, axis=1)

        materials_layer.append(materials)
        thicknesses_layer = onp.concatenate([thicknesses_layer, thicknesses], axis=0)
        angles_layer = onp.concatenate([angles_layer, angle], axis=0)
        Rs_TE_layer = onp.concatenate([Rs_TE_layer, Rs_TE], axis=1)
        Rs_TM_layer = onp.concatenate([Rs_TM_layer, Rs_TM], axis=1)
        Ts_TE_layer = onp.concatenate([Ts_TE_layer, Ts_TE], axis=1)
        Ts_TM_layer = onp.concatenate([Ts_TM_layer, Ts_TM], axis=1)
        times_consumed_layer = onp.concatenate([times_consumed_layer, times_consumed], axis=1)

    materials_layer = onp.array(materials_layer)
    print('', flush=True)
    print(materials_layer.shape, thicknesses_layer.shape, angles_layer.shape, Rs_TE_layer.shape, Rs_TM_layer.shape, Ts_TE_layer.shape, Ts_TM_layer.shape, times_consumed_layer.shape, flush=True)

    mean_times_consumed_layer = onp.mean(times_consumed_layer, axis=1)
    std_times_consumed_layer = onp.std(times_consumed_layer, axis=1)

    print(f'{num_layers} layers', flush=True)
    for method, mean_time_consumed_layer, std_time_consumed_layer in zip(methods, mean_times_consumed_layer, std_times_consumed_layer):
        print(f'{method} {mean_time_consumed_layer:.4f} +- {std_time_consumed_layer:.4f} sec.', flush=True)
    print('', flush=True)

    dict_result_layer = {
        'methods': methods,
        'num_layers': num_layers,
        'num_tests': num_tests,
        'use_zero_angle': use_zero_angle,
        'use_thick_layers': use_thick_layers,
        'materials_layer': materials_layer,
        'thicknesses_layer': thicknesses_layer,
        'angles_layer': angles_layer,
        'wavelengths': wavelengths,
        'frequencies': frequencies,
        'Rs_TE_layer': Rs_TE_layer,
        'Rs_TM_layer': Rs_TM_layer,
        'Ts_TE_layer': Ts_TE_layer,
        'Ts_TM_layer': Ts_TM_layer,
        'times_consumed_layer': times_consumed_layer
    }

    onp.save(
        f'results_{methods.shape[0]}_{num_layers}_{num_tests}_{use_zero_angle}_{use_thick_layers}.npy',
        dict_result_layer
    )

def compare_simulations(num_tests, use_zero_angle, use_thick_layers):
    methods = onp.array([
        'ansys',
        'tmm',
        'tmm_fast',
        'jaxlayerlumos_old',
        'jaxlayerlumos',
    ])

    num_layerss = onp.array([2, 4, 6, 8, 10, 12, 14, 16])

    for num_layers in num_layerss:
        compare_simulations_layer(methods, num_layers, num_tests, use_zero_angle, use_thick_layers)


if __name__ == "__main__":
    num_tests = 100

    compare_simulations(num_tests=num_tests, use_zero_angle=True, use_thick_layers=False)
    compare_simulations(num_tests=num_tests, use_zero_angle=True, use_thick_layers=True)
    compare_simulations(num_tests=num_tests, use_zero_angle=False, use_thick_layers=False)
    compare_simulations(num_tests=num_tests, use_zero_angle=False, use_thick_layers=True)
