import time
import jax
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units

import method_jaxlayerlumos

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def set_config(device):
    if device == 'cpu':
        return jax.default_device(jax.devices('cpu')[0])
    elif device == 'tpu':
        return jax.default_device(jax.devices('tpu')[0])
    else:
        raise ValueError


def run_simulation(frequencies, thicknesses, n_k, angle):
    time_start = time.monotonic()

    R_TE, R_TM, T_TE, T_TM = method_jaxlayerlumos.compute_properties_jaxlayerlumos(
        thicknesses, n_k, frequencies, angle)

    time_end = time.monotonic()
    time_consumed = time_end - time_start

    R_TE = onp.array(R_TE)
    R_TM = onp.array(R_TM)
    T_TE = onp.array(T_TE)
    T_TM = onp.array(T_TM)

    return R_TE, R_TM, T_TE, T_TM, time_consumed


def compare_simulations_layer(num_layers, num_tests, use_zero_angle, use_thick_layers):
    wavelengths = onp.linspace(300e-9, 900e-9, 10001)
    frequencies = utils_units.get_light_speed() / wavelengths

    all_materials = utils_materials.get_all_materials()

    random_state = onp.random.RandomState(42)

    materials_layer = []
    thicknesses_layer = onp.zeros((0, num_layers + 1))
    angles_layer = onp.zeros((0, ))
    Rs_TE_layer = onp.zeros((0, wavelengths.shape[0]))
    Rs_TM_layer = onp.zeros((0, wavelengths.shape[0]))
    Ts_TE_layer = onp.zeros((0, wavelengths.shape[0]))
    Ts_TM_layer = onp.zeros((0, wavelengths.shape[0]))
    times_consumed_layer = onp.zeros((0, ))

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

        Rs_TE, Rs_TM, Ts_TE, Ts_TM, times_consumed = run_simulation(frequencies, thicknesses, n_k, angle)

        thicknesses = onp.expand_dims(thicknesses, axis=0)
        angle = onp.array([angle])
        Rs_TE = onp.expand_dims(Rs_TE, axis=0)
        Rs_TM = onp.expand_dims(Rs_TM, axis=0)
        Ts_TE = onp.expand_dims(Ts_TE, axis=0)
        Ts_TM = onp.expand_dims(Ts_TM, axis=0)
        times_consumed = onp.array([times_consumed])

        materials_layer.append(materials)
        thicknesses_layer = onp.concatenate([thicknesses_layer, thicknesses], axis=0)
        angles_layer = onp.concatenate([angles_layer, angle], axis=0)
        Rs_TE_layer = onp.concatenate([Rs_TE_layer, Rs_TE], axis=0)
        Rs_TM_layer = onp.concatenate([Rs_TM_layer, Rs_TM], axis=0)
        Ts_TE_layer = onp.concatenate([Ts_TE_layer, Ts_TE], axis=0)
        Ts_TM_layer = onp.concatenate([Ts_TM_layer, Ts_TM], axis=0)
        times_consumed_layer = onp.concatenate([times_consumed_layer, times_consumed], axis=0)

    materials_layer = onp.array(materials_layer)
    print('', flush=True)
    print(materials_layer.shape, thicknesses_layer.shape, angles_layer.shape, Rs_TE_layer.shape, Rs_TM_layer.shape, Ts_TE_layer.shape, Ts_TM_layer.shape, times_consumed_layer.shape, flush=True)

    mean_times_consumed_layer = onp.mean(times_consumed_layer, axis=0)
    std_times_consumed_layer = onp.std(times_consumed_layer, axis=0)

    return mean_times_consumed_layer, std_times_consumed_layer


def compare_simulations(num_tests, use_zero_angle, use_thick_layers):
    devices = onp.array([
        'cpu',
        'tpu',
    ])

    num_layerss = onp.array([2, 4, 6, 8, 10, 12, 14, 16])

    for num_layers in num_layerss:
        for device in devices:
            with set_config(device):
                mean_time_consumed_layer, std_time_consumed_layer = compare_simulations_layer(num_layers, num_tests, use_zero_angle, use_thick_layers)

                print(f'{num_layers} layers {device} {mean_time_consumed_layer:.4f} +- {std_time_consumed_layer:.4f} sec.', flush=True)


if __name__ == "__main__":
    num_tests = 10

    compare_simulations(num_tests=num_tests, use_zero_angle=True, use_thick_layers=False)
    compare_simulations(num_tests=num_tests, use_zero_angle=True, use_thick_layers=True)
    compare_simulations(num_tests=num_tests, use_zero_angle=False, use_thick_layers=False)
    compare_simulations(num_tests=num_tests, use_zero_angle=False, use_thick_layers=True)
