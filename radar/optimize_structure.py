import argparse
import os
import jax.numpy as jnp
import jax
import optax
import numpy as onp

import jaxlayerlumos.utils_spectra as jll_utils_spectra
import jaxlayerlumos.utils_layers as jll_utils_layers

def objective(frequencies, materials, thicknesses, polarization, angle = 0):
    eps_stack, mu_stack = utils_materials.get_eps_mu(materials, frequencies)
    R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(
        eps_stack, mu_stack, thicknesses, frequencies, angle
    )
    if polarization.lower() == "te":
        # Reflectance for TE polarization
        R_linear = R_TE
    elif polarization.lower() == "tm":
        # Reflectance for TM polarization
        R_linear = R_TM
    elif polarization.lower() == "both":
        R_linear = (R_TM + R_TE) / 2
    else:
        raise ValueError(f"Unknown polarization: {polarization}")
    R_db = 10 * jnp.log10(R_linear).squeeze()
    return R_db

def initialize_parameters(num_layers, seed):
    params = jnp.concatenate(
        [
            jax.random.uniform(jax.random.key(seed), shape=(num_layers, ), minval=0, maxval=100),
            jnp.ones((num_layers * NUM_MATERIALS, )) * 10.0,
        ],
        axis=0
    )
    return params


def optimize_structures(num_layers, num_iter, seed):
    # assert isinstance(str_mode, str)
    assert isinstance(num_layers, int)
    # assert isinstance(target_color, str)

    frequencies = jnp.linspace(2e9, 8e9, 100)

    params = initialize_parameters(num_layers, seed)

    optimizer = optax.contrib.dog()
    optimizer_state = optimizer.init(params)

    fun_objective = lambda inputs: objective(
        frequencies, materials, thicknesses, polarization, angle)

    parameters = []
    loss_values = []

    for ind in range(0, num_iter):
        values, grads = jax.value_and_grad(fun_objective)(params)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params, value=values)
        params = optax.apply_updates(params, updates)

        thicknesses, materials_coefficients = transform(params)
        loss_value = fun_objective(params)

        parameters.append(onp.array(params))
        loss_values.append(loss_value)

        print(f"Iteration {ind + 1}: {loss_value:.8f}", flush=True)
        print(thicknesses, flush=True)
        print(jnp.argmax(materials_coefficients, axis=1), flush=True)
        print('', flush=True)

    dict_info = {
        'str_mode': str_mode,
        'num_layers': num_layers,
        'num_frequencies': num_frequencies,
        'num_iter': num_iter,
        'target_color': target_color,
        'seed': seed,
        'frequencies': onp.array(frequencies),
        'wavelengths': onp.array(wavelengths),
        'parameters': onp.array(parameters),
        'loss_values': onp.array(loss_values),
    }

    return onp.array(params), dict_info



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--layers", type=int, required=True)
    # parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    # str_mode = args.mode
    num_layers = args.layers
    num_iter = 1000
    # target_color = args.target
    seed = args.seed

    params, dict_info = optimize_structures(num_layers, num_iter, seed)
    assert onp.all(params == dict_info["parameters"][-1])
