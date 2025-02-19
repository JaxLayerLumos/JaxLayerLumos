import jax.numpy as jnp
import numpy as np


def calc_position_in_structure(thickness_materials, position):

    cs = np.cumsum(thickness_materials[:-1])
    layer = np.sum(position[:, None] >= cs, axis=1)
    # layer = np.sum(position >= cs, axis=0)
    distance = (position - cs[layer - 1]).T
    return [layer, distance]


def calc_position_data(layer, position_in_layer, results, update_results=True):
    coeff_TE = results["coeff_TE"][:, :, layer, :]
    coeff_TM = results["coeff_TM"][:, :, layer, :]
    n = results["n"][:, :, layer]
    kz = results["k_z"][:, :, layer]
    cos_theta = results["cos_theta"][:, :, layer]

    n0 = results["n"][:, :, 0]
    cos_theta0 = results["cos_theta"][:, :, 0]

    n0 = n0[:, :, np.newaxis]
    cos_theta0 = cos_theta0[:, :, np.newaxis]

    Ef_TE = coeff_TE[:, :, :, 0] * np.exp(1j * kz * position_in_layer)
    Eb_TE = coeff_TE[:, :, :, 1] * np.exp(-1j * kz * position_in_layer)

    poyn_TE = (
        np.real(n * cos_theta * np.conj(Ef_TE + Eb_TE) * (Ef_TE - Eb_TE))
    ) / np.real(n0 * cos_theta0)

    Ef_TM = coeff_TM[:, :, :, 0] * np.exp(1j * kz * position_in_layer)
    Eb_TM = coeff_TM[:, :, :, 1] * np.exp(-1j * kz * position_in_layer)

    poyn_TM = (
        np.real(n * np.conj(cos_theta) * (Ef_TM + Eb_TM) * np.conj(Ef_TM - Eb_TM))
    ) / (n0 * np.conj(cos_theta0))

    absorb_TE = np.imag(n * cos_theta * kz * np.abs(Ef_TE + Eb_TE) ** 2) / np.real(
        n0 * cos_theta0
    )
    absorb_TM = np.imag(
        n
        * np.conj(cos_theta)
        * (kz * np.abs(Ef_TM - Eb_TM) ** 2 - np.conj(kz) * np.abs(Ef_TM + Eb_TM) ** 2)
    ) / np.real(n0 * np.conj(cos_theta0))

    E_TE = [0, Ef_TE + Eb_TE, 0]
    E_TM = [
        (Ef_TM - Eb_TM) * cos_theta,
        0,
        (-Ef_TM - Eb_TM) * np.sqrt(1 - cos_theta**2),
    ]

    results_position = {
        "poyn_TE": poyn_TE,
        "poyn_TM": poyn_TM,
        "absorb_TE": absorb_TE,  # units of absorption/meter
        "absorb_TM": absorb_TM,  # units of absorption/meter
        "E_TE": E_TE,
        "E_TM": E_TM,
    }
    if update_results:
        results.update(results_position)
        return results
    else:
        return results_position


def calc_absorption_in_each_layer(thicknesses, results):
    # Assuming d, power_entering_TE, power_entering_TM, t_TE_i, t_TM_i, kz_layers,
    # cos_theta, n_layers, coeff_TE, coeff_TM, and calc_position_data are already defined.

    power_entering_TE = results["power_entering_TE"]
    power_entering_TM = results["power_entering_TM"]
    num_layers = len(thicknesses)

    power_entering_each_layer_TE = np.zeros((num_layers, power_entering_TE.size))
    power_entering_each_layer_TM = np.zeros((num_layers, power_entering_TM.size))

    # Set initial conditions
    power_entering_each_layer_TE[0, :] = 1
    power_entering_each_layer_TM[0, :] = 1

    power_entering_each_layer_TE[1, :] = power_entering_TE
    power_entering_each_layer_TM[1, :] = power_entering_TM

    power_entering_each_layer_TE[-1, :] = results["T_TE"]
    power_entering_each_layer_TM[-1, :] = results["T_TM"]

    # Loop for intermediate layers
    for i in range(2, num_layers - 1):
        results_out = calc_position_data([i], 0, results, update_results=False)

        power_entering_each_layer_TE[i, :] = np.real(results_out["poyn_TE"][0, :, 0])
        power_entering_each_layer_TM[i, :] = np.real(results_out["poyn_TM"][0, :, 0])

    # Calculate absorption
    absorption_TE = -np.diff(power_entering_each_layer_TE, axis=0)
    absorption_TM = -np.diff(power_entering_each_layer_TM, axis=0)

    # Add the final row to absorption (equivalent to MATLAB's behavior)
    absorption_TE = np.vstack((absorption_TE, power_entering_each_layer_TE[-1, :]))
    absorption_TM = np.vstack((absorption_TM, power_entering_each_layer_TM[-1, :]))

    results.update(
        {"absorption_layer_TE": absorption_TE, "absorption_layer_TM": absorption_TM}
    )
    return results
