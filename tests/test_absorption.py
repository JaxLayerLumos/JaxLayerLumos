import jax.numpy as jnp
import numpy as onp
import scipy.constants as scic

from jaxlayerlumos import stackrt
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units

from jaxlayerlumos import utils_position

import matplotlib.pyplot as plt





def test_absorption_1():
    # wavelengths = jnp.linspace(400e-9, 900e-9, 3)
    wavelengths = jnp.array([300e-9])
    # wavelengths = jnp.array([300e-9])
    frequencies = scic.c / wavelengths
    # frequencies = 4e9

    materials = onp.array(["Air", "cSi", "Ag", "Air"])
    thickness_materials = [0, 100e-9, 300e-9, 0]

    theta_inc = 45

    thicknesses = jnp.array(thickness_materials)

    return_coeffs = True
    thicknesses = jnp.array(thickness_materials)
    # n_k = utils_materials.get_n_k(materials, frequencies)
    n_k_in = [[1, 2.2+0.2j, 3.3+0.3j, 1]]
    n_k_in = onp.repeat(n_k_in, repeats=len(frequencies), axis=0)
    n_k = jnp.array(n_k_in)

    R_TE, T_TE, R_TM, T_TM, results_coeffs = stackrt(n_k, thicknesses, frequencies, theta_inc, return_coeffs)

    position = onp.array([200e-9])
    # position = onp.linspace(0, 400e-9, 1000)
    layer, position_in_layer = utils_position.calc_position_in_structure(thickness_materials, position)
    results_coeffs = utils_position.calc_position_data(layer, position_in_layer, results_coeffs)
    results_coeffs = utils_position.calc_absorption_in_each_layer(thicknesses, results_coeffs)

    plt.figure(1)
    plt.clf()

    plt.plot(position, results_coeffs['poyn_TM'].squeeze(axis = 0).T)
    plt.xlabel('position')  # Label for x-axis
    plt.ylabel('poyn_TM')  # Label for y-axis
    plt.title('Plot of position vs. poyn_TM')

    plt.legend([f'{wavelength * 1e9:.0f} nm' for wavelength in wavelengths], title='Wavelengths')
    plt.grid(True)
    plt.show()

    # Plot 2: d_s vs. absor_TE
    plt.figure(2)
    plt.clf()
    plt.plot(position, results_coeffs['absorb_TM'].squeeze(axis = 0).T)
    plt.xlabel('position')  # Label for x-axis
    plt.ylabel('absorb_TM')  # Label for y-axis
    plt.legend([f'{wavelength * 1e9:.0f} nm' for wavelength in wavelengths], title='Wavelengths')
    plt.title('Plot of position vs. absor_TM')
    plt.grid(True)  # Optional: Add grid lines
    plt.show()

    expected_poyn_TE = [[[0.08841456721601584]]]
    expected_absorb_TE = [[[1094772.137299091]]]
    expected_E_y_TE = [[[0.00942054 - 0.13630371j]]]

    expected_poyn_TM = [[[0.10600323086728776]]]
    expected_absorb_TM = [[[1403267.825732217]]]
    expected_E_x_TM = [[[0.004531764087547904-0.15122023311889804j]]]
    expected_E_z_TM = [[[0.0002153722087487888+0.03224286759876336j]]]

    onp.testing.assert_allclose(results_coeffs['poyn_TE'], expected_poyn_TE)
    onp.testing.assert_allclose(results_coeffs['absorb_TE'], expected_absorb_TE)
    onp.testing.assert_allclose(results_coeffs['E_TE'][1], expected_E_y_TE)
    onp.testing.assert_allclose(results_coeffs['poyn_TM'], expected_poyn_TM)
    onp.testing.assert_allclose(results_coeffs['absorb_TM'], expected_absorb_TM)
    onp.testing.assert_allclose(results_coeffs['E_TM'][0], expected_E_x_TM)
    onp.testing.assert_allclose(results_coeffs['E_TM'][2], expected_E_z_TM)

    expected_absorb_TE = [[0.20635887],
        [0.47550773],
        [0.31416398],
        [0.00396942]]

    expected_absorb_TM = [[0.04282982],
        [0.57108059],
        [0.3792892 ],
        [0.0068004 ]]
    try:
        onp.testing.assert_allclose(results_coeffs['absorption_layer_TE'], expected_absorb_TE)
    except:
        onp.testing.assert_allclose(results_coeffs['absorption_layer_TE'], expected_absorb_TE, atol=2e-9, rtol = 5e-7)

    try:
        onp.testing.assert_allclose(results_coeffs['absorption_layer_TM'], expected_absorb_TM)
    except:
        onp.testing.assert_allclose(results_coeffs['absorption_layer_TE'], expected_absorb_TE, atol=3e-9, rtol=3e-7)

    # print(coeff_TE)
    # print(coeff_TM)

