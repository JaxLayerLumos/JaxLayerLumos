import jax.numpy as jnp
import numpy as onp
import scipy.constants as scic

from jaxlayerlumos import stackrt
from jaxlayerlumos import utils_position


def test_absorption_1():
    wavelengths = jnp.array([300e-9])
    frequencies = scic.c / wavelengths

    materials = onp.array(["Air", "cSi", "Ag", "Air"])
    thickness_materials = [0, 100e-9, 300e-9, 0]

    theta_inc = 45

    thicknesses = jnp.array(thickness_materials)

    return_coeffs = True
    thicknesses = jnp.array(thickness_materials)

    n_k_in = [[1, 2.2 + 0.2j, 3.3 + 0.3j, 1]]
    n_k_in = onp.repeat(n_k_in, repeats=len(frequencies), axis=0)
    n_k = jnp.array(n_k_in)

    R_TE, T_TE, R_TM, T_TM, results_coeffs = stackrt(
        n_k, thicknesses, frequencies, theta_inc, return_coeffs
    )

    position = onp.array([200e-9])

    layer, position_in_layer = utils_position.calc_position_in_structure(
        thickness_materials, position
    )
    results_coeffs = utils_position.calc_position_data(
        layer, position_in_layer, results_coeffs
    )
    results_coeffs = utils_position.calc_absorption_in_each_layer(
        thicknesses, results_coeffs
    )

    expected_poyn_TE = [[[0.08841456721601584]]]
    expected_absorb_TE = [[[1094772.137299091]]]
    expected_E_y_TE = [[[0.00942054 - 0.13630371j]]]

    expected_poyn_TM = [[[0.10600323086728776]]]
    expected_absorb_TM = [[[1403267.825732217]]]
    expected_E_x_TM = [[[0.004531764087547904 - 0.15122023311889804j]]]
    expected_E_z_TM = [[[0.0002153722087487888 + 0.03224286759876336j]]]

    onp.testing.assert_allclose(results_coeffs["poyn_TE"], expected_poyn_TE)
    onp.testing.assert_allclose(results_coeffs["absorb_TE"], expected_absorb_TE)
    onp.testing.assert_allclose(results_coeffs["E_TE"][1], expected_E_y_TE)
    onp.testing.assert_allclose(results_coeffs["poyn_TM"], expected_poyn_TM)
    onp.testing.assert_allclose(results_coeffs["absorb_TM"], expected_absorb_TM)
    onp.testing.assert_allclose(results_coeffs["E_TM"][0], expected_E_x_TM)
    onp.testing.assert_allclose(results_coeffs["E_TM"][2], expected_E_z_TM)

    expected_absorb_TE = [
        [0.20635886753735466],
        [0.47550772909600736],
        [0.31416398175464727],
        [0.003969421611990711],
    ]

    expected_absorb_TM = [
        [0.04282982017862724],
        [0.5713572767577082],
        [0.3790125050837565],
        [0.006800397979908026],
    ]

    onp.testing.assert_allclose(
        results_coeffs["absorption_layer_TE"], expected_absorb_TE
    )
    onp.testing.assert_allclose(
        results_coeffs["absorption_layer_TM"], expected_absorb_TM
    )


def test_absorption_2():
    wavelengths = jnp.linspace(400e-9, 900e-9, 3)
    frequencies = scic.c / wavelengths

    materials = onp.array(["Air", "cSi", "Ag", "Air"])
    thickness_materials = [0, 100e-9, 300e-9, 0]

    theta_inc = 45

    thicknesses = jnp.array(thickness_materials)

    return_coeffs = True
    thicknesses = jnp.array(thickness_materials)

    n_k_in = [[1, 2.2 + 0.2j, 3.3 + 0.3j, 1]]
    n_k_in = onp.repeat(n_k_in, repeats=len(frequencies), axis=0)
    n_k = jnp.array(n_k_in)

    R_TE, T_TE, R_TM, T_TM, results_coeffs = stackrt(
        n_k, thicknesses, frequencies, theta_inc, return_coeffs
    )

    position = onp.array([250e-9])

    layer, position_in_layer = utils_position.calc_position_in_structure(
        thickness_materials, position
    )
    results_coeffs = utils_position.calc_position_data(
        layer, position_in_layer, results_coeffs
    )
    results_coeffs = utils_position.calc_absorption_in_each_layer(
        thicknesses, results_coeffs
    )

    expected_poyn_TE = [
        [
            [0.0795281215944815],
            [0.198347079419153],
            [0.2886886941649387],
        ]
    ]
    expected_absorb_TE = [
        [
            [583689.6416812242],
            [684878.0510809297],
            [2298377.052021518],
        ]
    ]
    expected_E_y_TE = [
        [
            [-0.016770398483731026 - 0.11396968805228336j],
            [0.14801402327428254 + 0.05826259855206729j],
            [0.04864237687986564 - 0.33941990436508995j],
        ]
    ]

    expected_poyn_TM = [
        [
            [0.10451568675009915 + 0j],
            [0.2508023557717255 + 0j],
            [0.35346825090169187 + 0j],
        ]
    ]
    expected_absorb_TM = [
        [
            [1199758.3071952716],
            [2049517.2231013447],
            [1156056.2976765223],
        ]
    ]
    expected_E_x_TM = [
        [
            [-0.03708390608189216 - 0.15817326497075732j],
            [0.253393608027317 + 0.09866437096711196j],
            [-0.01746833084159011 - 0.2312491440696341j],
        ]
    ]
    expected_E_z_TM = [
        [
            [0.007174054420912359 + 0.028833188573387777j],
            [-0.04058993797695523 - 0.011334845525958218j],
            [-0.004825942902922913 + 0.07302529365906378j],
        ]
    ]

    print("poyn_TE")
    for elem in results_coeffs["poyn_TE"].flatten():
        print(elem)

    print("absorb_TE")
    for elem in results_coeffs["absorb_TE"].flatten():
        print(elem)

    print("E_TE y")
    for elem in results_coeffs["E_TE"][1].flatten():
        print(elem)

    print("poyn_TM")
    for elem in results_coeffs["poyn_TM"].flatten():
        print(elem)

    print("absorb_TM")
    for elem in results_coeffs["absorb_TM"].flatten():
        print(elem)

    print("E_TM x")
    for elem in results_coeffs["E_TM"][0].flatten():
        print(elem)

    print("E_TM z")
    for elem in results_coeffs["E_TM"][2].flatten():
        print(elem)

    onp.testing.assert_allclose(results_coeffs["poyn_TE"], expected_poyn_TE)
    onp.testing.assert_allclose(results_coeffs["absorb_TE"], expected_absorb_TE)
    onp.testing.assert_allclose(results_coeffs["E_TE"][1], expected_E_y_TE)
    onp.testing.assert_allclose(results_coeffs["poyn_TM"], expected_poyn_TM)
    onp.testing.assert_allclose(results_coeffs["absorb_TM"], expected_absorb_TM)
    onp.testing.assert_allclose(results_coeffs["E_TM"][0], expected_E_x_TM)
    onp.testing.assert_allclose(results_coeffs["E_TM"][2], expected_E_z_TM)

    expected_absorb_TE = [
        [0.324162389538051, 0.2149672530033685, 0.23101480350621773],
        [0.3368245000189481, 0.2737383446556475, 0.17233346605599686],
        [0.3278304674932942, 0.4595937918602506, 0.498750594847855],
        [0.011182642949706735, 0.05170061048073337, 0.09790113558993042],
    ]

    expected_absorb_TM = [
        [0.10839060549108037, 0.04110188658649061, 0.04621602583930906],
        [0.4387196786035373, 0.332531724734938, 0.2545645047919112],
        [0.43200847030467016, 0.5378652829376861, 0.532870516961258],
        [0.020881245600712195, 0.08850110574088531, 0.1663489524075218],
    ]

    print("absorption_layer_TE")
    for elem in results_coeffs["absorption_layer_TE"].flatten():
        print(elem)

    print("absorption_layer_TM")
    for elem in results_coeffs["absorption_layer_TM"].flatten():
        print(elem)

    onp.testing.assert_allclose(
        results_coeffs["absorption_layer_TE"], expected_absorb_TE
    )
    onp.testing.assert_allclose(
        results_coeffs["absorption_layer_TM"], expected_absorb_TM
    )
