import jax.numpy as jnp
import numpy as np

# from jaxlayerlumos import stackrt
from jaxlayerlumos.jaxlayerlumos_old import stackrt
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_spectra
from jaxlayerlumos import utils_layers
from jaxlayerlumos import utils_units


def test_angles():

    #wavelengths = jnp.linspace(300e-9, 900e-9, 10)
    wavelengths = jnp.array([300e-9])
    frequencies = utils_units.get_light_speed() / wavelengths

    materials = ['FusedSilica', 'Si3N4']
    thickness_materials = [2.91937911, 6.12241042]
    theta = 47.1756

    n_k_air = jnp.ones_like(frequencies)
    thickness_air = 0.0

    n_k = [n_k_air]
    thicknesses = [thickness_air]
    thicknesses.extend(thickness_materials)

    for material in materials:
        n_material, k_material = utils_materials.interpolate_material_n_k(
            material, frequencies
        )
        n_k_material = n_material + 1j * k_material

        n_k.append(n_k_material)

    n_k = jnp.array(n_k).T
    thicknesses = jnp.array(thicknesses)
    thicknesses *= utils_units.get_nano()

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies, theta)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    print("R_avg")
    for elem_1 in R_avg:
        for elem_2 in elem_1:
            print(elem_2)

    print("T_avg")
    for elem_1 in T_avg:
        for elem_2 in elem_1:
            print(elem_2)

    # expected_R_avg = jnp.array([[0.14853669599855523]])
    # expected_T_avg = jnp.array([[0.6150967559499965]])
    #
    # np.testing.assert_allclose(R_avg, expected_R_avg)
    # np.testing.assert_allclose(T_avg, expected_T_avg)
