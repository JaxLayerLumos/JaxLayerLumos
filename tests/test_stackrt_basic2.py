import jax.numpy as jnp
import numpy as onp
import scipy.constants as scic

from jaxlayerlumos import stackrt
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def test_stackrt_1():
    wavelengths = jnp.linspace(300e-9, 900e-9, 100)
    frequencies = scic.c / wavelengths
    # frequencies = 4e9

    materials = onp.array(["Air", "Ag", "Air"])
    thickness_materials = [0, 5e-9, 0]

    thicknesses = jnp.array(thickness_materials)
    n_k = utils_materials.get_n_k(materials, frequencies)

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k, thicknesses, frequencies, 0.0)

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    R_avg_flat = R_avg.flatten()
    plt.plot(wavelengths, (1 - R_avg_flat))
    plt.show()
    print("R_avg")
    for elem in R_avg:
        for elem2 in elem:
            print(elem2)
    print("T_avg")
    for elem in T_avg:
        for elem2 in elem:
            print(elem2)

    # onp.testing.assert_allclose(R_avg, expected_R_avg)
    # onp.testing.assert_allclose(T_avg, expected_T_avg)

