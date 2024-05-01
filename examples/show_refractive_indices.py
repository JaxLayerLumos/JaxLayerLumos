import numpy as np
import jax.numpy as jnp
import scipy.constants as scic
import matplotlib.pyplot as plt

from jaxlayerlumos.utils_materials import (
    get_all_materials,
    load_material,
    interpolate_material,
)
from jaxlayerlumos.utils_spectra import (
    get_frequencies_visible_light,
    get_frequencies_wide_visible_light,
    convert_frequencies_to_wavelengths,
)


def plot_refractive_indices(wavelengths, n_k_material):
    wavelengths = wavelengths / scic.nano

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    ax.plot(wavelengths, n_k_material[:, 0], linewidth=4, label='n')
    ax.plot(wavelengths, n_k_material[:, 1], linewidth=4, label='k')

    ax.set_xlim([np.min(wavelengths), np.max(wavelengths)])
    ax.grid()

    ax.set_xlabel('Wavelengths (nm)', fontsize=18)
    ax.set_ylabel('n, k', fontsize=18)
    ax.legend(fontsize=18)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    frequencies = get_frequencies_visible_light()
    wavelengths = convert_frequencies_to_wavelengths(frequencies)
    materials = get_all_materials()

    for material in materials:
        data_material = load_material(material)
        n_k_material = interpolate_material(data_material, frequencies)

        print(material)
        print(n_k_material)
        print(n_k_material.shape)
        print("")

#        if np.any(n_k_material < 0):
        if True:
            plot_refractive_indices(wavelengths, n_k_material)
