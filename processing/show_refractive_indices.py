import numpy as np
import os
import scipy.constants as scic
import matplotlib.pyplot as plt

from jaxlayerlumos.utils_materials import (
    get_all_materials,
    interpolate_material_n_k,
)
from jaxlayerlumos.utils_spectra import (
    get_frequencies_wide_visible_light,
    convert_frequencies_to_wavelengths,
)


def plot_refractive_indices(material, wavelengths, n_material, k_material):
    wavelengths = wavelengths / scic.nano

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    ax.plot(wavelengths, n_material, linewidth=4, label="$n$")
    ax.plot(wavelengths, k_material, linewidth=4, label="$k$")

    ax.set_xlim([np.min(wavelengths), np.max(wavelengths)])
    ax.grid()

    ax.set_xlabel("Wavelength (nm)", fontsize=18)
    ax.set_ylabel("$n$, $k$", fontsize=18)
    ax.legend(fontsize=18)

    plt.tight_layout()

    str_directory = "../assets/refractive_indices"
    if not os.path.exists(str_directory):
        os.mkdir(str_directory)

    plt.savefig(os.path.join(str_directory, f"{material}.png"))
    plt.close("all")


if __name__ == "__main__":
    frequencies = get_frequencies_wide_visible_light(num_wavelengths=101)
    wavelengths = convert_frequencies_to_wavelengths(frequencies)
    materials = get_all_materials()

    for material in materials:
        n_material, k_material = interpolate_material_n_k(material, frequencies)

        print(material)
        print(n_material.shape)
        print(k_material.shape)
        print("")

        plot_refractive_indices(material, wavelengths, n_material, k_material)
