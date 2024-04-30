import numpy as np
import jax.numpy as jnp
import scipy.constants as scic

from jaxlayerlumos.utils_materials import (
    get_all_materials,
    load_material,
    interpolate_material,
)
from jaxlayerlumos.utils_spectra import (
    get_frequencies_visible_light,
    get_frequencies_wide_visible_light,
)


if __name__ == "__main__":
    frequencies = get_frequencies_wide_visible_light()
    materials = get_all_materials()

    for material in materials:
        data_material = load_material(material)
        n_k_material = interpolate_material(data_material, frequencies)

        print(material)
        print(n_k_material)
        print(n_k_material.shape)
        print("")
