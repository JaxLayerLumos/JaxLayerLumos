import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from jaxlayerlumos import stackrt
from jaxlayerlumos.utils_materials import (
    get_n_k_surrounded_by_air,
    get_n_k_surrounded_by_air_and_pec,
)
from jaxlayerlumos.utils_layers import (
    get_thicknesses_surrounded_by_air,
    convert_mm_to_m,
)

if __name__ == "__main__":
    # Define frequency range (0.1 GHz to 1 GHz)
    frequencies = jnp.arange(0.1e9, 1.01e9, 0.01e9)  # Frequencies from 0.1 GHz to 1 GHz
    # frequencies = jnp.array([1e9])
    # materials = ["Mat1", "Mat2", "Mat6", "Mat7", "Mat8"]  # Material list
    # materials = ["Mat16", "Mat6", "Mat5", "Mat11", "Mat2"]  # Material list
    # materials = ["Mat11", "Mat16", "Mat7", "Mat4", "Mat4"] # Material list
    materials = ["Mat6", "Mat1", "Mat6"]  # Material list
    # thicknesses = jnp.array([1.155, 0.885, 1.272, 1.446, 0.486])  # Thickness in mm
    thicknesses = jnp.array([2.0, 1.0, 1.0])  # Thickness in mm
    angles = 45.0  # Incident angles in degrees
    # angles = jnp.array([0.0, 45.0, 75.0, 89.0])  # Incident angles in degrees

    # Retrieve refractive indices for all frequencies at once
    # n_k_all = get_n_k_surrounded_by_air(materials, frequencies)
    n_k_all = get_n_k_surrounded_by_air_and_pec(materials, frequencies)

    # Inspect refractive indices
    print("Refractive indices (n_k_all):", n_k_all)

    # Convert thicknesses to meters
    layers = convert_mm_to_m(get_thicknesses_surrounded_by_air(thicknesses))

    R_TE, T_TE, R_TM, T_TM = stackrt(n_k_all, layers, frequencies, angles)
    # Inspect reflectance values before dB conversion
    print("Reflectance values (R_TE_values):", R_TE)

    # Ensure R_TE_values are valid for dB conversion
    R_TE_db = jnp.squeeze(10 * jnp.log10(R_TE))
    R_TM_db = jnp.squeeze(10 * jnp.log10(R_TM))

    # Plot reflection over frequency in dB
    plt.figure(figsize=(10, 6))
    plt.semilogx(
        frequencies * 1e-9, R_TE_db, label="Reflectance (TE) in dB"
    )  # Convert frequencies to GHz for x-axis
    plt.semilogx(
        frequencies * 1e-9, R_TM_db, label="Reflectance (TM) in dB"
    )  # Convert frequencies to GHz for x-axis
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Reflectance (dB)")
    plt.title("Reflectance vs Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()
