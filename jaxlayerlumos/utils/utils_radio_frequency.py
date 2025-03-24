import jax.numpy as jnp


Metals_sigma = {
    "Cu": 5.96e7,
    "Cr": 7.74e6,
    "Ag": 6.3e7,
    "Al": 3.77e7,
    "Ni": 1.43e7,
    "W": 1.79e7,
    "Ti": 2.38e6,
    "Pd": 9.52e6,
}

Metals_nk_updated_specific_sigma = {}
mu0 = 4 * jnp.pi * 1e-7  # H/m
Z_0 = 377  # Ohms, impedance of free space
# Frequency range
nu = jnp.linspace(8e9, 18e9, 11)  # From 8 GHz to 18 GHz
omega = 2 * jnp.pi * nu  # Angular frequency

for metal, sigma in Metals_sigma.items():
    Z = jnp.sqrt(
        1j * omega * mu0 / sigma
    )  # Impedance of the material using specific sigma
    n_complex = Z_0 / Z  # Complex refractive index

    # Extract real and imaginary parts of the refractive index
    n_real = jnp.real(n_complex)
    k_imag = jnp.imag(n_complex)

    # Update the dictionary with the new values
    Metals_nk_updated_specific_sigma[metal] = {
        "freq_data": nu.tolist(),  # Converting JAX arrays to lists for JSON serialization
        "n_data": n_real.tolist(),
        "k_data": k_imag.tolist(),
    }


def load_material_RF(material_name, frequencies):
    """
    Load material RF data for a given material and frequencies. Adapted for JAX.

    Parameters:
    - material_name: The name of the material to load.
    - frequencies: Array of frequencies for which data is requested.

    Returns:
    - A JAX array with columns for frequency, n, and k.
    """

    frequencies = jnp.array(frequencies)  # Ensure input frequencies are JAX arrays

    if material_name not in Metals_nk_updated_specific_sigma:
        n_default = jnp.ones_like(frequencies)
        k_default = jnp.zeros_like(frequencies)
        data = jnp.column_stack((frequencies, n_default, k_default))
    else:
        material_data = Metals_nk_updated_specific_sigma[material_name]
        freq_data = jnp.array(material_data["freq_data"])
        n_data = jnp.array(material_data["n_data"])
        k_data = jnp.array(material_data["k_data"])

        n_interpolated = jnp.interp(
            frequencies, freq_data, n_data, left="extrapolate", right="extrapolate"
        )
        k_interpolated = jnp.interp(
            frequencies, freq_data, k_data, left="extrapolate", right="extrapolate"
        )

        data = jnp.column_stack((frequencies, n_interpolated, k_interpolated))

    return data
