import jax.numpy as jnp
import csv
import json
from pathlib import Path

from jaxlayerlumos.utils_spectra import convert_wavelengths_to_frequencies


def load_json():
    current_dir = Path(__file__).parent
    materials_file = current_dir / "materials.json"

    with open(materials_file, "r") as file_json:
        material_indices = json.load(file_json)

    return material_indices, current_dir


def get_all_materials():
    material_indices, _ = load_json()
    return list(material_indices.keys())


def load_material_f_ghz(material):
    material_indices, str_directory = load_json()
    str_file = material_indices.get(material)

    if not str_file:
        raise ValueError(f"Material {material} not found in JaxLayerLumos.")

    str_csv = str_directory / str_file
    data_eps = []
    data_mu = []

    with open(str_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        start_eps = False
        start_mu = False

        for row in csvreader:
            if len(row) == 2:
                if row[0] == "f" and row[1] == "eps_r":
                    start_eps = True
                    start_mu = False
                elif row[0] == "f" and row[1] == "eps_mu":
                    start_eps = False
                    start_mu = True
                else:
                    f_GHz, value = map(float, row)

                    if start_eps and not start_mu:
                        data_mu.append([f_Ghz, value])
                    elif not start_eps and start_mu:
                        data_eps.append([f_Ghz, value])
                    else:
                        raise ValueError
            elif len(row) == 0:
                pass
            else:
                raise ValueError

    data_eps = jnp.array(data_eps)
    data_mu = jnp.array(data_mu)
    assert data_eps.shape[0] > 0 or data_mu.shape[0] > 0

    if data_eps.shape[0] == 0:
        data_eps = jnp.concatenate(
            [data_eps[:, 0][..., jnp.newaxis], jnp.zeros((data_eps.shape[0], 1))],
            axis=1,
        )
    if data_mu.shape[0] == 0:
        data_mu = jnp.concatenate(
            [data_mu[:, 0][..., jnp.newaxis], jnp.zeros((data_mu.shape[0], 1))], axis=1
        )

    return data_eps, data_mu


def load_material_wavelength_um(material):
    material_indices, str_directory = load_json()
    str_file = material_indices.get(material)

    if not str_file:
        raise ValueError(f"Material {material} not found in JaxLayerLumos.")

    str_csv = str_directory / str_file
    data_n = []
    data_k = []

    with open(str_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)

        start_n = False
        start_k = False

        for row in csvreader:
            if len(row) == 2:
                if row[0] == "wl" and row[1] == "n":
                    start_n = True
                    start_k = False
                elif row[0] == "wl" and row[1] == "k":
                    start_n = False
                    start_k = True
                else:
                    wavelength_um, value = map(float, row)

                    if start_n and not start_k:
                        data_n.append([wavelength_um, value])
                    elif not start_n and start_k:
                        data_k.append([wavelength_um, value])
                    else:
                        raise ValueError
            elif len(row) == 0:
                pass
            else:
                raise ValueError

    data_n = jnp.array(data_n)
    data_k = jnp.array(data_k)
    assert data_n.shape[0] > 0 or data_k.shape[0] > 0

    if data_n.shape[0] == 0:
        data_n = jnp.concatenate(
            [data_k[:, 0][..., jnp.newaxis], jnp.zeros((data_k.shape[0], 1))], axis=1
        )
    if data_k.shape[0] == 0:
        data_k = jnp.concatenate(
            [data_n[:, 0][..., jnp.newaxis], jnp.zeros((data_n.shape[0], 1))], axis=1
        )

    return data_n, data_k


def load_material_wavelength(material):
    data_n, data_k = load_material_wavelength_um(material)

    data_n = data_n.at[:, 0].set(data_n[:, 0] * 1e-6)
    data_k = data_k.at[:, 0].set(data_k[:, 0] * 1e-6)

    return data_n, data_k


def load_material(material):
    data_n, data_k = load_material_wavelength(material)

    data_n = data_n.at[:, 0].set(convert_wavelengths_to_frequencies(data_n[:, 0]))
    data_k = data_k.at[:, 0].set(convert_wavelengths_to_frequencies(data_k[:, 0]))

    return data_n, data_k


def interpolate(freqs_values, frequencies):
    assert isinstance(freqs_values, jnp.ndarray)
    assert isinstance(frequencies, jnp.ndarray)
    assert freqs_values.ndim == 2
    assert frequencies.ndim == 1

    freqs, values = freqs_values.T

#    assert jnp.min(freqs) <= jnp.min(frequencies)
#    assert jnp.max(frequencies) <= jnp.max(freqs)

    values_interpolated = jnp.interp(
        frequencies,
        freqs,
        values,
        left="extrapolate",
        right="extrapolate",
    )

    return values_interpolated


def interpolate_material_n_k(material, frequencies):
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    data_n, data_k = load_material(material)

    n_material = interpolate(data_n, frequencies)
    k_material = interpolate(data_k, frequencies)

    return n_material, k_material


def interpolate_material_eps_mu(material, frequencies):
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    data_eps_r, data_mu_r = load_material_f_ghz(material)

    eps_r_real = interpolate(jnp.real(data_eps_r), frequencies)
    eps_r_imag = interpolate(jnp.imag(data_eps_r), frequencies)

    mu_r_real = interpolate(jnp.real(data_eps_r), frequencies)
    mu_r_imag = interpolate(jnp.imag(data_eps_r), frequencies)

    return eps_r_real, eps_r_imag, mu_r_real, mu_r_imag


def get_eps_mu_Michielssen(materialInd, f_Hz):
    # Gets parameters from Michiellsen
    f = f_Hz / 1e9  # in GHz
    M_epsr = jnp.vstack(
        [
            jnp.tile(
                jnp.array([10, 50, 15, 15, 15])[:, None], (1, len(f))
            ),  # Materials 1 to 5
            jnp.array(
                [  # Frequency-dependent permittivity for materials 6 to 8
                    5 / (f**0.861) - 1j * (8 / (f**0.569)),
                    8 / (f**0.778) - 1j * (10 / (f**0.682)),
                    10 / (f**0.778) - 1j * (6 / (f**0.861)),
                ]
            ),
            jnp.full((8, len(f)), 15, dtype=complex),  # Materials 9 to 16
        ]
    )

    # Fill constant values for permeability (mur)
    M_mur = jnp.vstack(
        [
            jnp.ones((2, len(f))),  # Materials 1 and 2
            jnp.array(
                [  # Frequency-dependent permeability for materials 3 to 5
                    5 / (f**0.974) - 1j * (10 / (f**0.961)),
                    3 / (f**1.0) - 1j * (15 / (f**0.957)),
                    7 / (f**1.0) - 1j * (12 / (f**1.0)),
                ]
            ),
            jnp.ones((3, len(f))),  # Materials 6 to 8
            jnp.array(
                [  # Frequency-dependent permeability for materials 9 to 16
                    (35 * (0.8**2)) / (f**2 + 0.8**2)
                    - 1j * (35 * 0.8 * f) / (f**2 + 0.8**2),
                    (35 * (0.5**2)) / (f**2 + 0.5**2)
                    - 1j * (35 * 0.5 * f) / (f**2 + 0.5**2),
                    (30 * (1**2)) / (f**2 + 1**2) - 1j * (30 * f) / (f**2 + 1**2),
                    (18 * (0.5**2)) / (f**2 + 0.5**2)
                    - 1j * (18 * 0.5 * f) / (f**2 + 0.5**2),
                    (20 * (1.5**2)) / (f**2 + 1.5**2)
                    - 1j * (20 * 1.5 * f) / (f**2 + 1.5**2),
                    (30 * (2.5**2)) / (f**2 + 2.5**2)
                    - 1j * (30 * 2.5 * f) / (f**2 + 2.5**2),
                    (30 * (2**2)) / (f**2 + 2**2) - 1j * (30 * 2 * f) / (f**2 + 2**2),
                    (25 * (3.5**2)) / (f**2 + 3.5**2)
                    - 1j * (25 * 3.5 * f) / (f**2 + 3.5**2),
                ]
            ),
        ]
    )

    # Initialize epsr and mur for the given materialInd
    eps_r = M_epsr[materialInd - 1, :]  # Python uses 0-based indexing
    mu_r = M_mur[materialInd - 1, :]

    return eps_r, mu_r


def get_eps_mu(materials, frequencies):
    assert isinstance(materials, list)
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    num_layers = len(materials)
    num_frequencies = frequencies.shape[0]

    eps_r = jnp.ones((num_layers, num_frequencies), dtype=jnp.complex128)
    mu_r = jnp.ones((num_layers, num_frequencies), dtype=jnp.complex128)

    for ind, material in enumerate(materials):
        eps_r_real, eps_r_imag, mu_r_real, mu_r_imag = interpolate_material_eps_mu(
            material, frequencies
        )
        eps_r = eps_r.at[ind + 1, :].set(eps_r_real + 1j * eps_r_imag)
        mu_r = mu_r.at[ind + 1, :].set(mu_r_real + 1j * mu_r_imag)

    eps_r = eps_r.T
    mu_r = mu_r.T
    return eps_r, mu_r


def get_n_k_surrounded_by_air(materials, frequencies):
    assert isinstance(materials, list)
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1

    num_layers = len(materials) + 2
    num_frequencies = frequencies.shape[0]

    n_k = jnp.ones((num_layers, num_frequencies), dtype=jnp.complex128)

    for ind, material in enumerate(materials):
        n_material, k_material = interpolate_material_n_k(material, frequencies)
        n_k = n_k.at[ind + 1, :].set(n_material + 1j * k_material)

    assert jnp.all(jnp.real(n_k[0]) == 1)
    assert jnp.all(jnp.imag(n_k[0]) == 0)
    assert jnp.all(jnp.real(n_k[-1]) == 1)
    assert jnp.all(jnp.imag(n_k[-1]) == 0)

    n_k = n_k.T
    return n_k
