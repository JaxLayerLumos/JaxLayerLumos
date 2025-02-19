import pytest
import jax.numpy as jnp
import numpy as onp
import scipy.constants as scic

from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_spectra


LIST_MATERIALS = [
    "Ag",
    "Air",
    "Al2O3",
    "Al",
    "aSi",
    "aSi-Zarei",
    "Au",
    "AZO-Zarei",
    "BK7",
    "Cr",
    "cSi",
    "Cu",
    "Fe",
    "FusedSilica",
    "GaAs",
    "GaInP",
    "GaP",
    "Ge",
    "InP",
    "ITO",
    "ITO-Zarei",
    "Mg",
    "Mn",
    "Ni",
    "Pb",
    "Pd",
    "Pt",
    "Sapphire",
    "Si3N4",
    "Si3N4-Zarei",
    "SiO2",
    "SiO2-Zarei",
    "TiN",
    "TiO2",
    "TiO2-Zarei",
    "Ti",
    "W",
    "ZnO",
    "Zn",
]


def test_load_json():
    material_indices, str_directory = utils_materials.load_json()

    assert isinstance(material_indices, dict)
    assert len(material_indices) == len(LIST_MATERIALS)


def test_get_all_materials():
    all_materials = utils_materials.get_all_materials()

    assert len(all_materials) == len(LIST_MATERIALS)

    for material in all_materials:
        assert material in LIST_MATERIALS

    for material in LIST_MATERIALS:
        assert material in all_materials


def test_load_material_success():
    for material in LIST_MATERIALS:
        data_n, data_k = utils_materials.load_material(material)

        assert isinstance(data_n, jnp.ndarray)
        assert isinstance(data_k, jnp.ndarray)
        assert data_n.ndim == 2
        assert data_k.ndim == 2
        assert data_n.shape[0] > 0
        assert data_k.shape[0] > 0
        assert data_n.shape[1] == 2
        assert data_k.shape[1] == 2


def test_load_material_failure():
    with pytest.raises(ValueError) as e:
        utils_materials.load_material("FakeMaterial")
    onp.testing.assert_string_equal(
        str(e.value), "Material FakeMaterial not found in JaxLayerLumos."
    )

    with pytest.raises(ValueError) as e:
        utils_materials.load_material("Material")
    onp.testing.assert_string_equal(
        str(e.value), "Material Material not found in JaxLayerLumos."
    )


def test_interpolate():
    freqs_values = jnp.array([[100.0, 10.0], [200.0, 20.0], [350, 35.0]])
    frequencies = jnp.array([100.0, 150.0, 175.0, 210.0, 300.0, 350.0])
    frequencies_extrapolation = jnp.array(
        [50.0, 100.0, 150.0, 175.0, 210.0, 300.0, 500.0]
    )

    with pytest.raises(AssertionError):
        utils_materials.interpolate("abc", frequencies)
    with pytest.raises(AssertionError):
        utils_materials.interpolate(freqs_values, "abc")
    with pytest.raises(AssertionError):
        utils_materials.interpolate(freqs_values, frequencies_extrapolation)

    values_interpolated = utils_materials.interpolate(freqs_values, frequencies)
    values_truth = jnp.array([10.0, 15.0, 17.5, 21.0, 30.0, 35.0])

    onp.testing.assert_allclose(values_interpolated, values_truth)


def test_material_visible_light():
    num_wavelengths = 3456
    frequencies = utils_spectra.get_frequencies_visible_light(
        num_wavelengths=num_wavelengths
    )

    for material in LIST_MATERIALS:
        n_material, k_material = utils_materials.interpolate_material_n_k(
            material, frequencies
        )


def test_material_wide_visible_light():
    num_wavelengths = 3456
    frequencies = utils_spectra.get_frequencies_wide_visible_light(
        num_wavelengths=num_wavelengths
    )

    for material in LIST_MATERIALS:
        print(material)

        n_material, k_material = utils_materials.interpolate_material_n_k(
            material, frequencies
        )


def test_material_super_wide_light():
    num_wavelengths = 3456
    wavelengths = jnp.linspace(1 * scic.nano, 100000000 * scic.nano, num_wavelengths)
    frequencies = utils_spectra.convert_wavelengths_to_frequencies(wavelengths)

    for material in LIST_MATERIALS:
        if material == "Air":
            n_material, k_material = utils_materials.interpolate_material_n_k(
                material, frequencies
            )
        else:
            with pytest.raises(AssertionError):
                n_material, k_material = utils_materials.interpolate_material_n_k(
                    material, frequencies
                )


def test_material_data_conversion_and_interpolation():
    material_name = "SiO2"
    data_n, data_k = utils_materials.load_material(material_name)
    data = jnp.concatenate([data_n, data_k[:, 1][..., jnp.newaxis]], axis=1)

    expected_wavelength_um = 1.34065
    expected_n = 1.457795
    expected_k = 0.000774

    expected_frequency_hz = scic.c / (expected_wavelength_um * 1e-6)

    row = next(
        (row for row in data if jnp.isclose(row[0], expected_frequency_hz)),
        None,
    )

    assert row is not None

    _, actual_n, actual_k = row

    assert jnp.isclose(actual_n, expected_n)
    assert jnp.isclose(actual_k, expected_k)


def test_get_n_k():
    num_wavelengths = 34
    materials = ["Sapphire", "Al", "Cu", "SiO2", "Si3N4"]
    frequencies = utils_spectra.get_frequencies_visible_light(
        num_wavelengths=num_wavelengths
    )

    with pytest.raises(AssertionError):
        utils_materials.get_n_k("abc", frequencies)
    with pytest.raises(AssertionError):
        utils_materials.get_n_k(materials, "abc")

    n_k = utils_materials.get_n_k(materials, frequencies)

    assert n_k.ndim == 2
    assert n_k.shape[0] == num_wavelengths
    assert n_k.shape[1] == len(materials)

    assert jnp.allclose(n_k[10, 0], 1.7741475745023563 + 0j)
    assert jnp.allclose(n_k[20, 1], 1.291823225506282 + 7.198159119093977j)
    assert jnp.allclose(n_k[30, 2], 0.25578634278370915 + 4.487833873569254j)
    assert jnp.allclose(n_k[20, 3], 1.4638938204021967 + 0.0016911306115694245j)
    assert jnp.allclose(n_k[10, 4], 2.0341038418364823 + 0j)


def test_get_n_k_surrounded_by_air():
    num_wavelengths = 34
    materials = onp.array(["Ag", "Au", "W", "TiO2", "Si3N4"])
    frequencies = utils_spectra.get_frequencies_visible_light(
        num_wavelengths=num_wavelengths
    )

    n_k = utils_materials.get_n_k_surrounded_by_air(materials, frequencies)

    assert n_k.ndim == 2
    assert n_k.shape[0] == num_wavelengths
    assert n_k.shape[1] == len(materials) + 2

    print(n_k[0, 0])
    assert jnp.allclose(n_k[0, 0], 1.0 + 0j)
    assert jnp.allclose(n_k[10, 1], 0.13209204 + 2.77618307j)
    assert jnp.allclose(n_k[20, 2], 0.32581145 + 3.05379202j)
    assert jnp.allclose(n_k[30, 3], 3.74494455 + 2.78771241j)
    assert jnp.allclose(n_k[20, 4], 2.39396291 + 2.95673696e-09j)
    assert jnp.allclose(n_k[10, 5], 2.03410384 + 0.0j)
    assert jnp.allclose(n_k[0, 6], 1.0 + 0j)
