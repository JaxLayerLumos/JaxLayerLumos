import pytest
import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos import utils_materials


LIST_MATERIALS = [
    "Ag",
    "Al2O3",
    "Al",
    "Cr",
    "Ni",
    "Pd",
    "Si3N4",
    "SiO2",
    "TiN",
    "TiO2",
    "Ti",
    "W",
]


def test_get_all_materials():
    all_materials = utils_materials.get_all_materials()

    assert len(all_materials) == len(LIST_MATERIALS)

    for material in all_materials:
        assert material in LIST_MATERIALS

    for material in LIST_MATERIALS:
        assert material in all_materials


def test_load_material_success():
    for material in LIST_MATERIALS:
        data = utils_materials.load_material(material)

        assert isinstance(data, jnp.ndarray)
        assert data.ndim == 2
        assert data.shape[0] > 0
        assert data.shape[1] == 3


def test_load_material_failure():
    with pytest.raises(ValueError) as e:
        utils_materials.load_material("FakeMaterial")
    np.testing.assert_string_equal(str(e.value), "Material FakeMaterial not found in JaxLayerLumos.")

    with pytest.raises(ValueError) as e:
        utils_materials.load_material("Material")
    np.testing.assert_string_equal(str(e.value), "Material Material not found in JaxLayerLumos.")


def test_material_data_conversion_and_interpolation():
    material_name = "SiO2"
    data = utils_materials.load_material(material_name)

    # Values for a known row of SiO2 data
    expected_wavelength_um = 22.321
    expected_n = 1.804
    expected_k = 1.536

    # Convert the expected wavelength to frequency for comparison
    expected_frequency_hz = scic.c / (expected_wavelength_um * 1e-6)

    # Find the row in the data where the frequency matches the expected frequency
    # We use jnp.isclose to handle floating-point comparison
    row = next(
        (row for row in data if jnp.isclose(row[0], expected_frequency_hz, atol=1e-6)),
        None,
    )

    assert (
        row is not None
    ), f"Data row with frequency {expected_frequency_hz} not found."

    # Extract the actual n and k values from the data
    _, actual_n, actual_k = row

    # Check that the actual n and k values match the expected values within a tolerance
    assert jnp.isclose(
        actual_n, expected_n, atol=1e-6
    ), f"Expected n: {expected_n}, but got: {actual_n}"
    assert jnp.isclose(
        actual_k, expected_k, atol=1e-6
    ), f"Expected k: {expected_k}, but got: {actual_k}"
