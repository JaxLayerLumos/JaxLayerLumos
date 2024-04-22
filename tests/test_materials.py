import pytest
import jax.numpy as jnp
import scipy.constants as scic

from jaxlayerlumos.utils_materials import load_material


def test_load_material_success():
    """Test loading a material data successfully with JAX."""
    material_name = "SiO2"
    data = load_material(material_name)

    # Check that the data is not empty
    assert data.size > 0

    # Check for the correct structure: frequency, n, k
    # Assuming at least one data row exists
    assert len(data[0]) == 3


def test_load_material_failure():
    """Test loading a non-existent material data to ensure it fails gracefully."""
    with pytest.raises(ValueError) as e:
        load_material("FakeMaterial")

    assert str(e.value) == "Material FakeMaterial not found in the index."


def test_material_data_conversion_and_interpolation():
    """Test that the material data is correctly converted from wavelength to frequency
    and that n and k values are correctly retrieved with JAX."""
    material_name = "SiO2"
    data = load_material(material_name)

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
