import pytest
import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos import utils_materials


LIST_MATERIALS = [
    "Ag",
    "Al2O3",
    "Al",
    "Au",
    "BK7",
    "Cr",
    "Cu",
    "FusedSilica",
    "Ge",
    "Mn",
    "Ni",
    "Pd",
    "Pt",
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
    np.testing.assert_string_equal(
        str(e.value), "Material FakeMaterial not found in JaxLayerLumos."
    )

    with pytest.raises(ValueError) as e:
        utils_materials.load_material("Material")
    np.testing.assert_string_equal(
        str(e.value), "Material Material not found in JaxLayerLumos."
    )


def test_material_data_conversion_and_interpolation():
    material_name = "SiO2"
    data_n, data_k = utils_materials.load_material(material_name)
    data = jnp.concatenate([data_n, data_k[:, 1][..., jnp.newaxis]], axis=1)

    expected_wavelength_um = 1.34065
    expected_n = 1.457795
    expected_k = 0.000774

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

    assert jnp.isclose(actual_n, expected_n)
    assert jnp.isclose(actual_k, expected_k)
