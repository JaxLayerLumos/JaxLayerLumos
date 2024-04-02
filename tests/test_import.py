def test_import_jaxlayerlumos():
    import jaxlayerlumos

def test_import_functions():
    from jaxlayerlumos import stackrt
    # Since stackrt0 functionality is merged with stackrt, no need to test stackrt0 import

    assert callable(stackrt)

    from jaxlayerlumos.jaxlayerlumos import stackrt
    # Test importing directly from the module, assuming stackrt is within jaxlayerlumos/jaxlayerlumos.py

    assert callable(stackrt)

    import jaxlayerlumos.utils_spectra
    import jaxlayerlumos.utils_materials

    # Optionally, verify the callable status of specific functions you expect to exist
    assert callable(jaxlayerlumos.utils_spectra.convert_frequencies_to_wavelengths)
    assert callable(jaxlayerlumos.utils_materials.load_material)
