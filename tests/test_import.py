def test_import_jaxlayerlumos():
    import jaxlayerlumos


def test_import_functions():
    from jaxlayerlumos import stackrt
    from jaxlayerlumos import stackrt0
    from jaxlayerlumos import stackrt45

    assert callable(stackrt)
    assert callable(stackrt0)
    assert callable(stackrt45)

    from jaxlayerlumos.jaxlayerlumos import stackrt
    from jaxlayerlumos.jaxlayerlumos import stackrt_theta

    assert callable(stackrt)
    assert callable(stackrt_theta)

    import jaxlayerlumos.utils_spectra
    import jaxlayerlumos.utils_materials
    import jaxlayerlumos.utils_layers
