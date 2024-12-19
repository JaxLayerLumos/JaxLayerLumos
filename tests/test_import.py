def test_import_jaxlayerlumos():
    import jaxlayerlumos


def test_import_stackrt():
    from jaxlayerlumos import stackrt
    from jaxlayerlumos import stackrt0
    from jaxlayerlumos import stackrt45

    assert callable(stackrt)
    assert callable(stackrt0)
    assert callable(stackrt45)


def test_import_jaxlayerlumos_stackrt():
    from jaxlayerlumos.jaxlayerlumos import stackrt_eps_mu
    from jaxlayerlumos.jaxlayerlumos import stackrt_eps_mu_theta

    from jaxlayerlumos.jaxlayerlumos import stackrt_n_k

    assert callable(stackrt_eps_mu)
    assert callable(stackrt_eps_mu_theta)

    assert callable(stackrt_n_k)


def test_import_utils():
    import jaxlayerlumos.utils_spectra
    import jaxlayerlumos.utils_materials
    import jaxlayerlumos.utils_radar_materials
    import jaxlayerlumos.utils_layers
    import jaxlayerlumos.utils_units


def test_import_wrappers():
    import jaxlayerlumos.wrappers
