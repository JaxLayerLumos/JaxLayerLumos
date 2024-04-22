STR_VERSION = "0.1.0"


def test_version():
    import jaxlayerlumos

    assert jaxlayerlumos.__version__ == STR_VERSION


def test_version_setup():
    import importlib

    assert importlib.metadata.version("jaxlayerlumos") == STR_VERSION
