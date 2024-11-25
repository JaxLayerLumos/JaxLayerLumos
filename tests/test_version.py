STR_VERSION = "0.3.2"


def test_version_setup():
    import importlib

    assert importlib.metadata.version("jaxlayerlumos") == STR_VERSION
