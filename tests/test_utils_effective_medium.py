import pytest
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import utils_effective_medium
from jaxlayerlumos import utils_graded_layers
from jaxlayerlumos import utils_materials


MATERIAL1 = "SiO2"
MATERIAL2 = "TiO2"
FREQUENCIES = jnp.array([4.0e14, 5.0e14, 6.0e14])


def _get_material_eps():
    n_k = utils_materials.get_n_k([MATERIAL1, MATERIAL2], FREQUENCIES)
    eps_r, _ = utils_materials.convert_n_k_to_eps_mu_for_non_magnetic_materials(n_k)
    return eps_r[:, 0], eps_r[:, 1]


def test_get_effective_medium_linear_formula():
    f1 = 0.25
    eps1, eps2 = _get_material_eps()
    eps_eff, mu_eff = utils_effective_medium.get_effective_medium(
        MATERIAL1, MATERIAL2, f1, FREQUENCIES, method="linear"
    )

    onp.testing.assert_allclose(eps_eff, f1 * eps1 + (1 - f1) * eps2)
    onp.testing.assert_allclose(mu_eff, jnp.ones_like(eps_eff))


def test_get_effective_medium_maxwell_garnett_formula():
    f1 = 0.25
    eps1, eps2 = _get_material_eps()
    eps_expected = eps2 * (
        eps1 + 2 * eps2 + 2 * f1 * (eps1 - eps2)
    ) / (
        eps1 + 2 * eps2 - f1 * (eps1 - eps2)
    )

    eps_eff, mu_eff = utils_effective_medium.get_effective_medium(
        MATERIAL1, MATERIAL2, f1, FREQUENCIES, method="Maxwell-Garnett"
    )

    onp.testing.assert_allclose(eps_eff, eps_expected)
    onp.testing.assert_allclose(mu_eff, jnp.ones_like(eps_eff))


def test_get_effective_medium_bruggeman_satisfies_equation():
    f1 = 0.25
    f2 = 1 - f1
    eps1, eps2 = _get_material_eps()
    eps_eff, mu_eff = utils_effective_medium.get_effective_medium(
        MATERIAL1, MATERIAL2, f1, FREQUENCIES, method="Bruggeman"
    )

    residual = (
        f1 * (eps1 - eps_eff) / (eps1 + 2 * eps_eff)
        + f2 * (eps2 - eps_eff) / (eps2 + 2 * eps_eff)
    )

    onp.testing.assert_allclose(residual, jnp.zeros_like(residual), atol=1e-12)
    onp.testing.assert_allclose(mu_eff, jnp.ones_like(eps_eff))


@pytest.mark.parametrize("method", ["linear", "Maxwell-Garnett", "Bruggeman"])
def test_get_effective_medium_endpoints(method):
    eps1, eps2 = _get_material_eps()

    eps_0, _ = utils_effective_medium.get_effective_medium(
        MATERIAL1, MATERIAL2, 0.0, FREQUENCIES, method=method
    )
    eps_1, _ = utils_effective_medium.get_effective_medium(
        MATERIAL1, MATERIAL2, 1.0, FREQUENCIES, method=method
    )

    onp.testing.assert_allclose(eps_0, eps2)
    onp.testing.assert_allclose(eps_1, eps1)


@pytest.mark.parametrize("method", ["linear", "Maxwell-Garnett", "Bruggeman"])
def test_get_graded_effective_medium_uses_graded_layer_fractions(method):
    num_graded_layers = 5
    eps_graded, mu_graded = utils_effective_medium.get_graded_effective_medium(
        MATERIAL1,
        MATERIAL2,
        FREQUENCIES,
        num_graded_layers=num_graded_layers,
        method=method,
        graded_method="linear",
    )

    assert eps_graded.shape == (FREQUENCIES.shape[0], num_graded_layers)
    assert mu_graded.shape == (FREQUENCIES.shape[0], num_graded_layers)
    onp.testing.assert_allclose(mu_graded, jnp.ones_like(eps_graded))

    ratios = utils_graded_layers.get_mixing_ratios(num_graded_layers)
    for layer_index, ratio in enumerate(ratios):
        eps_direct, _ = utils_effective_medium.get_effective_medium(
            MATERIAL1,
            MATERIAL2,
            1 - ratio,
            FREQUENCIES,
            method=method,
        )
        onp.testing.assert_allclose(eps_graded[:, layer_index], eps_direct)


def test_get_graded_effective_medium_supports_graded_methods():
    num_graded_layers = 5
    beta = 3.0

    eps_sine, _ = utils_effective_medium.get_graded_effective_medium(
        MATERIAL1,
        MATERIAL2,
        FREQUENCIES,
        num_graded_layers=num_graded_layers,
        method="linear",
        graded_method="sine",
    )
    eps_exponential, _ = utils_effective_medium.get_graded_effective_medium(
        MATERIAL1,
        MATERIAL2,
        FREQUENCIES,
        num_graded_layers=num_graded_layers,
        method="linear",
        graded_method="exponential",
        beta=beta,
    )

    ratios_sine = utils_graded_layers.get_mixing_ratios(
        num_graded_layers, method="sine"
    )
    ratios_exponential = utils_graded_layers.get_mixing_ratios(
        num_graded_layers, method="exponential", beta=beta
    )

    eps_direct_sine, _ = utils_effective_medium.get_effective_medium(
        MATERIAL1, MATERIAL2, 1 - ratios_sine[0], FREQUENCIES, method="linear"
    )
    eps_direct_exponential, _ = utils_effective_medium.get_effective_medium(
        MATERIAL1,
        MATERIAL2,
        1 - ratios_exponential[0],
        FREQUENCIES,
        method="linear",
    )

    onp.testing.assert_allclose(eps_sine[:, 0], eps_direct_sine)
    onp.testing.assert_allclose(eps_exponential[:, 0], eps_direct_exponential)


def test_get_effective_medium_validates_inputs():
    with pytest.raises(ValueError):
        utils_effective_medium.get_effective_medium(
            MATERIAL1, MATERIAL2, 0.5, FREQUENCIES, method="quadratic"
        )

    with pytest.raises(ValueError):
        utils_effective_medium.get_graded_effective_medium(
            MATERIAL1, MATERIAL2, FREQUENCIES, graded_method="quadratic"
        )

    with pytest.raises(ValueError):
        utils_effective_medium.get_effective_medium(
            MATERIAL1, MATERIAL2, -0.1, FREQUENCIES
        )

    with pytest.raises(ValueError):
        utils_effective_medium.get_effective_medium(
            MATERIAL1, MATERIAL2, 1.1, FREQUENCIES
        )

    with pytest.raises(ValueError):
        utils_effective_medium.get_effective_medium(
            MATERIAL1, MATERIAL2, jnp.ones((2, 2)), FREQUENCIES
        )

    with pytest.raises(ValueError):
        utils_effective_medium.get_effective_medium(
            "FakeMaterial", MATERIAL2, 0.5, FREQUENCIES
        )
