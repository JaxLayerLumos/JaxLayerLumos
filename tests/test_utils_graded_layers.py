import pytest
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import utils_graded_layers


def test_get_mixing_ratios_methods():
    num_graded_layers = 10
    beta = 3.0
    x = jnp.arange(1, num_graded_layers + 1) / (num_graded_layers + 1)

    ratios_linear = utils_graded_layers.get_mixing_ratios(num_graded_layers)
    ratios_sine = utils_graded_layers.get_mixing_ratios(
        num_graded_layers, method="sine"
    )
    ratios_exponential = utils_graded_layers.get_mixing_ratios(
        num_graded_layers, method="exponential", beta=beta
    )
    ratios_exponential_linear = utils_graded_layers.get_mixing_ratios(
        num_graded_layers, method="exponential", beta=0
    )

    onp.testing.assert_allclose(ratios_linear, x)
    onp.testing.assert_allclose(ratios_sine, jnp.sin(x * jnp.pi / 2))
    onp.testing.assert_allclose(
        ratios_exponential, jnp.expm1(beta * x) / jnp.expm1(beta)
    )
    onp.testing.assert_allclose(ratios_exponential_linear, x)


def test_get_graded_layers_linear_two_materials():
    n_k = jnp.array(
        [
            [2.0 + 0.1j, 4.0 + 0.3j],
            [3.0 + 0.2j, 5.0 + 0.4j],
        ]
    )
    thicknesses = jnp.array([100.0, 200.0])

    n_k_graded, thicknesses_graded = utils_graded_layers.get_graded_layers(
        n_k,
        thicknesses,
        num_graded_layers=10,
        grade_thickness=20.0,
    )

    assert n_k_graded.shape == (2, 12)
    assert thicknesses_graded.shape == (12,)

    onp.testing.assert_allclose(n_k_graded[:, 0], n_k[:, 0])
    onp.testing.assert_allclose(n_k_graded[:, -1], n_k[:, 1])
    onp.testing.assert_allclose(
        n_k_graded[:, 1], (10 / 11) * n_k[:, 0] + (1 / 11) * n_k[:, 1]
    )
    onp.testing.assert_allclose(
        n_k_graded[:, 10], (1 / 11) * n_k[:, 0] + (10 / 11) * n_k[:, 1]
    )

    onp.testing.assert_allclose(thicknesses_graded[0], 90.0)
    onp.testing.assert_allclose(thicknesses_graded[1:-1], jnp.full((10,), 2.0))
    onp.testing.assert_allclose(thicknesses_graded[-1], 190.0)
    onp.testing.assert_allclose(jnp.sum(thicknesses_graded), jnp.sum(thicknesses))


def test_get_graded_layers_sine_two_materials():
    n_k = jnp.array(
        [
            [2.0 + 0.1j, 4.0 + 0.3j],
            [3.0 + 0.2j, 5.0 + 0.4j],
        ]
    )
    thicknesses = jnp.array([100.0, 200.0])
    ratio = jnp.sin((1 / 11) * jnp.pi / 2)

    n_k_graded, _ = utils_graded_layers.get_graded_layers(
        n_k,
        thicknesses,
        num_graded_layers=10,
        grade_thickness=20.0,
        method="sine",
    )

    onp.testing.assert_allclose(
        n_k_graded[:, 1], (1 - ratio) * n_k[:, 0] + ratio * n_k[:, 1]
    )


def test_get_graded_layers_exponential_two_materials():
    n_k = jnp.array(
        [
            [2.0 + 0.1j, 4.0 + 0.3j],
            [3.0 + 0.2j, 5.0 + 0.4j],
        ]
    )
    thicknesses = jnp.array([100.0, 200.0])
    beta = 3.0
    ratio = jnp.expm1(beta / 11) / jnp.expm1(beta)

    n_k_graded, _ = utils_graded_layers.get_graded_layers(
        n_k,
        thicknesses,
        num_graded_layers=10,
        grade_thickness=20.0,
        method="exponential",
        beta=beta,
    )

    onp.testing.assert_allclose(
        n_k_graded[:, 1], (1 - ratio) * n_k[:, 0] + ratio * n_k[:, 1]
    )


def test_get_graded_layers_skips_zero_thickness_air_boundaries():
    n_k = jnp.array(
        [
            [1.0 + 0j, 2.0 + 0.1j, 4.0 + 0.3j, 1.0 + 0j],
            [1.0 + 0j, 3.0 + 0.2j, 5.0 + 0.4j, 1.0 + 0j],
        ]
    )
    thicknesses = jnp.array([0.0, 100.0, 200.0, 0.0])

    n_k_graded, thicknesses_graded = utils_graded_layers.get_graded_layers(
        n_k,
        thicknesses,
        num_graded_layers=10,
        grade_thickness=20.0,
    )

    assert n_k_graded.shape == (2, 14)
    assert thicknesses_graded.shape == (14,)
    onp.testing.assert_allclose(n_k_graded[:, 0], n_k[:, 0])
    onp.testing.assert_allclose(n_k_graded[:, -1], n_k[:, -1])
    onp.testing.assert_allclose(thicknesses_graded[0], 0.0)
    onp.testing.assert_allclose(thicknesses_graded[-1], 0.0)
    onp.testing.assert_allclose(jnp.sum(thicknesses_graded), jnp.sum(thicknesses))


def test_get_graded_layers_validates_inputs():
    n_k = jnp.ones((2, 2), dtype=jnp.complex128)
    thicknesses = jnp.array([10.0, 10.0])

    with pytest.raises(AssertionError):
        utils_graded_layers.get_graded_layers(
            n_k,
            thicknesses,
            num_graded_layers=0,
            grade_thickness=2.0,
        )

    with pytest.raises(AssertionError):
        utils_graded_layers.get_graded_layers(
            jnp.ones((2, 3), dtype=jnp.complex128),
            thicknesses,
            grade_thickness=2.0,
        )

    with pytest.raises(ValueError):
        utils_graded_layers.get_graded_layers(
            n_k,
            thicknesses,
            grade_thickness=30.0,
        )

    with pytest.raises(ValueError):
        utils_graded_layers.get_graded_layers(
            n_k,
            thicknesses,
            grade_thickness=2.0,
            method="quadratic",
        )
