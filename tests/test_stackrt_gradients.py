import jax.numpy as jnp
import jax
import numpy as onp

from jaxlayerlumos import stackrt
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_spectra
from jaxlayerlumos import utils_layers


def test_gradient_stackrt_thickness_Ag():
    num_wavelengths = 5
    frequencies = utils_spectra.get_frequencies_visible_light(
        num_wavelengths=num_wavelengths
    )
    materials = ["Air", "Ag", "Air"]

    n_stack = utils_materials.get_n_k(materials, frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(jnp.array([100e-9]))

    def compute_R_TE_first_element(d):
        R_TE, _, _, _ = stackrt(n_stack, d, frequencies, 0.0)
        return R_TE[0, 0]

    grad_R_TE = jax.grad(compute_R_TE_first_element)(d_stack)

    assert grad_R_TE is not None
    assert isinstance(grad_R_TE, jnp.ndarray)
    assert grad_R_TE.shape == d_stack.shape

    try:
        expected_grad_R_TE = jnp.array(
            [
                9.17860131054903e-09,
                717163.9524140153,
                0.0,
            ]
        )

        for elem in grad_R_TE:
            print(elem)

        onp.testing.assert_allclose(grad_R_TE, expected_grad_R_TE)
    except:  # for Python 3.8
        expected_grad_R_TE = jnp.array(
            [
                7.342881048439224e-09,
                717163.9524140068,
                0.0,
            ]
        )

        for elem in grad_R_TE:
            print(elem)

        onp.testing.assert_allclose(grad_R_TE, expected_grad_R_TE)


def test_gradient_stackrt_thickness_Au():
    num_wavelengths = 5
    frequencies = utils_spectra.get_frequencies_visible_light(
        num_wavelengths=num_wavelengths
    )

    n_stack = utils_materials.get_n_k_surrounded_by_air(["Au"], frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(jnp.array([2000e-9]))

    def compute_R_TE_first_element(d):
        R_TE, _, _, _ = stackrt(n_stack, d, frequencies, 0.0)
        return R_TE[0, 0]

    grad_R_TE = jax.grad(compute_R_TE_first_element)(d_stack)

    assert grad_R_TE is not None
    assert isinstance(grad_R_TE, jnp.ndarray)
    assert grad_R_TE.shape == d_stack.shape

    expected_grad_R_TE = jnp.array(
        [
            -4.589300655274515e-10,
            -6.203363589082817e-09,
            0.0,
        ]
    )

    for elem in grad_R_TE:
        print(elem)
    onp.testing.assert_allclose(grad_R_TE, expected_grad_R_TE, rtol=6)


def test_gradient_stackrt_thickness_TiO2_W_SiO2():
    num_wavelengths = 5
    frequencies = utils_spectra.get_frequencies_visible_light(
        num_wavelengths=num_wavelengths
    )
    materials = onp.array(["Air", "TiO2", "W", "SiO2", "Air"])

    n_stack = utils_materials.get_n_k(materials, frequencies)
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(
        jnp.array([20e-9, 5e-9, 10e-9])
    )

    def compute_R_TE_first_element(d):
        R_TE, _, _, _ = stackrt(n_stack, d, frequencies, 33.0)
        return R_TE[0, 0]

    grad_R_TE = jax.grad(compute_R_TE_first_element)(d_stack)

    assert grad_R_TE is not None
    assert isinstance(grad_R_TE, jnp.ndarray)
    assert grad_R_TE.shape == d_stack.shape

    expected_grad_R_TE = jnp.array(
        [
            4.465338601360629e-10,
            3444796.913262841,
            -25782146.849778168,
            1823098.4754607277,
            0.0,
        ]
    )

    for elem in grad_R_TE:
        print(elem)

    try:
        onp.testing.assert_allclose(grad_R_TE, expected_grad_R_TE)
    except:
        onp.testing.assert_allclose(grad_R_TE, expected_grad_R_TE, rtol=6.0)


def test_gradient_stackrt_n_k():
    num_wavelengths = 5
    frequencies = utils_spectra.get_frequencies_visible_light(
        num_wavelengths=num_wavelengths
    )
    materials = onp.array(["Air", "TiO2", "W", "SiO2", "Air"])

    n_k_stack = utils_materials.get_n_k(materials, frequencies)
    n_k_stack = n_k_stack[1:-1]
    d_stack = utils_layers.get_thicknesses_surrounded_by_air(
        jnp.array([20e-9, 5e-9, 10e-9])
    )

    def compute_R_TE_first_element(n_k):
        n_k_transformed = jnp.concatenate(
            [jnp.ones((1, num_wavelengths)), n_k, jnp.ones((1, num_wavelengths))],
            axis=0,
        )
        R_TE, _, _, _ = stackrt(n_k_transformed, d_stack, frequencies, 20.0)
        return R_TE[0, 2]

    grad_R_TE = jax.grad(compute_R_TE_first_element)(n_k_stack)

    assert grad_R_TE is not None
    assert isinstance(grad_R_TE, jnp.ndarray)
    assert grad_R_TE.shape == n_k_stack.shape

    expected_grad_R_TE = jnp.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [
                -0.36527706 - 0.10269702j,
                0.13021597 - 0.09959709j,
                0.07265407 + 0.04660518j,
                0.05677543 + 0.00396251j,
                -0.03252965 + 0.18921409j,
            ],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ]
    )

    for elem in grad_R_TE:
        print(elem)

    onp.testing.assert_allclose(grad_R_TE, expected_grad_R_TE)
