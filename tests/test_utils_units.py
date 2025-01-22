import pytest
import jax.numpy as jnp
import numpy as onp
import scipy.constants as scic

from jaxlayerlumos import utils_units


def test_get_light_speed():
    onp.testing.assert_allclose(utils_units.get_light_speed(), scic.c)
    onp.testing.assert_allclose(utils_units.get_light_speed(), 299792458)


def test_get_nano():
    onp.testing.assert_allclose(utils_units.get_nano(), scic.nano)
    onp.testing.assert_allclose(utils_units.get_nano(), 1e-9)


def test_get_milli():
    onp.testing.assert_allclose(utils_units.get_milli(), scic.milli)
    onp.testing.assert_allclose(utils_units.get_milli(), 1e-3)


def test_get_centi():
    onp.testing.assert_allclose(utils_units.get_centi(), scic.centi)
    onp.testing.assert_allclose(utils_units.get_centi(), 1e-2)


def test_get_giga():
    onp.testing.assert_allclose(utils_units.get_giga(), scic.giga)
    onp.testing.assert_allclose(utils_units.get_giga(), 1e9)


def test_convert_nm_to_m():
    thicknesses = jnp.array([10.0, 20.0, 4.0, 1.0, 2.0])
    thicknesses_in_m = utils_units.convert_nm_to_m(thicknesses)

    assert thicknesses_in_m.ndim == 1
    assert thicknesses_in_m.shape[0] == thicknesses.shape[0]

    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * scic.nano)
    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * 1e-9)

    thicknesses = jnp.array(1000.0)
    thicknesses_in_m = utils_units.convert_nm_to_m(thicknesses)

    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * scic.nano)
    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * 1e-9)

    thickness = 1000.0
    thickness_in_m = utils_units.convert_nm_to_m(thickness)

    onp.testing.assert_allclose(thickness_in_m, thickness * scic.nano)
    onp.testing.assert_allclose(thickness_in_m, thickness * 1e-9)


def test_convert_mm_to_m():
    thicknesses = jnp.array([10.0, 20.0, 4.0, 1.0, 2.0])
    thicknesses_in_m = utils_units.convert_mm_to_m(thicknesses)

    assert thicknesses_in_m.ndim == 1
    assert thicknesses_in_m.shape[0] == thicknesses.shape[0]

    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * scic.milli)
    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * 1e-3)

    thicknesses = jnp.array(1000.0)
    thicknesses_in_m = utils_units.convert_mm_to_m(thicknesses)

    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * scic.milli)
    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * 1e-3)

    thickness = 1000.0
    thickness_in_m = utils_units.convert_mm_to_m(thickness)

    onp.testing.assert_allclose(thickness_in_m, thickness * scic.milli)
    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * 1e-3)


def test_convert_cm_to_m():
    thicknesses = jnp.array([10.0, 20.0, 4.0, 1.0, 2.0])
    thicknesses_in_m = utils_units.convert_cm_to_m(thicknesses)

    assert thicknesses_in_m.ndim == 1
    assert thicknesses_in_m.shape[0] == thicknesses.shape[0]

    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * scic.centi)
    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * 1e-2)

    thicknesses = jnp.array(1000.0)
    thicknesses_in_m = utils_units.convert_cm_to_m(thicknesses)

    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * scic.centi)
    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * 1e-2)

    thickness = 1000.0
    thickness_in_m = utils_units.convert_cm_to_m(thickness)

    onp.testing.assert_allclose(thickness_in_m, thickness * scic.centi)
    onp.testing.assert_allclose(thicknesses_in_m, thicknesses * 1e-2)
