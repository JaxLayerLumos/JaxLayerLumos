import pytest
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import utils_radar_materials


def test_get_eps_mu_Michielssen():
    material_indices = onp.array([1, 5, 2, 16])
    frequencies = jnp.linspace(0.1e9, 1e9, 11)

    with pytest.raises(AssertionError):
        utils_radar_materials.get_eps_mu_Michielssen("abc", frequencies)
    with pytest.raises(AssertionError):
        utils_radar_materials.get_eps_mu_Michielssen([5, 2, 3], frequencies)
    with pytest.raises(AssertionError):
        utils_radar_materials.get_eps_mu_Michielssen(1234, frequencies)
    with pytest.raises(AssertionError):
        utils_radar_materials.get_eps_mu_Michielssen(
            onp.array([5, 2, 3, 0]), frequencies
        )
    with pytest.raises(AssertionError):
        utils_radar_materials.get_eps_mu_Michielssen(
            onp.array([18, 20, 17, 5, 2, 3, 0, -1]), frequencies
        )
    with pytest.raises(AssertionError):
        utils_radar_materials.get_eps_mu_Michielssen(
            onp.array([16, 5, 2, 3, 0, -1]), frequencies
        )
    with pytest.raises(AssertionError):
        utils_radar_materials.get_eps_mu_Michielssen(material_indices, "abc")
    with pytest.raises(AssertionError):
        utils_radar_materials.get_eps_mu_Michielssen(material_indices, 123)

    eps, mu = utils_radar_materials.get_eps_mu_Michielssen(
        material_indices, frequencies
    )

    assert eps.shape[0] == mu.shape[0] == material_indices.shape[0]
    assert eps.shape[1] == mu.shape[1] == frequencies.shape[0]

    for elem in eps[0]:
        print(elem)
    for elem in mu[0]:
        print(elem)
    for elem in eps[1]:
        print(elem)
    for elem in mu[1]:
        print(elem)
    for elem in eps[2]:
        print(elem)
    for elem in mu[2]:
        print(elem)
    for elem in eps[3]:
        print(elem)
    for elem in mu[3]:
        print(elem)

    truth_eps_0 = onp.array(
        [
            (10 + 0j),
            (10 + 0j),
            (10 + 0j),
            (10 + 0j),
            (10 + 0j),
            (10 + 0j),
            (10 + 0j),
            (10 + 0j),
            (10 + 0j),
            (10 + 0j),
            (10 + 0j),
        ]
    )
    truth_eps_1 = onp.array(
        [
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
        ]
    )
    truth_eps_2 = onp.array(
        [
            (50 + 0j),
            (50 + 0j),
            (50 + 0j),
            (50 + 0j),
            (50 + 0j),
            (50 + 0j),
            (50 + 0j),
            (50 + 0j),
            (50 + 0j),
            (50 + 0j),
            (50 + 0j),
        ]
    )
    truth_eps_3 = onp.array(
        [
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
            (15 + 0j),
        ]
    )

    truth_mu_0 = onp.array(
        [
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
        ]
    )
    truth_mu_1 = onp.array(
        [
            (70 - 120j),
            (36.8421052631579 - 63.1578947368421j),
            (24.999999999999996 - 42.857142857142854j),
            (18.91891891891892 - 32.432432432432435j),
            (15.217391304347826 - 26.08695652173913j),
            (12.727272727272727 - 21.818181818181817j),
            (10.9375 - 18.75j),
            (9.589041095890412 - 16.438356164383563j),
            (8.536585365853659 - 14.634146341463415j),
            (7.692307692307692 - 13.186813186813186j),
            (7 - 12j),
        ]
    )
    truth_mu_2 = onp.array(
        [
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
            (1 + 0j),
        ]
    )
    truth_mu_3 = onp.array(
        [
            (24.979608482871125 - 0.7137030995106036j),
            (24.926543003882436 - 1.3531551916393323j),
            (24.84101748807631 - 1.9872813990461051j),
            (24.72370003794331 - 2.6136482897254356j),
            (24.575495923476918 - 3.2299223213712525j),
            (24.39753037243577 - 3.8338976299541927j),
            (24.191127681759298 - 4.423520490378843j),
            (23.957787356546635 - 4.9969099343654415j),
            (23.699158051136013 - 5.552374171980437j),
            (23.417010116148372 - 6.088422630198576j),
            (23.11320754716981 - 6.60377358490566j),
        ]
    )

    onp.testing.assert_allclose(eps[0], truth_eps_0)
    onp.testing.assert_allclose(eps[1], truth_eps_1)
    onp.testing.assert_allclose(eps[2], truth_eps_2)
    onp.testing.assert_allclose(eps[3], truth_eps_3)

    onp.testing.assert_allclose(mu[0], truth_mu_0)
    onp.testing.assert_allclose(mu[1], truth_mu_1)
    onp.testing.assert_allclose(mu[2], truth_mu_2)
    onp.testing.assert_allclose(mu[3], truth_mu_3)
