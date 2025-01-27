import numpy as np

from jaxlayerlumos import utils_units

import tmm_core


def compute_properties_tmm(thicknesses, n_k, frequencies, angle_of_incidence):
    assert isinstance(thicknesses, np.ndarray)
    assert isinstance(n_k, np.ndarray)
    assert isinstance(frequencies, np.ndarray)
    assert isinstance(angle_of_incidence, float)

    assert thicknesses.ndim == 1
    assert n_k.ndim == 2
    assert frequencies.ndim == 1

    assert thicknesses.shape[0] == n_k.shape[0]
    assert frequencies.shape[0] == n_k.shape[1]

    wavelengths = utils_units.get_light_speed() / frequencies

    angle = angle_of_incidence * (np.pi / 180)

    thicknesses = thicknesses.copy()
    thicknesses[0] = np.inf
    thicknesses[-1] = np.inf

    Rs_all = []
    Rp_all = []
    Ts_all = []
    Tp_all = []

    for wavelength, n_k_single in zip(wavelengths, n_k.T):
        result_s = tmm_core.coh_tmm('s', n_k_single, thicknesses, angle, wavelength)
        result_p = tmm_core.coh_tmm('p', n_k_single, thicknesses, angle, wavelength)

        Rs = result_s['R']
        Ts = result_s['T']
        Rp = result_p['R']
        Tp = result_p['T']

        Rs_all.append(Rs)
        Rp_all.append(Rp)
        Ts_all.append(Ts)
        Tp_all.append(Tp)

    Rs = np.array(Rs_all)
    Rp = np.array(Rp_all)
    Ts = np.array(Ts_all)
    Tp = np.array(Tp_all)

    return Rs, Rp, Ts, Tp
