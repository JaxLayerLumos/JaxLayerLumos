import numpy as np
import torch

from jaxlayerlumos import utils_units

from tmm_fast import coh_tmm


def compute_properties_tmm_fast(thicknesses, n_k, frequencies, angle_of_incidence, device='cpu'):
    assert isinstance(thicknesses, np.ndarray)
    assert isinstance(n_k, np.ndarray)
    assert isinstance(frequencies, np.ndarray)
    assert isinstance(angle_of_incidence, float)

    assert thicknesses.ndim == 1
    assert n_k.ndim == 2
    assert frequencies.ndim == 1

    assert thicknesses.shape[0] == n_k.shape[0]
    assert frequencies.shape[0] == n_k.shape[1]

    assert device in ['cpu', 'cuda']

    wavelengths = utils_units.get_light_speed() / frequencies
    wavelengths = torch.tensor(wavelengths)

    angle = torch.tensor([angle_of_incidence * (np.pi / 180)])

    n_k = n_k.copy()
    n_k = n_k[np.newaxis, ...]
    n_k = torch.tensor(n_k)

    thicknesses = thicknesses.copy()
    thicknesses = thicknesses[np.newaxis, ...]
    thicknesses[:, 0] = np.inf
    thicknesses[:, -1] = np.inf

    thicknesses = torch.tensor(thicknesses)

    result_s = coh_tmm('s', n_k, thicknesses, angle, wavelengths, device=device)
    result_p = coh_tmm('p', n_k, thicknesses, angle, wavelengths, device=device)

    Rs = result_s['R']
    Ts = result_s['T']
    Rp = result_p['R']
    Tp = result_p['T']

    if device == 'cuda':
        Rs = Rs.cpu()
        Ts = Ts.cpu()
        Rp = Rp.cpu()
        Tp = Tp.cpu()

    Rs = np.array(Rs)
    Ts = np.array(Ts)
    Rp = np.array(Rp)
    Tp = np.array(Tp)

    Rs = np.squeeze(Rs, axis=0)
    Ts = np.squeeze(Ts, axis=0)
    Rp = np.squeeze(Rp, axis=0)
    Tp = np.squeeze(Tp, axis=0)

    Rs = np.squeeze(Rs, axis=0)
    Ts = np.squeeze(Ts, axis=0)
    Rp = np.squeeze(Rp, axis=0)
    Tp = np.squeeze(Tp, axis=0)

    return Rs, Rp, Ts, Tp
