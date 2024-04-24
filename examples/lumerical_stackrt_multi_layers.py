import numpy as np
import scipy.constants as scic
import os
import sys

if os.name == "nt":
    sys.path.append("C:\\Program Files\\Lumerical\\v241\\api\\python\\")
    sys.path.append("C:\\Program Files\\Lumerical\\v232\\api\\python\\")

import lumapi


def compute_properties_via_stackrt(layers, n_k, frequencies, angle_of_incidence=0):
    assert isinstance(layers, np.ndarray)
    assert isinstance(n_k, np.ndarray)
    assert isinstance(frequencies, np.ndarray)
    assert layers.ndim == 1
    assert n_k.ndim == 2
    assert frequencies.ndim == 1
    assert layers.shape[0] == n_k.shape[0]
    assert frequencies.shape[0] == n_k.shape[1]

    fdtd = lumapi.FDTD(hide=True)

    assert np.all(np.real(n_k[0]) == 1)
    assert np.all(np.imag(n_k[0]) == 0)
    assert np.all(np.real(n_k[-1]) == 1)
    assert np.all(np.imag(n_k[-1]) == 0)

    RT = fdtd.stackrt(n_k, layers, frequencies, angle_of_incidence)

    Ts = RT["Ts"]
    Rs = RT["Rs"]
    Tp = RT["Tp"]
    Rp = RT["Rp"]

    return Rs, Rp, Ts, Tp
