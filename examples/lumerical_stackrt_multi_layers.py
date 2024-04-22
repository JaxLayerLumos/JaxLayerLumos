import numpy as np
import scipy.constants as scic
import lumapi


def compute_properties_via_stackrt(thicknesses, n_k, frequencies, angle_of_incidence=0):
    assert isinstance(thicknesses, np.ndarray)

    layers = [0]
    for thickness in thicknesses:
        layers.append(thickness * scic.nano)
    layers.append(0)

    layers = np.array(layers)
    num_layers = layers.shape[0]
    assert num_layers == (thicknesses.shape[0] + 2)

    fdtd = lumapi.FDTD(hide=True)

    assert np.all(np.real(n_k[0]) == 1)
    assert np.all(np.imag(n_k[0]) == 0)
    assert np.all(np.real(n_k[-1]) == 1)
    assert np.all(np.imag(n_k[-1]) == 0)

    RT = fdtd.stackrt(n_k, layers, frequencies, angle_of_incidence)

    Ts = RT['Ts']
    Rs = RT['Rs']
    Tp = RT['Tp']
    Rp = RT['Rp']

    return Rs, Rp, Ts, Tp
