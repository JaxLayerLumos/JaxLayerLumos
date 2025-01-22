import numpy as np

try:
    import lumapi
except:
    lumapi = None


def compute_properties_ansys(thicknesses, n_k, frequencies, angle_of_incidence):
    assert isinstance(thicknesses, np.ndarray)
    assert isinstance(n_k, np.ndarray)
    assert isinstance(frequencies, np.ndarray)
    assert isinstance(angle_of_incidence, float)

    assert thicknesses.ndim == 1
    assert n_k.ndim == 2
    assert frequencies.ndim == 1

    assert thicknesses.shape[0] == n_k.shape[0]
    assert frequencies.shape[0] == n_k.shape[1]

    fdtd = lumapi.FDTD(hide=True)

    RT = fdtd.stackrt(n_k, thicknesses, frequencies, angle_of_incidence)

    Rs = RT["Rs"]
    Rp = RT["Rp"]
    Ts = RT["Ts"]
    Tp = RT["Tp"]

    Rs = np.squeeze(Rs, axis=1)
    Rp = np.squeeze(Rp, axis=1)
    Ts = np.squeeze(Ts, axis=1)
    Tp = np.squeeze(Tp, axis=1)

    return Rs, Rp, Ts, Tp
