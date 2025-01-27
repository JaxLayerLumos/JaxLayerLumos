import numpy as np

import plotting


if __name__ == '__main__':
    num_methods = 5
    num_layers = 8
    num_tests = 10

    use_zero_angle = True
    use_thick_layers = True

    dict_results = np.load(f'results_{num_methods}_{num_layers}_{num_tests}_{use_zero_angle}_{use_thick_layers}.npy', allow_pickle=True)
    dict_results = dict_results[()]

    print(dict_results.keys())

    assert num_layers == dict_results['num_layers']
    assert num_tests == dict_results['num_tests']
    assert use_zero_angle == dict_results['use_zero_angle']
    assert use_thick_layers == dict_results['use_thick_layers']

    methods = dict_results['methods']
    materials_layer = dict_results['materials_layer']
    thicknesses_layer = dict_results['thicknesses_layer']
    angles_layer = dict_results['angles_layer']

    wavelengths = dict_results['wavelengths']
    frequencies = dict_results['frequencies']

    Rs_TE_layer = dict_results['Rs_TE_layer']
    Rs_TM_layer = dict_results['Rs_TM_layer']
    Ts_TE_layer = dict_results['Ts_TE_layer']
    Ts_TM_layer = dict_results['Ts_TM_layer']

    times_consumed_layer = dict_results['times_consumed_layer']

    print(methods.shape, materials_layer.shape, thicknesses_layer.shape, angles_layer.shape)
    print(wavelengths.shape, frequencies.shape)
    print(Rs_TE_layer.shape, Rs_TM_layer.shape, Ts_TE_layer.shape, Ts_TM_layer.shape)
    print(times_consumed_layer.shape)

    assert methods.shape[0] == Rs_TE_layer.shape[0] == Rs_TM_layer.shape[0] == Ts_TE_layer.shape[0] == Ts_TM_layer.shape[0]
    assert methods.shape[0] == times_consumed_layer.shape[0]

    assert materials_layer.shape[0] == thicknesses_layer.shape[0] == angles_layer.shape[0]
    assert materials_layer.shape[0] == Rs_TE_layer.shape[1] == Rs_TM_layer.shape[1] == Ts_TE_layer.shape[1] == Ts_TM_layer.shape[1]
    assert materials_layer.shape[0] == times_consumed_layer.shape[1]

    assert materials_layer.shape[1] == thicknesses_layer.shape[1] == (num_layers + 1)
    assert wavelengths.shape[0] == frequencies.shape[0] == Rs_TE_layer.shape[2] == Rs_TM_layer.shape[2] == Ts_TE_layer.shape[2] == Ts_TM_layer.shape[2]

    for ind in range(0, num_tests):
        Rs_TE = Rs_TE_layer[:, ind]
        Rs_TM = Rs_TM_layer[:, ind]
        Ts_TE = Ts_TE_layer[:, ind]
        Ts_TM = Ts_TM_layer[:, ind]

        str_file = f'comparisons_{num_methods}_{num_layers}_{num_tests}_{use_zero_angle}_{use_thick_layers}_{ind}'
        linestyles = [
            'solid',
            'dotted',
            'dashed',
            'dashdot',
            (0, (3, 5, 1, 5)),
        ]

        assert methods.shape[0] == len(linestyles)

        plotting.plot_spectra(
            frequencies, Rs_TE, Rs_TM, Ts_TE, Ts_TM, methods, linestyles, str_file
        )
