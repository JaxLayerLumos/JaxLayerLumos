import numpy as np
import matplotlib.pyplot as plt
import os


def plot_spectra(
    frequencies, Rs_TE, Rs_TM, Ts_TE, Ts_TM, str_labels, linestyles, str_file
):
    assert frequencies.ndim == 1
    assert Rs_TE.ndim == 2
    assert Rs_TM.ndim == 2
    assert Ts_TE.ndim == 2
    assert Ts_TM.ndim == 2
    assert Rs_TE.shape[0] == Rs_TM.shape[0] == Ts_TE.shape[0] == Ts_TM.shape[0]
    assert Rs_TE.shape[0] == len(str_labels) == len(linestyles)
    assert Rs_TE.shape[1] == Rs_TM.shape[1] == Ts_TE.shape[1] == Ts_TM.shape[1]
    assert Rs_TE.shape[1] == frequencies.shape[0]

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    for R_TE, R_TM, T_TE, T_TM, str_label, linestyle in zip(
        Rs_TE, Rs_TM, Ts_TE, Ts_TM, str_labels, linestyles
    ):
        axs[0, 0].plot(
            frequencies, R_TE, label=str_label, linestyle=linestyle, linewidth=4
        )
        axs[0, 1].plot(
            frequencies, R_TM, label=str_label, linestyle=linestyle, linewidth=4
        )
        axs[1, 0].plot(
            frequencies, T_TE, label=str_label, linestyle=linestyle, linewidth=4
        )
        axs[1, 1].plot(
            frequencies, T_TM, label=str_label, linestyle=linestyle, linewidth=4
        )

    fontsize = 18

    axs[0, 0].set_xlabel("Frequency (Hz)", fontsize=fontsize)
    axs[0, 0].set_ylabel("Reflectance_TE", fontsize=fontsize)
    axs[0, 0].legend(fontsize=fontsize)
    axs[0, 0].grid()

    axs[0, 1].set_xlabel("Frequency (Hz)", fontsize=fontsize)
    axs[0, 1].set_ylabel("Reflectance_TM", fontsize=fontsize)
    axs[0, 1].legend(fontsize=fontsize)
    axs[0, 1].grid()

    axs[1, 0].set_xlabel("Frequency (Hz)", fontsize=fontsize)
    axs[1, 0].set_ylabel("Transmittance_TE", fontsize=fontsize)
    axs[1, 0].legend(fontsize=fontsize)
    axs[1, 0].grid()

    axs[1, 1].set_xlabel("Frequency (Hz)", fontsize=fontsize)
    axs[1, 1].set_ylabel("Transmittance_TM", fontsize=fontsize)
    axs[1, 1].legend(fontsize=fontsize)
    axs[1, 1].grid()

    plt.tight_layout()
#    plt.suptitle(str_file, fontsize=fontsize)
    plt.subplots_adjust(top=0.95)

    str_directory = "../assets/comparisons"
    if not os.path.exists(str_directory):
        os.mkdir(str_directory)

    plt.savefig(os.path.join(str_directory, str_file + ".png"))
