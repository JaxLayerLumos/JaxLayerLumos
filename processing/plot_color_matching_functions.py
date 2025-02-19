import numpy as np
import matplotlib.pyplot as plt
from jaxlayerlumos.colors import color_matching_functions


def plot(cmfs, str_variable):
    plt.rc("text", usetex=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    ax.plot(cmfs[:, 0], cmfs[:, 1], linewidth=4, label=r"$\bar{x}$")
    ax.plot(cmfs[:, 0], cmfs[:, 2], linewidth=4, label=r"$\bar{y}$")
    ax.plot(cmfs[:, 0], cmfs[:, 3], linewidth=4, label=r"$\bar{z}$")

    fontsize = 30
    ax.set_xlabel(r"\textrm{Wavelength (nm)}", fontsize=fontsize)

    ax.set_xlim([np.min(cmfs[:, 0]), np.max(cmfs[:, 0])])
    ax.set_ylim(bottom=0.0)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.grid()
    ax.legend(
        loc="best",
        fancybox=False,
        fontsize=fontsize,
        edgecolor="black",
        ncol=3,
        columnspacing=1.0,
        handlelength=1.2,
    )

    plt.tight_layout()
    plt.savefig(
        f"../assets/colors/color_matching_functions_{str_variable}.png",
        format="png",
        transparent=True,
        bbox_inches="tight",
    )
    plt.savefig(
        f"../assets/colors/color_matching_functions_{str_variable}.pdf",
        format="pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    cmfs_cie1931 = np.array(color_matching_functions.cmfs_cie1931)
    print(cmfs_cie1931.shape)

    plot(cmfs_cie1931, "cie1931")
