"""
A module for fitting the complex refractive index profile over a broad
bandwidth to a sum of Lorentzian polarizability terms using gradient-based
optimization via NLopt (nlopt.readthedocs.io). The fitting parameters are
then used to define a `Medium` object.
"""
from typing import Tuple
import meep as mp
import nlopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_materials
import os

materials_dir = 'materials'            # Directory containing material CSV files
processed_data_dir = 'processed_data'  # Directory to save processed CSV files
plots_dir = 'plots'                    # Directory to save plots
# Create output directories if they don't exist
os.makedirs(processed_data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def lorentzfunc(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Returns the complex ε profile given a set of Lorentzian parameters p
    (σ_0, ω_0, γ_0, σ_1, ω_1, γ_1, ...) for a set of frequencies x.
    """
    N = len(p) // 3
    y = np.zeros(len(x))
    for n in range(N):
        A_n = p[3 * n + 0]
        x_n = p[3 * n + 1]
        g_n = p[3 * n + 2]
        y = y + A_n / (np.square(x_n) - np.square(x) - 1j * x * g_n)
    return y


def lorentzerr(p: np.ndarray, x: np.ndarray, y: np.ndarray, grad: np.ndarray) -> float:
    """
    Returns the error (or residual or loss) as the L2 norm
    of the difference of ε(p,x) and y over a set of frequencies x as
    well as the gradient of this error with respect to each Lorentzian
    polarizability parameter in p and saving the result in grad.
    """
    N = len(p) // 3
    yp = lorentzfunc(p, x)
    val = np.sum(np.square(abs(y - yp)))
    for n in range(N):
        A_n = p[3 * n + 0]
        x_n = p[3 * n + 1]
        g_n = p[3 * n + 2]
        d = 1 / (np.square(x_n) - np.square(x) - 1j * x * g_n)
        if grad.size > 0:
            grad[3 * n + 0] = 2 * np.real(np.dot(np.conj(yp - y), d))
            grad[3 * n + 1] = (
                -4 * x_n * A_n * np.real(np.dot(np.conj(yp - y), np.square(d)))
            )
            grad[3 * n + 2] = (
                -2 * A_n * np.imag(np.dot(np.conj(yp - y), x * np.square(d)))
            )
    return val


def lorentzfit(
    p0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    alg=nlopt.LD_LBFGS,
    tol: float = 1e-25,
    maxeval: float = 10000,
) -> Tuple[np.ndarray, float]:
    """
    Returns the optimal Lorentzian polarizability parameters and error
    which minimize the error in ε(p0,x) relative to y for an initial
    set of Lorentzian polarizability parameters p0 over a set of
    frequencies x using the NLopt algorithm alg for a relative
    tolerance tol and a maximum number of iterations maxeval.
    """
    opt = nlopt.opt(alg, len(p0))
    opt.set_ftol_rel(tol)
    opt.set_maxeval(maxeval)
    opt.set_lower_bounds(np.zeros(len(p0)))
    opt.set_upper_bounds(float("inf") * np.ones(len(p0)))
    opt.set_min_objective(lambda p, grad: lorentzerr(p, x, y, grad))
    local_opt = nlopt.opt(nlopt.LD_LBFGS, len(p0))
    local_opt.set_ftol_rel(1e-10)
    local_opt.set_xtol_rel(1e-8)
    opt.set_local_optimizer(local_opt)
    popt = opt.optimize(p0)
    minf = opt.last_optimum_value()
    return popt, minf

def process_file(filename):
    base_name = os.path.splitext(filename)[0]
    csv_output_path = os.path.join(processed_data_dir, base_name + '_processed.csv')
    plot_output_path = os.path.join(plots_dir, base_name + '_comparison.png')

    # Load the data
    mydata = plot_materials.load_csv(filename)
    if mydata is None:
        print(f"Failed to load data from {filename}. Skipping.")
        return

    # Extract refractive index data
    n = mydata[:, 1] + 1j * mydata[:, 2]
    eps_inf = 1.1  # High-frequency dielectric constant (adjust as needed)
    eps = np.square(n) - eps_inf

    # Fit only the data in the wavelength range of [wl_min, wl_max]
    wl = mydata[:, 0]  # Wavelength in nm
    wl_min = 300       # Minimum wavelength (nm)
    wl_max = 20000     # Maximum wavelength (nm)
    idx_start = np.searchsorted(wl, wl_min, side='left')
    idx_end = np.searchsorted(wl, wl_max, side='right')

    # The fitting function is ε(f) where f is the frequency
    freqs = 1000 / wl  # Frequency units of 1/μm
    freqs_reduced = freqs[idx_start:idx_end]
    wl_reduced = wl[idx_start:idx_end]
    eps_reduced = eps[idx_start:idx_end]

    # Number of times to repeat optimization
    num_repeat = 30
    num_lorentzians = 5  # Number of Lorentzian terms (adjust as needed)

    ps = np.zeros((num_repeat, 3 * num_lorentzians))
    mins = np.zeros(num_repeat)
    for m in range(num_repeat):
        # Random initial parameters for Lorentzian terms
        p_rand = [10 ** (np.random.random()) for _ in range(3 * num_lorentzians)]
        ps[m, :], mins[m] = lorentzfit(
            p_rand, freqs_reduced, eps_reduced, nlopt.LD_MMA, 1e-25, 50000
        )
        ps_str = "( " + ", ".join(f"{prm:.4f}" for prm in ps[m, :]) + " )"
        print(f"Iteration {m+1}/{num_repeat}: Parameters: {ps_str}, Min value: {mins[m]:.6f}")

    # Find the best set of parameters
    idx_opt = np.argmin(mins)
    popt_str = "( " + ", ".join(f"{prm:.4f}" for prm in ps[idx_opt]) + " )"
    print(f"Optimal parameters: {popt_str}, Min value: {mins[idx_opt]:.6f}")

    # Define a Medium object using the optimal parameters
    E_susceptibilities = []

    for n in range(num_lorentzians):
        mymaterial_freq = ps[idx_opt][3 * n + 1]
        mymaterial_gamma = ps[idx_opt][3 * n + 2]

        if mymaterial_freq == 0:
            mymaterial_sigma = ps[idx_opt][3 * n + 0]
            print(f'DrudeSusceptibility frequency 1.0 gamma {mymaterial_gamma} sigma {mymaterial_sigma} epsilon {eps_inf}')

            E_susceptibilities.append(
                mp.DrudeSusceptibility(
                    frequency=1.0, gamma=mymaterial_gamma, sigma=mymaterial_sigma
                )
            )
        else:
            mymaterial_sigma = ps[idx_opt][3 * n + 0] / mymaterial_freq**2
            print(f'LorentzianSusceptibility frequency {mymaterial_freq} gamma {mymaterial_gamma} sigma {mymaterial_sigma} epsilon {eps_inf}')

            E_susceptibilities.append(
                mp.LorentzianSusceptibility(
                    frequency=mymaterial_freq,
                    gamma=mymaterial_gamma,
                    sigma=mymaterial_sigma,
                )
            )

    mymaterial = mp.Medium(epsilon=eps_inf, E_susceptibilities=E_susceptibilities)

    # Calculate dielectric function values
    mymaterial_eps = [mymaterial.epsilon(f)[0][0] for f in freqs_reduced]

    # Generate extrapolated data
    wl_fit = np.linspace(300, 20000, 2000)
    freq_fit = 1000 / wl_fit
    extra_material_eps = [mymaterial.epsilon(f)[0][0] for f in freq_fit]

    # Save the extrapolated data to CSV
    eps_fit = np.array(extra_material_eps)
    n_fit = np.sqrt(eps_fit)
    n_real = np.real(n_fit)
    n_imag = np.imag(n_fit)

    df_output = pd.DataFrame({
        'wl': wl_fit,
        'n': n_real,
        'k': n_imag
    })
    df_output.to_csv(csv_output_path, index=False)
    print(f"Saved extrapolated data to {csv_output_path}")

    # Create and save the comparison plot
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))

    # Plot the real part of epsilon
    ax[0].plot(wl_reduced, np.real(eps_reduced) + eps_inf, "bo-", label="Actual Data")
    ax[0].plot(wl_reduced, np.real(mymaterial_eps), "r-", label="Fitted Data")
    ax[0].plot(wl_fit, np.real(eps_fit), "g--", label="Extrapolated Fit")
    ax[0].set_xlabel("Wavelength (nm)")
    ax[0].set_ylabel(r"Re($\epsilon$)")
    ax[0].legend()
    ax[0].set_title("Real Part of Dielectric Function")

    # Plot the imaginary part of epsilon
    ax[1].plot(wl_reduced, np.imag(eps_reduced), "bo-", label="Actual Data")
    ax[1].plot(wl_reduced, np.imag(mymaterial_eps), "r-", label="Fitted Data")
    ax[1].plot(wl_fit, np.imag(eps_fit), "g--", label="Extrapolated Fit")
    ax[1].set_xlabel("Wavelength (nm)")
    ax[1].set_ylabel(r"Im($\epsilon$)")
    ax[1].legend()
    ax[1].set_title("Imaginary Part of Dielectric Function")

    fig.suptitle(
        f"Comparison of Actual Material Data and Fit\n"
        f"using Drude-Lorentzian Susceptibility"
    )

    fig.subplots_adjust(wspace=0.3)
    plt.savefig(plot_output_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison plot to {plot_output_path}")

def transform_file(filename):
    base_name = os.path.splitext(filename)[0]
    csv_output_path = os.path.join(processed_data_dir, base_name + '_processed.csv')

    # Load the data
    mydata = plot_materials.load_csv(filename)
    if mydata is None:
        print(f"Failed to load data from {filename}. Skipping.")
        return

    # Extract refractive index data
    n = mydata[:, 1] + 1j * mydata[:, 2]
    wl = mydata[:, 0]  # Wavelength in nm
    n_real = np.real(n)
    n_imag = np.imag(n)

    df_output = pd.DataFrame({
        'wl': wl,
        'n': n_real,
        'k': n_imag
    })
    df_output.to_csv(csv_output_path, index=False)
    print(f"Saved extrapolated data to {csv_output_path}")
    
# if __name__ == "__main__":
#     transform_file('SiC-Larruquert-2011.csv')
    process_file('AlN-Beliaev-2021.csv')
    # Loop over all CSV files in the materials directory
    for filename in os.listdir(materials_dir):
        if filename.endswith('.csv'):
            print(f"\nProcessing {filename}...")
            process_file(filename)
