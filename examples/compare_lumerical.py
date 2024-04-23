import numpy as np
import jax.numpy as jnp
import scipy.constants as scic
import matplotlib.pyplot as plt
from jaxlayerlumos import stackrt
from tmm.tmm_core import coh_tmm
from jaxlayerlumos.utils_materials import get_n_k_surrounded_by_air
from jaxlayerlumos.utils_spectra import get_frequencies_visible_light, convert_frequencies_to_wavelengths
from jaxlayerlumos.utils_layers import (
    get_thicknesses_surrounded_by_air,
    convert_nm_to_m,
)

from lumerical_stackrt_multi_layers import compute_properties_via_stackrt


if __name__ == "__main__":
    frequencies = get_frequencies_visible_light()
    list_materials = [
        ["Ag"],
        ["Ag"],
        ["Ag", "Al", "Ag"],
        ["TiO2", "Ag", "TiO2"],
        # ['TiO2', 'SiO2','TiO2', 'Cr', 'SiO2', 'Cr']
    ]
    list_thicknesses = [
        jnp.array([100.0]),
        jnp.array([10.0]),
        jnp.array([10.0, 11.0, 12.0]),
        jnp.array([20.0, 5.0, 30.0]),
        # jnp.array([24e-9, 141e-9, 153e-9, 18e-9, 135e-9, 164e-9])
    ]
    angles = jnp.array([0.0, 20.0, 45.0, 75.0])
    # angles = jnp.array([76.0])
    for angle in angles:
        for materials, thicknesses in zip(list_materials, list_thicknesses):
            assert len(materials) == thicknesses.shape[0]

            n_k = get_n_k_surrounded_by_air(materials, frequencies)
            layers = convert_nm_to_m(get_thicknesses_surrounded_by_air(thicknesses))

            lambda_list = np.sort(convert_frequencies_to_wavelengths(frequencies)) *  1e9 # Match the number of points in `frequencies`
            d_list = [np.inf] + list(thicknesses) + [np.inf]  # Convert thickness from meters to nanometers for tmm, add infinite layers for ambient
            s_R_list = []
            s_T_list = []
            p_R_list = []
            p_T_list = []

            for i, lambda_vac in enumerate(lambda_list):
                n_list = [1] + [n_k[i, j] for j in range(1, len(materials) + 1)] + [1]  # Creating the refractive index list for each wavelength
                # print(n_list)
                s_result = coh_tmm('s', n_list, d_list, angle, lambda_vac)
                p_result = coh_tmm('p', n_list, d_list, angle, lambda_vac)
                s_R_list.append(s_result['R'])
                s_T_list.append(s_result['T'])
                p_R_list.append(p_result['R'])
                p_T_list.append(p_result['T'])

            R_TE, T_TE, R_TM, T_TM = stackrt(n_k, layers, frequencies, thetas=jnp.array([angle]))

            Rs, Rp, Ts, Tp = compute_properties_via_stackrt(
                np.array(layers),
                np.array(n_k).T,
                np.array(frequencies),
                angle_of_incidence=np.array([angle]),
            )

            # assert jnp.allclose(R_TE, Rs.T)
            # assert jnp.allclose(R_TM, Rp.T)
            # assert jnp.allclose(T_TE, Ts.T)
            # assert jnp.allclose(T_TM, Tp.T)

            Rs_squeezed = Rs.squeeze()
            Rp_squeezed = Rp.squeeze()
            Ts_squeezed = Ts.squeeze()
            Tp_squeezed = Tp.squeeze()

            R_TE_squeezed = R_TE.squeeze()
            Rs_T_squeezed = s_R_list
            R_TM_squeezed = R_TM.squeeze()
            Rp_T_squeezed = p_R_list
            T_TE_squeezed = T_TE.squeeze()
            Ts_T_squeezed = s_T_list
            T_TM_squeezed = T_TM.squeeze()
            Tp_T_squeezed = p_T_list

            # Set up the plot with 4 subplots
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # Adjust the size as needed

            # Plot R_TE and Rs.T with frequencies on the x-axis
            axs[0, 0].plot(frequencies, R_TE_squeezed, label='JaxLayerLumos', color='blue')
            axs[0, 0].plot(frequencies, Rs_T_squeezed, label='tmm', color='black', linestyle=':')
            axs[0, 0].plot(frequencies, Rs_squeezed, label='Lumerical', color='red', linestyle='--')
            axs[0, 0].set_xlabel('Frequency (Hz)')
            axs[0, 0].set_ylabel('Reflectance_TE')
            axs[0, 0].legend()

            # Plot R_TM and Rp.T with frequencies on the x-axis
            axs[0, 1].plot(frequencies, R_TM_squeezed, label='JaxLayerLumos', color='blue')
            axs[0, 1].plot(frequencies, Rp_T_squeezed, label='tmm', color='black', linestyle=':')
            axs[0, 1].plot(frequencies, Rp_squeezed, label='Lumerical', color='red', linestyle='--')
            axs[0, 1].set_xlabel('Frequency (Hz)')
            axs[0, 1].set_ylabel('Reflectance_TM')
            axs[0, 1].legend()

            # Plot T_TE and Ts.T with frequencies on the x-axis
            axs[1, 0].plot(frequencies, T_TE_squeezed, label='JaxLayerLumos', color='blue')
            axs[1, 0].plot(frequencies, Ts_T_squeezed, label='tmm', color='black', linestyle=':')
            axs[1, 0].plot(frequencies, Ts_squeezed, label='Lumerical', color='red', linestyle='--')
            axs[1, 0].set_xlabel('Frequency (Hz)')
            axs[1, 0].set_ylabel('Transmittance_TE')
            axs[1, 0].legend()

            # Plot T_TM and Tp.T with frequencies on the x-axis
            axs[1, 1].plot(frequencies, T_TM_squeezed, label='JaxLayerLumos', color='blue')
            axs[1, 1].plot(frequencies, Tp_T_squeezed, label='tmm', color='black', linestyle=':')
            axs[1, 1].plot(frequencies, Tp_squeezed, label='Lumerical', color='red', linestyle='--')
            axs[1, 1].set_xlabel('Frequency (Hz)')
            axs[1, 1].set_ylabel('Transmittance_TM')
            axs[1, 1].legend()

            filename = f"tmm_{'_'.join(f'{mat}_{thick}nm' for mat, thick in zip(materials, thicknesses))}_angle_{angle}_deg"

            # Tight layout to prevent overlap
            plt.tight_layout()
            plt.suptitle(filename)
            plt.subplots_adjust(top=0.95)  # Adjust the top margin to fit the suptitle

            # Save to an image file
            plt.savefig(filename + ".png")