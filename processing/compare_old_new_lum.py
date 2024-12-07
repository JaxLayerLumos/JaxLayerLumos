import time
import numpy as np
import jax.numpy as jnp

from jaxlayerlumos.jaxlayerlumos import stackrt_n_k as stackrt_new
from jaxlayerlumos.jaxlayerlumos_old import stackrt as stackrt_old
from jaxlayerlumos import utils_materials
from jaxlayerlumos import utils_units
from lumerical_stackrt_multi_layers import compute_properties_via_stackrt


def compare_simulations(use_zero_angle, use_thick_layers):
    wavelengths = jnp.linspace(300e-9, 900e-9, 10)
    frequencies = utils_units.get_light_speed() / wavelengths

    all_materials = utils_materials.get_all_materials()

    num_layerss = jnp.array([2, 4, 6, 8, 10])
    num_tests = 20

    random_state = np.random.RandomState(42)

    for num_layers in num_layerss:
        for _ in range(num_tests):
            materials = random_state.choice(all_materials, num_layers)

            angle = 0.0 if use_zero_angle else random_state.uniform(0.0, 89.9)

            n_k_air = jnp.ones_like(frequencies, dtype=jnp.complex128)
            thickness_air = 0.0

            n_k = [n_k_air]
            thicknesses = [thickness_air]

            for material in materials:
                n_material, k_material = utils_materials.interpolate_material_n_k(
                    material, frequencies
                )
                n_k_material = n_material + 1j * k_material

                thickness_material = (
                    random_state.uniform(5.0, 100.0) if use_thick_layers else random_state.uniform(0.01, 10.0)
                )
                thickness_material *= utils_units.get_nano()

                n_k.append(n_k_material)
                thicknesses.append(thickness_material)

            thicknesses[-1] = 0.0
            n_k = jnp.array(n_k).T
            thicknesses = jnp.array(thicknesses)

            # Old simulation
            R_TE_old, T_TE_old, R_TM_old, T_TM_old = stackrt_old(
                n_k, thicknesses, frequencies, angle
            )

            # New simulation
            R_TE_new, T_TE_new, R_TM_new, T_TM_new = stackrt_new(
                n_k, thicknesses, frequencies, angle
            )

            # Compare old and new
            is_close_R_TE = np.allclose(R_TE_old, R_TE_new, rtol=1e-5)
            is_close_T_TE = np.allclose(T_TE_old, T_TE_new, rtol=1e-5)
            is_close_R_TM = np.allclose(R_TM_old, R_TM_new, rtol=1e-5)
            is_close_T_TM = np.allclose(T_TM_old, T_TM_new, rtol=1e-5)

            if not (is_close_R_TE and is_close_T_TE and is_close_R_TM and is_close_T_TM):
                print(f"Mismatch detected for materials: {materials} at angle: {angle:.2f} degrees")

                # Calculate Lumerical results
                Rs, Rp, Ts, Tp = compute_properties_via_stackrt(
                    np.array(thicknesses),
                    np.array(n_k).T,
                    np.array(frequencies),
                    angle_of_incidence=np.array([angle]),
                )

                R_TE_lum = np.squeeze(np.array(Rs), axis=1)
                R_TM_lum = np.squeeze(np.array(Rp), axis=1)
                T_TE_lum = np.squeeze(np.array(Ts), axis=1)
                T_TM_lum = np.squeeze(np.array(Tp), axis=1)

                # Print results
                print("R_TE:")
                print(f"Old: {R_TE_old}")
                print(f"New: {R_TE_new}")
                print(f"Lumerical: {R_TE_lum}\n")

                print("T_TE:")
                print(f"Old: {T_TE_old}")
                print(f"New: {T_TE_new}")
                print(f"Lumerical: {T_TE_lum}\n")

                print("R_TM:")
                print(f"Old: {R_TM_old}")
                print(f"New: {R_TM_new}")
                print(f"Lumerical: {R_TM_lum}\n")

                print("T_TM:")
                print(f"Old: {T_TM_old}")
                print(f"New: {T_TM_new}")
                print(f"Lumerical: {T_TM_lum}\n")


if __name__ == "__main__":
    # Example configurations to run the comparisons
    compare_simulations(use_zero_angle=True, use_thick_layers=False)
    compare_simulations(use_zero_angle=True, use_thick_layers=True)
    compare_simulations(use_zero_angle=False, use_thick_layers=False)
    compare_simulations(use_zero_angle=False, use_thick_layers=True)
