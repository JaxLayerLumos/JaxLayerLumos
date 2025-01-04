import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import stackrt_eps_mu
from jaxlayerlumos import utils_materials

def read_csv_file(filename):
    """
    Reads a CSV file which contains:
      - Two metadata lines starting with '#':
        #freq_range=2-8
        #num_layers=5
      - Following lines contain materials and thicknesses.
    Returns:
        all_materials : list of lists (each sub-list is the material stack, plus "Air" and "PEC")
        d_stack       : jnp.array of thicknesses, each row is one design
        freq_range    : tuple (low_freq, high_freq)
        num_layers    : integer number of layers
    """

    all_materials = []
    d_stack_list = []
    freq_range = None
    num_layers = None
    R_db_list = []
    inc_angles = 0
    polarization = "both"

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                # skip empty lines
                continue

            if line.startswith("#"):
                # Parse metadata
                if line.startswith("#freq_range="):
                    # e.g. "#freq_range=2-8" => "2-8"
                    range_str = line.split("=")[1]
                    low_str, high_str = range_str.split("-")
                    freq_range = (float(low_str), float(high_str))

                elif line.startswith("#num_layers="):
                    # e.g. "#num_layers=5" => "5"
                    num_layers_str = line.split("=")[1]
                    num_layers = int(num_layers_str)
                elif line.startswith("#inc_angle="):
                    # e.g. "#inc_angle=0,40" => "0,40"
                    angles_str = line.split("=")[1]
                    # Split on commas, convert each to float
                    inc_angles = [float(a) for a in angles_str.split(",")]
                elif line.startswith("#polarization="):
                    # e.g. "#polarization=tm" -> polarization = "tm"
                    polarization = line.split("=")[1].strip()
            else:
                # This is a data line (materials + thicknesses).
                # Split by comma
                row = line.split(",")

                # First num_layers columns = materials (strings)
                materials_part = row[:num_layers]

                # Next num_layers columns = thicknesses (numeric)
                # (Assuming the CSV *actually* has num_layers more columns for thickness)
                thickness_part = [float(x) * 1e-3 for x in row[num_layers:2*num_layers]]
                R_db_part = [float(x) for x in row[2 * num_layers:]]

                if len(R_db_part) != len(inc_angles):
                    raise ValueError(
                        f"Mismatch: {len(R_db_part)} R_db values for {len(inc_angles)} incidence angles"
                    )

                # Map each R_db value to its corresponding incidence angle
                # R_db_dict = {angle: R_db_part[idx] for idx, angle in enumerate(inc_angles)}

                # Optionally prepend "Air" and append "PEC"
                materials_part = ["Air"] + materials_part + ["PEC"]
                thickness_part = [0.0] + thickness_part + [0.0]

                all_materials.append(materials_part)
                d_stack_list.append(thickness_part)
                R_db_list.append(R_db_part)

    # Convert the list of lists into a jax.numpy array
    d_stack = jnp.array(d_stack_list)

    return all_materials, d_stack, freq_range, inc_angles, polarization, R_db_list


def get_max_reflection(freq_ghz, R_db, freq_range):
    f_min, f_max = freq_range

    # Create a boolean mask for frequencies within [f_min, f_max]
    mask = (freq_ghz >= f_min) & (freq_ghz <= f_max)

    # Slice the frequencies and reflections according to the mask
    freq_in_range = freq_ghz[mask]
    r_db_in_range = R_db[mask]

    # Find the index of the maximum reflection within the slice
    idx_max = onp.argmax(r_db_in_range)

    # Retrieve the corresponding frequency and reflection value
    f_max_val = freq_in_range[idx_max]
    r_max_val = r_db_in_range[idx_max]

    return r_max_val, f_max_val

def plot_R_frequency():
    # Table 2
    frequencies = jnp.linspace(0.1e9, 10e9, 100)
    #frequencies = jnp.array([0.1e9])
    filename = "./data/TMH.csv"
    #filename = "./data/CHF.csv"
    prefix = filename.split("/")[-1].replace(".csv", "")

    #all_materials, d_stack = read_csv_file("./data/HF.csv")
    all_materials, d_stack, freq_range, inc_angles, polarization = read_csv_file(filename)
    freq_ghz = frequencies / 1e9
    max_R_db = {}
    for i, (materials_stack, thicknesses) in enumerate(zip(all_materials, d_stack), start=1):
        # Get permittivity and permeability
        eps_stack, mu_stack = utils_materials.get_eps_mu(materials_stack, frequencies)

        # Compute reflection / transmission for TE & TM at 0° incidence
        for angle in inc_angles:
            R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(
                eps_stack, mu_stack, thicknesses, frequencies, angle
            )
            if polarization.lower() == "te":
                # Reflectance for TE polarization
                R_linear = R_TE
            elif polarization.lower() == "tm":
                # Reflectance for TM polarization
                R_linear = R_TM
            elif polarization.lower() == "both":
                R_linear = (R_TM + R_TE)/2
            else:
                raise ValueError(f"Unknown polarization: {polarization}")
            R_db = 10 * jnp.log10(R_linear).squeeze()
            if angle == 0:
                label_str = f"{prefix}{i}"
            else:
                label_str = f"{prefix}{i} {polarization} angle={angle}°"

            plt.semilogx(freq_ghz, R_db, label=label_str)
            r_max, f_max = get_max_reflection(freq_ghz, R_db, freq_range)
            max_R_db[(i, angle)] = {"freq": f_max, "R_db": r_max}

    for key, value in max_R_db.items():
        print(f"Material {key[0]}, Angle {key[1]}°: Max Reflection = {value['R_db']:.2f} dB at {value['freq']:.2f} GHz")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Reflection [dB]")
    plt.title("Reflection vs. Frequency (All Stacks)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    plot_R_frequency()