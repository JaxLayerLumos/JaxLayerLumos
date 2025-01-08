# random_search.py

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import random
import csv
import os
from datetime import datetime
import sys
import jax
from jaxlayerlumos import stackrt_eps_mu
from jaxlayerlumos import utils_materials

# Import configurations
from optimization_configs import BB, CHF, HF, LF

# Define the date for saving results
today_date = datetime.now().strftime("%Y-%m-%d")

# Define the list of materials (ensure this list aligns with your actual materials)
MATERIALS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'
]

n_materials = len(MATERIALS)  # Number of materials

def encode_material(material, materials_list):
    """
    Encode a material string to an integer based on its position in the materials list.
    """
    return materials_list.index(material)

def decode_material(index, materials_list):
    """
    Decode an integer back to its corresponding material string.
    """
    return materials_list[int(index)]

def get_max_reflection(freq_ghz, R_db, freq_range):
    """
    Compute the maximum reflection and its corresponding frequency within a specified range.
    
    Parameters:
    - freq_ghz: Array of frequencies in GHz.
    - R_db: Reflectance in dB.
    - freq_range: Tuple specifying the (min_freq, max_freq) in GHz.
    
    Returns:
    - r_max: Maximum reflectance in dB within the frequency range.
    - f_max: Frequency at which the maximum reflectance occurs.
    """
    mask = (freq_ghz >= freq_range[0]) & (freq_ghz <= freq_range[1])
    if not jnp.any(mask):
        return -jnp.inf, 0.0  # If no frequencies are within the range
    R_filtered = R_db[mask]
    freq_filtered = freq_ghz[mask]
    r_max = jnp.max(R_filtered)
    f_max = freq_filtered[jnp.argmax(R_filtered)]
    return r_max, f_max

def objective_function(X, n_layers, config, num_freq_points=1000):
    """
    Evaluate the objective functions based on the given design variables.
    
    Parameters:
    - X: A 1D array where the first n_layers elements are material indices and the next n_layers elements are thicknesses.
    - n_layers: Number of layers.
    - config: OptimizationConfig object containing freq_range and inc_angle.
    - num_freq_points: Number of frequency points for simulation.
    
    Returns:
    - A tuple containing the two objective values (rmax, sum_thicknesses).
    """
    # Split X into materials and thicknesses
    layer_materials_encoded = X[:n_layers]
    layer_materials = ['Air'] + [decode_material(index, MATERIALS) for index in layer_materials_encoded] + ['PEC']
    layer_thicknesses = jnp.array([0.0] + list(X[n_layers:]) + [0.0])
    
    # Convert freq_range from GHz to Hz and generate frequency array
    freq_start_hz = config.freq_range[0] * 1e9
    freq_end_hz = config.freq_range[1] * 1e9
    frequencies = jnp.linspace(freq_start_hz, freq_end_hz, num_freq_points)
    freq_ghz = frequencies / 1e9  # Convert back to GHz for get_max_reflection
    
    # Get permittivity and permeability
    eps_stack, mu_stack = utils_materials.get_eps_mu(layer_materials, frequencies)
    
    # Compute reflection for all incidence angles
    max_R_db = -jnp.inf
    print("eps :", eps_stack)
    print("thick : ", layer_thicknesses)
    print("freq : ", frequencies)
    for angle in config.inc_angle:
        R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(
            eps_stack, mu_stack, layer_thicknesses, frequencies, angle
        )
        R_linear = (R_TM + R_TE) / 2

        R_db = 10 * jnp.log10(R_linear).squeeze()
        r_max, _ = get_max_reflection(freq_ghz, R_db, config.freq_range)
        max_R_db = jnp.maximum(max_R_db, r_max)
    
    # Sum of thicknesses
    sum_thicknesses = jnp.sum(layer_thicknesses[1:-1])
    
    return max_R_db, sum_thicknesses

def identify_pareto(scores):
    """
    Identify Pareto front from a set of objective scores.
    
    Parameters:
    - scores: A NumPy array of shape (n_points, n_objectives).
    
    Returns:
    - Indices of points that are on the Pareto front.
    """
    population_size = scores.shape[0]
    population_ids = np.arange(population_size)
    pareto_front = np.ones(population_size, dtype=bool)
    
    for i in range(population_size):
        for j in range(population_size):
            # Assuming maximization for rmax and minimization for sum_thicknesses
            # Adjust the condition based on your objectives
            # Here, we consider rmax to be maximized and sum_thicknesses to be minimized
            if (scores[j][0] >= scores[i][0] and scores[j][1] <= scores[i][1]) and \
               (scores[j][0] > scores[i][0] or scores[j][1] < scores[i][1]):
                pareto_front[i] = False
                break
    return population_ids[pareto_front]

def random_search(num_layers, config, total_evaluations=5000):
    """
    Perform Random Search optimization.
    
    Parameters:
    - num_layers: Number of layers in the design.
    - config: OptimizationConfig object containing freq_range and inc_angle.
    - total_evaluations: Total number of evaluations to perform.
    """
    n_layers = num_layers
    evaluations = 0

    # Lists to store results
    all_train_x = []
    all_train_obj = []

    # Define bounds for thicknesses based on configuration
    thickness_min, thickness_max = (15e-9, 1500e-9)  # Updated bounds

    # Generate initial random data (e.g., 10 samples)
    initial_samples = 10
    for _ in range(initial_samples):
        # Randomly select materials
        cat_samples = [random.randint(0, n_materials - 1) for _ in range(n_layers)]
        # Randomly select thicknesses within bounds [15e-9, 1500e-9]
        cont_samples = [random.uniform(thickness_min, thickness_max) for _ in range(n_layers)]
        # Combine categorical and continuous samples
        sample = cat_samples + cont_samples
        all_train_x.append(sample)
        # Evaluate objective function
        obj = objective_function(sample, n_layers, config)
        all_train_obj.append(obj)
        evaluations += 1

    # Perform random sampling for the remaining evaluations
    while evaluations < total_evaluations:
        # Randomly select materials
        cat_samples = [random.randint(0, n_materials - 1) for _ in range(n_layers)]
        # Randomly select thicknesses within bounds [15e-9, 1500e-9]
        cont_samples = [random.uniform(thickness_min, thickness_max) for _ in range(n_layers)]
        # Combine categorical and continuous samples
        sample = cat_samples + cont_samples
        all_train_x.append(sample)
        # Evaluate objective function
        obj = objective_function(sample, n_layers, config)
        all_train_obj.append(obj)
        evaluations += 1

    # Convert to NumPy arrays for processing
    train_x = np.array(all_train_x)
    train_obj = np.array(all_train_obj)

    # Identify Pareto front
    pareto_indices = identify_pareto(train_obj)
    pareto_parameters = train_x[pareto_indices]
    pareto_objectives = train_obj[pareto_indices]

    # Decode materials
    decoded_materials = [[decode_material(index, MATERIALS) for index in row[:n_layers]] for row in pareto_parameters]
    thicknesses = pareto_parameters[:, n_layers:].tolist()

    # Plot the Pareto frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(train_obj[:, 0], train_obj[:, 1], c="blue", label="All Points", alpha=0.5)
    plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], c="red", label="Pareto Frontier")
    plt.title("Pareto Frontier - Random Search")
    plt.xlabel("Objective 1 (r_max [dB])")
    plt.ylabel("Objective 2 (Sum of Thicknesses [m])")
    plt.legend()
    plt.grid(True)

    # Define plot path
    plot_dir = f'res/RANDOM/{today_date}/{num_layers}layers/{config.name}/'
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'pareto_frontier_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Saving the Pareto frontier in a CSV file
    csv_path = os.path.join(plot_dir, 'pareto_frontier.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Writing headers
        headers = [f'Material{i}' for i in range(1, n_layers + 1)] + \
                  [f'Thickness{i}' for i in range(1, n_layers + 1)] + \
                  ['Objective1 (r_max [dB])', 'Objective2 (Sum Thickness [m])']
        writer.writerow(headers)

        # Writing data rows
        for materials, thickness, objectives in zip(decoded_materials, thicknesses, pareto_objectives.tolist()):
            row = materials + thickness + objectives
            writer.writerow(row)

    # Saving all Random Search data in a CSV file
    all_data_csv_path = os.path.join(plot_dir, 'all_random_search_data.csv')
    with open(all_data_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Headers for all data
        headers = [f'Material{i}' for i in range(1, n_layers + 1)] + \
                  [f'Thickness{i}' for i in range(1, n_layers + 1)] + \
                  ['Objective1 (r_max [dB])', 'Objective2 (Sum Thickness [m])']
        writer.writerow(headers)

        # Writing all data rows
        for i in range(train_x.shape[0]):
            materials = [decode_material(index, MATERIALS) for index in train_x[i, :n_layers]]
            thicknesses = train_x[i, n_layers:].tolist()
            objectives = train_obj[i].tolist()
            row = materials + thicknesses + objectives
            writer.writerow(row)

    print(f"Random Search completed for {num_layers} layers using configuration '{config.name}'. Results saved in 'res/RANDOM/{today_date}/{num_layers}layers/{config.name}/'")

def main():
    """
    Main function to execute Random Search.
    """
    if len(sys.argv) < 2:
        print("Usage: python random_search.py <configuration>")
        print("Available configurations: BB, CHF, HF, LF")
        sys.exit(1)
    
    config_name = sys.argv[1]
    
    # Select the configuration based on user input
    config_map = {
        "BB": BB,
        "CHF": CHF,
        "HF": HF,
        "LF": LF
    }
    
    if config_name not in config_map:
        print(f"Unknown configuration '{config_name}'. Available configurations: BB, CHF, HF, LF")
        sys.exit(1)
    
    config = config_map[config_name]
    num_layers = config.num_layers
    
    # Perform random search
    random_search(num_layers, config, total_evaluations=5)

if __name__ == "__main__":
    main()
