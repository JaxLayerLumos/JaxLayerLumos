# Two_level_jax.py

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
import sys
import optax  # Ensure Optax is installed: pip install optax
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

def objective(
    thicknesses,
    material_layout,
    frequencies,
    inc_angles,
    freq_range,
    materials_list
):
    """
    Objective function for optimization.
    
    Parameters:
    - thicknesses: JAX array of thicknesses (including boundary layers).
    - material_layout: list of material indices (integers).
    - frequencies: Array of frequencies for simulation (in Hz).
    - inc_angles: List of incidence angles in degrees.
    - freq_range: Tuple specifying the (min_freq, max_freq) in GHz for r_max calculation.
    - materials_list: List of material strings.
    
    Returns:
    - combined_obj: Scalar value representing the combined objective.
    - (max_R_db, sum_thicknesses): Tuple of individual objectives.
    """
    # Extract internal thicknesses
    internal_thicknesses = thicknesses[1:-1]
    
    # Decode material stack
    layer_materials = ['Air'] + [decode_material(index, materials_list) for index in material_layout] + ['PEC']
    
    # Get permittivity and permeability
    eps_stack, mu_stack = utils_materials.get_eps_mu(layer_materials, frequencies)
    
    # Compute reflection for all incidence angles
    max_R_db = -jnp.inf
    # Debugging statements
    # print("eps :", eps_stack)
    # print("thick : ", thicknesses[0], thicknesses[-1])
    # print("freq : ", frequencies)
    for angle in inc_angles:
        R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(
            eps_stack, mu_stack, thicknesses, frequencies, angle
        )
        R_linear = (R_TM + R_TE) / 2
        R_db = 10 * jnp.log10(R_linear).squeeze()
        r_max, _ = get_max_reflection(frequencies / 1e9, R_db, freq_range)
        max_R_db = jnp.maximum(max_R_db, r_max)
    
    # Sum of thicknesses
    sum_thicknesses = jnp.sum(internal_thicknesses)
    
    # Combined objective: maximize rmax and minimize sum_thicknesses
    combined_obj = -max_R_db + sum_thicknesses  # Adjust signs based on your optimization goal
    
    return combined_obj, (max_R_db, sum_thicknesses)

def optimize_thicknesses_optax(
    material_layout, 
    initial_thicknesses_with_boundaries, 
    bounds, 
    frequencies,
    inc_angles,
    polarization,
    prefix,
    freq_range,
    materials_list,
    max_iters=1000, 
    tol=1e-6
):
    """
    Optimize thicknesses for a given material layout using Optax's Adam optimizer and record trajectory.
    
    Parameters:
    - material_layout: list of material indices (integers).
    - initial_thicknesses_with_boundaries: initial thicknesses with boundary layers (including zeros).
    - bounds: list of tuples specifying (min, max) for each internal layer.
    - frequencies: Array of frequencies for simulation (in Hz).
    - inc_angles: List of incidence angles in degrees.
    - polarization: Polarization type ('te', 'tm', 'both').
    - prefix: Prefix string for labeling.
    - freq_range: Tuple specifying the (min_freq, max_freq) in GHz for r_max calculation.
    - materials_list: List of material strings.
    - max_iters: maximum number of iterations.
    - tol: tolerance for convergence based on gradient norm.
    
    Returns:
    - optimized_thicknesses: NumPy array of optimized thicknesses (internal layers only).
    - objectives: list of objective values [rmax, sum_thicknesses].
    - trajectory_evaluations: list of tuples (params, objectives) for each iteration.
    """
    # Initialize parameters as JAX array (including boundary layers)
    params = jnp.array(initial_thicknesses_with_boundaries)
    
    # Create a mask: False for boundary layers, True for internal layers
    mask = jnp.array([False] + [True] * (len(params) - 2) + [False])
    
    # Initialize the Adam optimizer
    optimizer = optax.adam(learning_rate=1e-3)  # Adjust the learning rate as needed
    opt_state = optimizer.init(params)
    
    # Initialize trajectory evaluations list for this run
    trajectory_evaluations = []
    
    # Extract lower and upper bounds for internal layers
    lower_bounds = jnp.array([b[0] for b in bounds])
    upper_bounds = jnp.array([b[1] for b in bounds])
    
    # Define the objective function with fixed parameters
    def objective_fixed(thicknesses):
        return objective(thicknesses, material_layout, frequencies, inc_angles, freq_range, materials_list)
    
    # JIT compile the objective and its gradient
    objective_jit = jax.jit(objective_fixed)
    grad_fn = jax.jit(jax.grad(lambda x: objective_jit(x)[0]))
    
    for step in range(max_iters):
        # Compute objective and gradients
        scalar_obj, objectives = objective_jit(params)
        gradient = grad_fn(params)
        
        # Debugging: Print the values
        print(f"Iteration {step+1}: r_max={objectives[0]:.4f} dB, Sum Thickness={objectives[1]:.4e} m")
        
        # Append to trajectory evaluations
        trajectory_evaluations.append((np.array(params), objectives))
        
        # Compute updates
        updates, opt_state = optimizer.update(gradient, opt_state, params)
        
        # Apply updates
        params = optax.apply_updates(params, updates)
        
        # Enforce bounds on internal layers
        internal_layers = params[1:-1]
        clamped_internal = jnp.clip(internal_layers, lower_bounds, upper_bounds)
        params = jnp.concatenate([params[:1], clamped_internal, params[-1:]])
        
        # Compute gradient norm for convergence
        grad_norm = jnp.linalg.norm(gradient).item()
        if grad_norm < tol:
            print(f"Converged at step {step+1} with gradient norm {grad_norm:.4e}")
            break
        # Print progress every 100 steps and the first 5 steps for short iterations
        if (step+1) % 100 == 0 or step < 5:
            print(f"Step {step+1}: Combined Objective = {scalar_obj.item():.4f}, Gradient Norm = {grad_norm:.4e}")
    
    # After optimization, compute final objectives
    final_obj, final_objectives = objective_jit(params)
    objectives = list(final_objectives)
    
    # Extract only the internal thicknesses (exclude boundaries)
    optimized_thicknesses = np.array(params[1:-1])
    
    return optimized_thicknesses, objectives, trajectory_evaluations

def identify_pareto(scores):
    """
    Identify the indices of Pareto optimal points.
    
    Parameters:
    - scores: A 2D NumPy array where each row represents an objective vector.
    
    Returns:
    - pareto_front_indices: Indices of Pareto optimal points.
    """
    population_size = scores.shape[0]
    population_ids = np.arange(population_size)
    pareto_front = np.ones(population_size, dtype=bool)
    
    for i in range(population_size):
        for j in range(population_size):
            # Assuming maximization for rmax and minimization for sum_thicknesses
            if (scores[j][0] >= scores[i][0] and scores[j][1] <= scores[i][1]) and \
               (scores[j][0] > scores[i][0] or scores[j][1] < scores[i][1]):
                pareto_front[i] = False
                break
    return population_ids[pareto_front]

def read_initial_points(csv_path, num_layers):
    """
    Read initial material layouts and thicknesses from a CSV file.
    
    Parameters:
    - csv_path: Path to the CSV file.
    - num_layers: Number of internal layers to read.
    
    Returns:
    - initial_points: List of tuples containing (material_layout_indices, thicknesses_with_boundaries).
    """
    initial_points = []
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        material_cols = [col for col in reader.fieldnames if col.startswith('Material')]
        thickness_cols = [col for col in reader.fieldnames if col.startswith('Thickness')]

        for row_num, row in enumerate(reader, start=1):
            if len(material_cols) < num_layers or len(thickness_cols) < num_layers:
                print(f"Row {row_num} has insufficient number of layers. Expected {num_layers}, got {len(material_cols)} materials and {len(thickness_cols)} thicknesses.")
                continue
            try:
                material_layout = [encode_material(row[col], MATERIALS) for col in material_cols[:num_layers]]
                thicknesses = [float(row[col]) for col in thickness_cols[:num_layers]]
                # Append zeros for boundary layers
                thicknesses_with_boundaries = [0.0] + thicknesses + [0.0]
                initial_points.append((material_layout, thicknesses_with_boundaries))
            except ValueError as e:
                print(f"Error parsing row {row_num}: {e}")
                continue
    return initial_points

def two_level_optimization(
    num_layers, 
    initial_points, 
    total_evaluations=None, 
    max_iters=1000, 
    tol=1e-6,
    config=None  # Accept configuration object
):
    """
    Perform two-level optimization:
    - Level 1: Use provided material layouts from initial_points.
    - Level 2: For each material layout, optimize thicknesses using Optax's Adam optimizer and record trajectory.
    
    Parameters:
    - num_layers: Number of layers in the design.
    - initial_points: List of tuples containing (material_layout_indices, thicknesses_with_boundaries).
    - total_evaluations: Total number of material layouts to evaluate. If None, evaluate all initial_points.
    - max_iters: Maximum number of iterations per optimization.
    - tol: Tolerance for convergence based on gradient norm.
    - config: OptimizationConfig object containing freq_range, num_layers, inc_angle
    """
    evaluations = 0
    
    # Lists to store results
    all_train_materials = []
    all_train_thicknesses = []
    all_train_obj = []
    all_trajectories = []  # To store optimization trajectories
    all_evaluations = []   # List of lists: each sublist contains (params, objectives) tuples for one run
    
    # Determine how many evaluations to perform
    if total_evaluations is None:
        total_evaluations = len(initial_points)
    else:
        total_evaluations = min(total_evaluations, len(initial_points))
    
    # Define bounds for all internal layers
    bounds = [(15e-9, 1500e-9) for _ in range(num_layers)]
    
    # Define simulation parameters based on configuration
    freq_min, freq_max = config.freq_range
    frequencies = jnp.linspace(freq_min * 1e9, freq_max * 1e9, 1000)  # Updated to 1000 points
    inc_angles = config.inc_angle
    polarization = "both"     # 'te', 'tm', or 'both'
    prefix = "Layer"          # Prefix for labeling
    freq_range = config.freq_range  # Frequency range in GHz for r_max calculation
    
    # Iterate over initial points
    for i, (material_layout, initial_thicknesses_with_boundaries) in enumerate(initial_points[:total_evaluations]):
        # Optimize thicknesses
        optimized_thicknesses, objectives, trajectory_evaluations = optimize_thicknesses_optax(
            material_layout, 
            initial_thicknesses_with_boundaries, 
            bounds=bounds,  # Pass updated bounds to the optimizer
            frequencies=frequencies,
            inc_angles=inc_angles,
            polarization=polarization,
            prefix=prefix,
            freq_range=freq_range,
            materials_list=MATERIALS,
            max_iters=max_iters, 
            tol=tol
        )
    
        # Store results
        all_train_materials.append(material_layout)
        all_train_thicknesses.append(optimized_thicknesses)
        all_train_obj.append(objectives)
        all_trajectories.append(trajectory_evaluations.copy())
        all_evaluations.append(trajectory_evaluations.copy())  # Each trajectory_evaluations is a list of tuples
        evaluations += 1
        print(f"Evaluated material layout {evaluations}/{total_evaluations} using configuration '{config.name}'")
    
    # Convert to NumPy arrays for processing
    train_materials = np.array(all_train_materials)
    train_thicknesses = np.array(all_train_thicknesses)
    train_obj = np.array(all_train_obj)
    
    # Identify Pareto front
    pareto_indices = identify_pareto(train_obj)
    pareto_materials = train_materials[pareto_indices]
    pareto_thicknesses = train_thicknesses[pareto_indices]
    pareto_objectives = train_obj[pareto_indices]
    
    # Decode materials
    decoded_materials = [[decode_material(index, MATERIALS) for index in row] for row in pareto_materials]
    
    # Plot the Pareto frontier
    plt.figure(figsize=(12, 8))
    # plt.scatter(train_obj[:, 0], train_obj[:, 1], c="blue", label="All Points", alpha=0.5)
    plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], c="red", label="Pareto Frontier", edgecolors='k')
    
    # Plot optimization trajectories
    cmap = plt.get_cmap("tab20")
    for idx, trajectory in enumerate(all_trajectories):
        if len(trajectory) == 0:
            continue  # Skip if no evaluations were recorded
        traj_rmax = [step[1][0] for step in trajectory]    # Objective1 (rmax)
        traj_sum_thick = [step[1][1] for step in trajectory]  # Objective2 (Sum Thickness)
        # Debugging: Print lengths
        print(f"Trajectory {idx+1}: traj_rmax length = {len(traj_rmax)}, traj_sum_thick length = {len(traj_sum_thick)}")
        
        if len(traj_rmax) != len(traj_sum_thick):
            print(f"Warning: Trajectory {idx+1} has mismatched lengths. Skipping plot for this trajectory.")
            continue  # Skip plotting this trajectory
        
        color = cmap(idx % 20)  # Cycle through colors if more than 20 trajectories
        plt.plot(traj_rmax, traj_sum_thick, marker='o', color=color, alpha=0.7, linewidth=1)
    
    plt.title("Pareto Frontier with Optimization Trajectories - Two-Level Optimization")
    plt.xlabel("Objective 1 (r_max [dB])")
    plt.ylabel("Objective 2 (Sum of Thicknesses [m])")
    plt.legend(loc='upper right', fontsize='small', ncol=2, markerscale=1, framealpha=0.9)
    plt.grid(True)
    
    # Define plot path
    plot_dir = f'res/TWO_LEVEL/{today_date}/{num_layers}layers/{config.name}/'  # Updated to use config.name
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'pareto_frontier_plot_with_trajectories.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Saving the Pareto frontier in a CSV file
    csv_path = os.path.join(plot_dir, 'pareto_frontier.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Writing headers
        headers = [f'Material{i}' for i in range(1, num_layers + 1)] + \
                  [f'Thickness{i}' for i in range(1, num_layers + 1)] + \
                  ['Objective1 (r_max [dB])', 'Objective2 (Sum Thickness [m])']
        writer.writerow(headers)
    
        # Writing data rows
        for materials, thickness, objectives in zip(decoded_materials, pareto_thicknesses.tolist(), pareto_objectives.tolist()):
            row = materials + thickness.tolist() + objectives
            writer.writerow(row)
    
    # Saving all optimization data in a CSV file
    all_data_csv_path = os.path.join(plot_dir, 'all_two_level_optimization_data.csv')
    with open(all_data_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Writing headers
        headers = [f'Material{i}' for i in range(1, num_layers + 1)] + \
                  [f'Thickness{i}' for i in range(1, num_layers + 1)] + \
                  ['Objective1 (r_max [dB])', 'Objective2 (Sum Thickness [m])']
        writer.writerow(headers)
    
        # Writing all data rows
        for i in range(train_materials.shape[0]):
            materials = [decode_material(index, MATERIALS) for index in train_materials[i]]
            thicknesses = train_thicknesses[i]
            objectives = train_obj[i]
            row = materials + thicknesses.tolist() + objectives.tolist()
            writer.writerow(row)
    
    print(f"Pareto frontier saved in '{csv_path}'")
    print(f"All optimization data saved in '{all_data_csv_path}'")
    print(f"Pareto frontier plot with trajectories saved in '{plot_path}'")
    
    # Optionally, save trajectories to a CSV file for future plotting
    traj_csv_path = os.path.join(plot_dir, 'optimization_trajectories.csv')
    with open(traj_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header can include trajectory ID, iteration, thicknesses, objectives
        thickness_headers = [f'Thickness{i}' for i in range(1, num_layers + 1)]
        objective_headers = ['Objective1 (r_max [dB])', 'Objective2 (Sum Thickness [m])']
        header = ['Trajectory_ID', 'Iteration'] + thickness_headers + objective_headers
        writer.writerow(header)
        traj_id = 1
        for trajectory in all_evaluations:
            for iter_id, (params, objectives) in enumerate(trajectory, start=1):
                row = [traj_id, iter_id] + list(params[1:-1]) + list(objectives)
                writer.writerow(row)
            traj_id += 1
    print(f"Optimization trajectories saved in '{traj_csv_path}'")

def main():
    """
    Main function to execute Two-Level Optimization.
    """
    if len(sys.argv) < 3:
        print("Usage: python Two_level_jax.py <configuration> <initial_points_csv_path>")
        print("Available configurations: BB, CHF, HF, LF")
        sys.exit(1)
    
    config_name = sys.argv[1]
    initial_points_csv = sys.argv[2]
    
    if not os.path.isfile(initial_points_csv):
        print(f"The file {initial_points_csv} does not exist.")
        sys.exit(1)
    
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
    
    # Read initial points
    initial_points = read_initial_points(initial_points_csv, config.num_layers)
    if not initial_points:
        print("No valid initial points found in the CSV file.")
        sys.exit(1)
    
    # Perform optimization using the selected configuration
    two_level_optimization(
        num_layers=config.num_layers, 
        initial_points=initial_points, 
        total_evaluations=10, 
        max_iters=30,
        tol=1e-6,
        config=config  # Pass the entire configuration if needed
    )

if __name__ == "__main__":
    main()
