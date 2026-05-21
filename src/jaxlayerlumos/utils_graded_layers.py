"""
Utilities for replacing sharp interfaces with graded material layers.
"""

import jax.numpy as jnp


def get_mixing_ratios(num_graded_layers, method="linear", beta=3.0):
    """
    Return material-2 fractions for true intermediate graded layers.

    The pure endpoint materials are not included in the graded layer set. For
    10 linear graded layers this returns 1/11, 2/11, ..., 10/11.

    Args:
        num_graded_layers (int): Number of true intermediate mixed layers.
        method (str): One of "linear", "sine", or "exponential".
        beta (float): Exponential steepness. beta=0 falls back to linear.
    """
    assert isinstance(num_graded_layers, int)
    assert num_graded_layers >= 1

    x = jnp.arange(1, num_graded_layers + 1) / (num_graded_layers + 1)

    if method == "linear":
        return x
    if method == "sine":
        return jnp.sin(x * jnp.pi / 2)
    if method == "exponential":
        if beta == 0:
            return x
        return jnp.expm1(beta * x) / jnp.expm1(beta)

    raise ValueError(
        "Unsupported graded layer method. Use 'linear', 'sine', or 'exponential'."
    )


def get_graded_layers(
    n_k,
    thicknesses,
    num_graded_layers=10,
    grade_thickness=None,
    method="linear",
    beta=3.0,
    interface_indices=None,
):
    """
    Replace selected sharp interfaces with mixed graded layers.

    Args:
        n_k (jnp.ndarray): Complex refractive indices with shape
            (n_frequencies, n_layers).
        thicknesses (jnp.ndarray): Layer thicknesses with shape (n_layers,).
        num_graded_layers (int): Number of true intermediate mixed layers per
            selected interface. Pure endpoint materials are not counted.
        grade_thickness: Total thickness assigned to each graded interface.
            May be a scalar or a 1D array with one value per selected interface.
        method (str): Mixing method. Use "linear", "sine", or "exponential".
        beta (float): Exponential steepness. Used only when
            method="exponential". beta=0 falls back to linear.
        interface_indices: Optional layer indices i selecting interfaces
            between layer i and layer i + 1. By default, all interfaces whose
            left and right layers both have positive thickness are selected.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: The graded n_k stack and matching
        thickness array, ready to pass to stackrt.
    """
    assert isinstance(n_k, jnp.ndarray)
    assert isinstance(thicknesses, jnp.ndarray)
    assert n_k.ndim == 2
    assert thicknesses.ndim == 1
    assert n_k.shape[1] == thicknesses.shape[0]

    if grade_thickness is None:
        raise ValueError("grade_thickness must be provided.")

    ratios = get_mixing_ratios(num_graded_layers, method=method, beta=beta)
    interface_indices = _get_interface_indices(thicknesses, interface_indices)
    grade_thicknesses = _get_grade_thicknesses(grade_thickness, interface_indices)

    layer_deductions = jnp.zeros_like(thicknesses)
    for interface_index, total_grade_thickness in zip(
        interface_indices, grade_thicknesses
    ):
        half_grade_thickness = total_grade_thickness / 2
        left_thickness = thicknesses[interface_index]
        right_thickness = thicknesses[interface_index + 1]

        if left_thickness < half_grade_thickness:
            raise ValueError("grade_thickness is too large for the left layer.")
        if right_thickness < half_grade_thickness:
            raise ValueError("grade_thickness is too large for the right layer.")

        layer_deductions = layer_deductions.at[interface_index].add(
            half_grade_thickness
        )
        layer_deductions = layer_deductions.at[interface_index + 1].add(
            half_grade_thickness
        )

    adjusted_thicknesses = thicknesses - layer_deductions
    if jnp.any(adjusted_thicknesses < 0):
        raise ValueError("grade_thickness values are too large for the stack.")

    graded_n_k_layers = []
    graded_thickness_layers = []
    grade_by_interface = {
        int(interface_index): grade_thickness
        for interface_index, grade_thickness in zip(
            interface_indices, grade_thicknesses
        )
    }

    for layer_index in range(thicknesses.shape[0]):
        graded_n_k_layers.append(n_k[:, layer_index])
        graded_thickness_layers.append(adjusted_thicknesses[layer_index])

        if layer_index in grade_by_interface:
            total_grade_thickness = grade_by_interface[layer_index]
            layer_grade_thickness = total_grade_thickness / num_graded_layers
            left_n_k = n_k[:, layer_index]
            right_n_k = n_k[:, layer_index + 1]

            for ratio in ratios:
                graded_n_k_layers.append((1 - ratio) * left_n_k + ratio * right_n_k)
                graded_thickness_layers.append(layer_grade_thickness)

    n_k_graded = jnp.stack(graded_n_k_layers, axis=1)
    thicknesses_graded = jnp.array(graded_thickness_layers, dtype=thicknesses.dtype)

    return n_k_graded, thicknesses_graded


def _get_interface_indices(thicknesses, interface_indices):
    if interface_indices is None:
        return [
            layer_index
            for layer_index in range(thicknesses.shape[0] - 1)
            if thicknesses[layer_index] > 0 and thicknesses[layer_index + 1] > 0
        ]

    interface_indices = jnp.asarray(interface_indices)
    assert interface_indices.ndim == 1

    indices = [int(interface_index) for interface_index in interface_indices]
    for interface_index in indices:
        if interface_index < 0 or interface_index >= thicknesses.shape[0] - 1:
            raise ValueError("interface_indices contains an invalid interface.")

    if len(indices) != len(set(indices)):
        raise ValueError("interface_indices must not contain duplicates.")

    return indices


def _get_grade_thicknesses(grade_thickness, interface_indices):
    grade_thicknesses = jnp.asarray(grade_thickness)

    if grade_thicknesses.ndim == 0:
        grade_thicknesses = jnp.repeat(grade_thicknesses, len(interface_indices))
    else:
        assert grade_thicknesses.ndim == 1
        assert grade_thicknesses.shape[0] == len(interface_indices)

    if jnp.any(grade_thicknesses < 0):
        raise ValueError("grade_thickness values must be non-negative.")

    return grade_thicknesses
