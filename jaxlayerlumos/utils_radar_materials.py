"""
Radar material properties for electromagnetic calculations.

This module provides functions for accessing radar material properties including
permittivity and permeability data based on the Michielssen database. It supports
16 different radar materials with frequency-dependent properties in the GHz range.
"""

import jax
import jax.numpy as jnp
import numpy as onp

from jaxlayerlumos import utils_graded_layers
from jaxlayerlumos import utils_units


def get_eps_mu_Michielssen(material_indices, frequencies):
    """
    Get permittivity and permeability for radar materials from the Michielssen database.
    
    This function provides frequency-dependent permittivity and permeability values
    for 16 different radar materials. The materials include various types of
    absorbers and coatings commonly used in radar applications.
    
    Args:
        material_indices (onp.ndarray): Material indices (1-16) corresponding to
                                       different radar materials.
        frequencies (jnp.ndarray): Frequencies in Hz for which to calculate properties.
    
    Returns:
        tuple: (eps_r, mu_r) - Relative permittivity and permeability arrays
               with shape (n_materials, n_frequencies).
    
    Note:
        - Material indices must be between 1 and 16
        - Frequencies are converted to GHz internally for the calculations
        - Materials 1-5 have constant permittivity and frequency-dependent permeability
        - Materials 6-8 have frequency-dependent permittivity and constant permeability
        - Materials 9-16 have frequency-dependent permittivity and permeability
    """
    assert isinstance(material_indices, onp.ndarray)
    assert isinstance(frequencies, jnp.ndarray)
    assert material_indices.ndim == 1
    assert frequencies.ndim == 1
    for material_index in material_indices:
        assert material_index in onp.arange(1, 17)

    # Gets parameters from Michiellsen
    f = frequencies / utils_units.get_giga()  # in GHz
    M_epsr = jnp.vstack(
        [
            jnp.tile(
                jnp.array([10, 50, 15, 15, 15])[:, None], (1, len(f))
            ),  # Materials 1 to 5
            jnp.array(
                [  # Frequency-dependent permittivity for materials 6 to 8
                    5 / (f**0.861) - 1j * (8 / (f**0.569)),
                    8 / (f**0.778) - 1j * (10 / (f**0.682)),
                    10 / (f**0.778) - 1j * (6 / (f**0.861)),
                ]
            ),
            jnp.full((8, len(f)), 15, dtype=complex),  # Materials 9 to 16
        ]
    )

    # Fill constant values for permeability (mur)
    M_mur = jnp.vstack(
        [
            jnp.ones((2, len(f))),  # Materials 1 and 2
            jnp.array(
                [  # Frequency-dependent permeability for materials 3 to 5
                    5 / (f**0.974) - 1j * (10 / (f**0.961)),
                    3 / (f**1.0) - 1j * (15 / (f**0.957)),
                    7 / (f**1.0) - 1j * (12 / (f**1.0)),
                ]
            ),
            jnp.ones((3, len(f))),  # Materials 6 to 8
            jnp.array(
                [  # Frequency-dependent permeability for materials 9 to 16
                    (35 * (0.8**2)) / (f**2 + 0.8**2)
                    - 1j * (35 * 0.8 * f) / (f**2 + 0.8**2),
                    (35 * (0.5**2)) / (f**2 + 0.5**2)
                    - 1j * (35 * 0.5 * f) / (f**2 + 0.5**2),
                    (30 * (1**2)) / (f**2 + 1**2) - 1j * (30 * f) / (f**2 + 1**2),
                    (18 * (0.5**2)) / (f**2 + 0.5**2)
                    - 1j * (18 * 0.5 * f) / (f**2 + 0.5**2),
                    (20 * (1.5**2)) / (f**2 + 1.5**2)
                    - 1j * (20 * 1.5 * f) / (f**2 + 1.5**2),
                    (30 * (2.5**2)) / (f**2 + 2.5**2)
                    - 1j * (30 * 2.5 * f) / (f**2 + 2.5**2),
                    (30 * (2**2)) / (f**2 + 2**2) - 1j * (30 * 2 * f) / (f**2 + 2**2),
                    (25 * (3.5**2)) / (f**2 + 3.5**2)
                    - 1j * (25 * 3.5 * f) / (f**2 + 3.5**2),
                ]
            ),
        ]
    )

    # Initialize epsr and mur for the given material_indices
    eps_r = M_epsr[material_indices - 1, :]  # Python uses 0-based indexing
    mu_r = M_mur[material_indices - 1, :]

    return eps_r, mu_r


def get_air_mixed_graded_stack(
    material_index,
    frequencies,
    total_thickness_m,
    num_graded_layers,
    fractions=None,
    thicknesses_m=None,
):
    """
    Return an Air | material-air graded layers | PEC stack for radar materials.

    Args:
        material_index (int): Michielssen radar material index from 1 to 16.
        frequencies (jnp.ndarray): Frequencies in Hz with shape (n_frequencies,).
        total_thickness_m: Total physical thickness of the graded region in meters.
        num_graded_layers (int): Number of equal-thickness graded sublayers.
        fractions: Optional material fractions for each graded layer. Defaults to
            the linear fractions from utils_graded_layers.get_mixing_ratios.
        thicknesses_m: Optional physical thickness for each graded sublayer.
            Defaults to equal sublayer thicknesses.

    Returns:
        tuple: (eps_stack, mu_stack, thicknesses_m, fractions)
            eps_stack and mu_stack have shape (n_frequencies, num_graded_layers + 2).
            thicknesses_m has shape (num_graded_layers + 2,), with zero-thickness
            Air and PEC boundary layers.
    """
    assert isinstance(material_index, int)
    assert material_index in onp.arange(1, 17)
    assert isinstance(frequencies, jnp.ndarray)
    assert frequencies.ndim == 1
    assert isinstance(num_graded_layers, int)
    assert num_graded_layers >= 1

    total_thickness_m = float(total_thickness_m)
    if total_thickness_m < 0:
        raise ValueError("total_thickness_m must be non-negative.")

    fill_fractions = _get_fill_fractions(num_graded_layers, fractions)
    graded_thicknesses_m = _get_graded_thicknesses_m(
        total_thickness_m, num_graded_layers, thicknesses_m
    )

    eps_stack, mu_stack = _build_air_mixed_graded_stack_eps_mu(
        material_index, frequencies, fill_fractions
    )

    return eps_stack, mu_stack, graded_thicknesses_m, fill_fractions


def _get_fill_fractions(num_graded_layers, fractions):
    return utils_graded_layers.get_mixing_ratios(
        num_graded_layers, method="linear", fractions=fractions
    )


def _get_graded_thicknesses_m(total_thickness_m, num_graded_layers, thicknesses_m):
    if thicknesses_m is None:
        sublayer_thickness_m = total_thickness_m / num_graded_layers
        inner_thicknesses_m = jnp.array([sublayer_thickness_m] * num_graded_layers)
    else:
        inner_thicknesses_m = jnp.asarray(thicknesses_m)
        assert inner_thicknesses_m.ndim == 1
        assert inner_thicknesses_m.shape[0] == num_graded_layers
        if jnp.any(inner_thicknesses_m <= 0):
            raise ValueError("thicknesses_m must be positive.")
        if not onp.isclose(
            float(jnp.sum(inner_thicknesses_m)),
            float(total_thickness_m),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("thicknesses_m must sum to total_thickness_m.")

    return jnp.concatenate([jnp.array([0.0]), inner_thicknesses_m, jnp.array([0.0])])


def _build_air_mixed_graded_stack_eps_mu(material_index, frequencies, fill_fractions):
    eps_material, mu_material = get_eps_mu_Michielssen(
        onp.array([material_index]), frequencies
    )
    eps_material = eps_material[0]
    mu_material = mu_material[0]
    eps_air = jnp.ones_like(eps_material)
    mu_air = jnp.ones_like(mu_material)
    eps_pec = jnp.zeros_like(eps_material) + jnp.inf
    mu_pec = jnp.ones_like(mu_material)

    fill = fill_fractions[jnp.newaxis, :]
    eps_graded = (1 - fill) * eps_air[:, jnp.newaxis] + fill * eps_material[
        :, jnp.newaxis
    ]
    mu_graded = (1 - fill) * mu_air[:, jnp.newaxis] + fill * mu_material[
        :, jnp.newaxis
    ]

    eps_stack = jnp.concatenate(
        [eps_air[:, jnp.newaxis], eps_graded, eps_pec[:, jnp.newaxis]], axis=1
    )
    mu_stack = jnp.concatenate(
        [mu_air[:, jnp.newaxis], mu_graded, mu_pec[:, jnp.newaxis]], axis=1
    )

    return eps_stack, mu_stack


def raw_parameters_to_graded_profile(
    raw_fractions, raw_thicknesses, total_thickness_m
):
    """
    Convert unconstrained raw optimization variables to a graded RAM profile.

    The final graded sublayer fraction is fixed to 1.0 so the layer next to the
    PEC boundary is pure material. This avoids non-finite gradients through the
    infinite PEC permittivity while keeping all thicknesses optimizable.
    """
    raw_fractions = jnp.asarray(raw_fractions)
    raw_thicknesses = jnp.asarray(raw_thicknesses)
    assert raw_fractions.ndim == 1
    assert raw_thicknesses.ndim == 1
    assert raw_fractions.shape[0] == raw_thicknesses.shape[0] - 1

    fractions = jnp.concatenate(
        [jax.nn.sigmoid(raw_fractions), jnp.array([1.0], dtype=raw_thicknesses.dtype)]
    )
    thicknesses_m = float(total_thickness_m) * jax.nn.softmax(raw_thicknesses)

    return fractions, thicknesses_m


def reflection_linear_air_mixed_graded_stack(
    material_index,
    frequencies,
    total_thickness_m,
    fractions,
    thicknesses_m,
    inc_angle_deg,
):
    """Return linear reflection for Air | graded material-air sublayers | PEC."""
    eps_stack, mu_stack = _build_air_mixed_graded_stack_eps_mu(
        material_index, frequencies, fractions
    )
    stack_thicknesses_m = jnp.concatenate(
        [jnp.array([0.0]), jnp.asarray(thicknesses_m), jnp.array([0.0])]
    )
    R_TE, R_TM = _stackrt_eps_mu_reflection_no_assert(
        eps_stack, mu_stack, stack_thicknesses_m, frequencies, inc_angle_deg
    )
    return jnp.maximum(((R_TE + R_TM) / 2.0).squeeze(), 1e-300)


def _stackrt_eps_mu_reflection_no_assert(eps_r, mu_r, d, f, theta):
    theta_rad = jnp.radians(theta)
    fun_mapped = jax.vmap(
        _stackrt_eps_mu_base_reflection_no_assert,
        (0, 0, None, 0, None),
        (0, 0),
    )
    return fun_mapped(eps_r, mu_r, d, f, theta_rad)


def _stackrt_eps_mu_base_reflection_no_assert(eps_r, mu_r, thicknesses, f_i, thetas_k):
    c = utils_units.get_light_speed()
    n = jnp.conj(jnp.sqrt(eps_r * mu_r))
    k = 2 * jnp.pi / c * f_i * n
    eta = jnp.conj(jnp.sqrt(mu_r / eps_r))

    sin_theta = jnp.expand_dims(jnp.sin(thetas_k), axis=0)
    sin_theta = sin_theta * n[0] / n
    cos_theta_t = jnp.sqrt(1 - sin_theta**2)
    kz = k * cos_theta_t

    upper_bound = 600.0
    delta = thicknesses * kz
    delta = jnp.real(delta) + 1j * jnp.clip(jnp.imag(delta), -upper_bound, upper_bound)

    Z_TE = eta / cos_theta_t
    Z_TM = eta * cos_theta_t

    r_jk_TE = (Z_TE[1:] - Z_TE[:-1]) / (Z_TE[1:] + Z_TE[:-1])
    t_jk_TE = (2 * Z_TE[1:]) / (Z_TE[1:] + Z_TE[:-1])

    r_jk_TM = (Z_TM[1:] - Z_TM[:-1]) / (Z_TM[1:] + Z_TM[:-1])
    t_jk_TM = (
        (2 * Z_TM[1:]) / (Z_TM[1:] + Z_TM[:-1]) * cos_theta_t[:-1] / cos_theta_t[1:]
    )

    pec_boundary = (
        jnp.isinf(jnp.real(eps_r[-1]))
        & jnp.isclose(jnp.real(mu_r[-1]), 1)
        & jnp.isclose(jnp.imag(mu_r[-1]), 0)
    )
    r_jk_TE = jax.lax.cond(
        pec_boundary, lambda value: value.at[-1].set(-1.0), lambda value: value, r_jk_TE
    )
    t_jk_TE = jax.lax.cond(
        pec_boundary, lambda value: value.at[-1].set(1.0), lambda value: value, t_jk_TE
    )
    r_jk_TM = jax.lax.cond(
        pec_boundary, lambda value: value.at[-1].set(-1.0), lambda value: value, r_jk_TM
    )
    t_jk_TM = jax.lax.cond(
        pec_boundary, lambda value: value.at[-1].set(1.0), lambda value: value, t_jk_TM
    )

    t_inv_TE = 1 / t_jk_TE
    r_over_t_TE = r_jk_TE / t_jk_TE
    D_TE = jnp.stack(
        [
            jnp.stack([t_inv_TE, r_over_t_TE], axis=-1),
            jnp.stack([r_over_t_TE, t_inv_TE], axis=-1),
        ],
        axis=-2,
    )

    t_inv_TM = 1 / t_jk_TM
    r_over_t_TM = r_jk_TM / t_jk_TM
    D_TM = jnp.stack(
        [
            jnp.stack([t_inv_TM, r_over_t_TM], axis=-1),
            jnp.stack([r_over_t_TM, t_inv_TM], axis=-1),
        ],
        axis=-2,
    )

    exp_neg_jdelta = jnp.exp(-1j * delta[0:-1])
    exp_pos_jdelta = jnp.exp(1j * delta[0:-1])
    zeros = jnp.zeros_like(exp_neg_jdelta)
    P = jnp.stack(
        [
            jnp.stack([exp_neg_jdelta, zeros], axis=-1),
            jnp.stack([zeros, exp_pos_jdelta], axis=-1),
        ],
        axis=-2,
    )

    DP_TE = jnp.matmul(P, D_TE)
    DP_TM = jnp.matmul(P, D_TM)

    def matmul_scan(a, b):
        return jnp.matmul(a, b)

    M_TE = jax.lax.associative_scan(matmul_scan, DP_TE)[-1]
    M_TM = jax.lax.associative_scan(matmul_scan, DP_TM)[-1]

    r_TE_i = M_TE[1, 0] / M_TE[0, 0]
    r_TM_i = M_TM[1, 0] / M_TM[0, 0]

    return jnp.abs(r_TE_i) ** 2, jnp.abs(r_TM_i) ** 2


def lp_reflection_objective(
    material_index,
    raw_fractions,
    raw_thicknesses,
    frequencies,
    total_thickness_m,
    p_norm,
    inc_angle_deg,
):
    fractions, thicknesses_m = raw_parameters_to_graded_profile(
        raw_fractions, raw_thicknesses, total_thickness_m
    )
    reflection_linear = reflection_linear_air_mixed_graded_stack(
        material_index,
        frequencies,
        total_thickness_m,
        fractions,
        thicknesses_m,
        inc_angle_deg,
    )
    return jnp.mean(reflection_linear**p_norm) ** (1.0 / p_norm)


def initial_linear_fraction_raws(num_graded_layers):
    """Return logit parameters for linear fractions with final layer fixed at 1."""
    assert isinstance(num_graded_layers, int)
    assert num_graded_layers >= 1
    if num_graded_layers == 1:
        return jnp.array([])

    fractions = jnp.arange(1, num_graded_layers) / (num_graded_layers + 1)
    return jnp.log(fractions / (1.0 - fractions))


def optimize_air_mixed_graded_stack(
    material_index,
    frequencies,
    total_thickness_m,
    num_graded_layers,
    p_norm,
    inc_angle_deg,
    num_steps,
    learning_rate,
    fraction_history_interval=200,
    worst_history_interval=50,
    return_history=True,
):
    """Optimize fractions and sublayer thicknesses for one radar material."""
    raw_fractions = initial_linear_fraction_raws(num_graded_layers)
    raw_thicknesses = jnp.zeros(num_graded_layers)

    def objective(raw_fractions, raw_thicknesses):
        return lp_reflection_objective(
            material_index,
            raw_fractions,
            raw_thicknesses,
            frequencies,
            total_thickness_m,
            p_norm,
            inc_angle_deg,
        )

    value_and_grad = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
    fraction_history_steps = []
    fraction_history = []
    fraction_thickness_history_m = []
    worst_reflection_history_steps = []
    worst_reflection_history_db = []

    def record_history(step, current_raw_fractions, current_raw_thicknesses):
        fractions, thicknesses_m = raw_parameters_to_graded_profile(
            current_raw_fractions, current_raw_thicknesses, total_thickness_m
        )

        if (
            return_history
            and fraction_history_interval
            and (
                step == 0
                or step == int(num_steps)
                or step % int(fraction_history_interval) == 0
            )
        ):
            fraction_history_steps.append(int(step))
            fraction_history.append(onp.asarray(fractions))
            fraction_thickness_history_m.append(onp.asarray(thicknesses_m))

        if (
            return_history
            and worst_history_interval
            and (
                step == 0
                or step == int(num_steps)
                or step % int(worst_history_interval) == 0
            )
        ):
            reflection_linear = reflection_linear_air_mixed_graded_stack(
                material_index,
                frequencies,
                total_thickness_m,
                fractions,
                thicknesses_m,
                inc_angle_deg,
            )
            worst_reflection_db = 10 * jnp.log10(jnp.max(reflection_linear))
            worst_reflection_history_steps.append(int(step))
            worst_reflection_history_db.append(float(worst_reflection_db))

    record_history(0, raw_fractions, raw_thicknesses)

    try:
        import optax

        optimizer = optax.adam(learning_rate)
        params = (raw_fractions, raw_thicknesses)
        opt_state = optimizer.init(params)
        for step in range(1, int(num_steps) + 1):
            value, grads = value_and_grad(params[0], params[1])
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            record_history(step, params[0], params[1])
        raw_fractions, raw_thicknesses = params
        objective_value = value
    except ImportError:
        objective_value = None
        for step in range(1, int(num_steps) + 1):
            objective_value, grads = value_and_grad(raw_fractions, raw_thicknesses)
            raw_fractions = raw_fractions - learning_rate * grads[0]
            raw_thicknesses = raw_thicknesses - learning_rate * grads[1]
            record_history(step, raw_fractions, raw_thicknesses)

    fractions, thicknesses_m = raw_parameters_to_graded_profile(
        raw_fractions, raw_thicknesses, total_thickness_m
    )
    reflection_linear = reflection_linear_air_mixed_graded_stack(
        material_index,
        frequencies,
        total_thickness_m,
        fractions,
        thicknesses_m,
        inc_angle_deg,
    )
    objective_value = jnp.mean(reflection_linear**p_norm) ** (1.0 / p_norm)

    return {
        "material_id": int(material_index),
        "raw_fractions": raw_fractions,
        "raw_thicknesses": raw_thicknesses,
        "fractions": fractions,
        "thicknesses_m": thicknesses_m,
        "reflection_linear": reflection_linear,
        "lp_linear": objective_value,
        "fraction_history_steps": onp.asarray(fraction_history_steps, dtype=int),
        "fraction_history": onp.asarray(fraction_history),
        "fraction_thickness_history_m": onp.asarray(fraction_thickness_history_m),
        "worst_reflection_history_steps": onp.asarray(
            worst_reflection_history_steps, dtype=int
        ),
        "worst_reflection_history_db": onp.asarray(worst_reflection_history_db),
    }


def get_segment_indices(num_graded_layers, num_materials):
    """Assign consecutive graded layers to material-axis segments."""
    assert isinstance(num_graded_layers, int)
    assert isinstance(num_materials, int)
    assert num_graded_layers >= 1
    assert num_materials >= 1

    base_count = num_graded_layers // num_materials
    remainder = num_graded_layers % num_materials
    counts = [
        base_count + (1 if segment >= num_materials - remainder else 0)
        for segment in range(num_materials)
    ]
    return jnp.asarray(
        onp.concatenate(
            [
                onp.full(count, segment, dtype=int)
                for segment, count in enumerate(counts)
                if count > 0
            ]
        ),
        dtype=int,
    )


def contains_magnetic_material(material_sequence, magnetic_material_ids):
    """Return True if a material sequence contains at least one magnetic material."""
    magnetic_material_ids = {int(material_id) for material_id in magnetic_material_ids}
    return any(int(material_id) in magnetic_material_ids for material_id in material_sequence)


def initial_segment_fraction_raws(num_graded_layers, num_materials):
    """Return logits initialized inside each assigned material-axis segment."""
    assert isinstance(num_graded_layers, int)
    assert num_graded_layers >= 1
    if num_graded_layers == 1:
        return jnp.array([])

    segment_indices = onp.asarray(
        get_segment_indices(num_graded_layers, num_materials), dtype=int
    )
    local_fractions = []
    seen_by_segment = {segment: 0 for segment in range(num_materials)}
    counts_by_segment = {
        segment: int(onp.sum(segment_indices == segment))
        for segment in range(num_materials)
    }

    for segment in segment_indices[:-1]:
        segment = int(segment)
        seen_by_segment[segment] += 1
        local_fractions.append(
            seen_by_segment[segment] / (counts_by_segment[segment] + 1)
        )

    local_fractions = jnp.clip(jnp.asarray(local_fractions), 1e-6, 1.0 - 1e-6)
    return jnp.log(local_fractions / (1.0 - local_fractions))


def raw_parameters_to_multi_material_profile(
    raw_fraction_axis, raw_thicknesses, total_thickness_m, num_materials
):
    """
    Convert raw variables to constrained multi-material fraction-axis profile.

    Each layer is constrained to its assigned segment: segment_index + sigmoid(raw).
    The final layer is fixed to pure final material at fraction_axis = num_materials.
    """
    raw_fraction_axis = jnp.asarray(raw_fraction_axis)
    raw_thicknesses = jnp.asarray(raw_thicknesses)
    assert raw_fraction_axis.ndim == 1
    assert raw_thicknesses.ndim == 1
    assert raw_fraction_axis.shape[0] == raw_thicknesses.shape[0] - 1

    num_graded_layers = int(raw_thicknesses.shape[0])
    segment_indices = get_segment_indices(num_graded_layers, int(num_materials))
    local_fraction = jnp.concatenate(
        [
            jax.nn.sigmoid(raw_fraction_axis),
            jnp.array([1.0], dtype=raw_thicknesses.dtype),
        ]
    )
    fraction_axis = segment_indices.astype(raw_thicknesses.dtype) + local_fraction
    fraction_axis = fraction_axis.at[-1].set(float(num_materials))
    thicknesses_m = float(total_thickness_m) * jax.nn.softmax(raw_thicknesses)

    return fraction_axis, local_fraction, segment_indices, thicknesses_m


def _build_multi_material_graded_stack_eps_mu(
    material_sequence, frequencies, local_fraction, segment_indices
):
    material_sequence = tuple(int(material_index) for material_index in material_sequence)
    assert len(material_sequence) >= 1
    for material_index in material_sequence:
        assert material_index in onp.arange(1, 17)

    eps_materials, mu_materials = get_eps_mu_Michielssen(
        onp.asarray(material_sequence, dtype=int), frequencies
    )
    eps_air = jnp.ones_like(eps_materials[0])
    mu_air = jnp.ones_like(mu_materials[0])
    eps_pec = jnp.zeros_like(eps_air) + jnp.inf
    mu_pec = jnp.ones_like(mu_air)

    eps_endpoints = jnp.concatenate([eps_air[jnp.newaxis, :], eps_materials], axis=0)
    mu_endpoints = jnp.concatenate([mu_air[jnp.newaxis, :], mu_materials], axis=0)

    segment_indices = jnp.asarray(segment_indices, dtype=int)
    local_fraction = jnp.asarray(local_fraction)
    eps_left = eps_endpoints[segment_indices]
    eps_right = eps_endpoints[segment_indices + 1]
    mu_left = mu_endpoints[segment_indices]
    mu_right = mu_endpoints[segment_indices + 1]
    fill = local_fraction[:, jnp.newaxis]
    eps_graded = (1 - fill) * eps_left + fill * eps_right
    mu_graded = (1 - fill) * mu_left + fill * mu_right

    eps_stack = jnp.concatenate(
        [eps_air[:, jnp.newaxis], eps_graded.T, eps_pec[:, jnp.newaxis]], axis=1
    )
    mu_stack = jnp.concatenate(
        [mu_air[:, jnp.newaxis], mu_graded.T, mu_pec[:, jnp.newaxis]], axis=1
    )
    return eps_stack, mu_stack


def reflection_linear_multi_material_graded_stack(
    material_sequence,
    frequencies,
    total_thickness_m,
    local_fraction,
    segment_indices,
    thicknesses_m,
    inc_angle_deg,
    polarization="average",
):
    """Return linear reflection for Air | constrained multi-material graded layers | PEC."""
    eps_stack, mu_stack = _build_multi_material_graded_stack_eps_mu(
        material_sequence, frequencies, local_fraction, segment_indices
    )
    stack_thicknesses_m = jnp.concatenate(
        [jnp.array([0.0]), jnp.asarray(thicknesses_m), jnp.array([0.0])]
    )
    R_TE, R_TM = _stackrt_eps_mu_reflection_no_assert(
        eps_stack, mu_stack, stack_thicknesses_m, frequencies, inc_angle_deg
    )
    if polarization == "TE":
        reflection_linear = R_TE
    elif polarization == "TM":
        reflection_linear = R_TM
    elif polarization == "average":
        reflection_linear = (R_TE + R_TM) / 2.0
    elif polarization == "worst_case":
        reflection_linear = jnp.maximum(R_TE, R_TM)
    else:
        raise ValueError(
            "polarization must be 'average', 'TE', 'TM', or 'worst_case'."
        )
    return jnp.maximum(reflection_linear.squeeze(), 1e-300)


def lp_reflection_objective_multi_material(
    material_sequence,
    raw_fraction_axis,
    raw_thicknesses,
    frequencies,
    total_thickness_m,
    p_norm,
    inc_angle_deg,
    polarization="average",
):
    _, local_fraction, segment_indices, thicknesses_m = (
        raw_parameters_to_multi_material_profile(
            raw_fraction_axis,
            raw_thicknesses,
            total_thickness_m,
            len(tuple(material_sequence)),
        )
    )
    reflection_linear = reflection_linear_multi_material_graded_stack(
        material_sequence,
        frequencies,
        total_thickness_m,
        local_fraction,
        segment_indices,
        thicknesses_m,
        inc_angle_deg,
        polarization=polarization,
    )
    return jnp.mean(reflection_linear**p_norm) ** (1.0 / p_norm)


def optimize_multi_material_graded_stack(
    material_sequence,
    frequencies,
    total_thickness_m,
    num_graded_layers,
    p_norm,
    inc_angle_deg,
    num_steps,
    learning_rate,
    fraction_history_interval=200,
    worst_history_interval=50,
    return_history=True,
    initial_raw_fraction_axis=None,
    initial_raw_thicknesses=None,
    polarization="average",
):
    """Optimize a constrained material-axis graded stack for one material sequence."""
    material_sequence = tuple(int(material_index) for material_index in material_sequence)
    if initial_raw_fraction_axis is None:
        raw_fraction_axis = initial_segment_fraction_raws(
            num_graded_layers, len(material_sequence)
        )
    else:
        raw_fraction_axis = jnp.asarray(initial_raw_fraction_axis)
        assert raw_fraction_axis.ndim == 1
        assert raw_fraction_axis.shape[0] == num_graded_layers - 1

    if initial_raw_thicknesses is None:
        raw_thicknesses = jnp.zeros(num_graded_layers)
    else:
        raw_thicknesses = jnp.asarray(initial_raw_thicknesses)
        assert raw_thicknesses.ndim == 1
        assert raw_thicknesses.shape[0] == num_graded_layers

    def objective(raw_fraction_axis, raw_thicknesses):
        return lp_reflection_objective_multi_material(
            material_sequence,
            raw_fraction_axis,
            raw_thicknesses,
            frequencies,
            total_thickness_m,
            p_norm,
            inc_angle_deg,
            polarization=polarization,
        )

    value_and_grad = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
    fraction_history_steps = []
    fraction_axis_history = []
    local_fraction_history = []
    fraction_thickness_history_m = []
    worst_reflection_history_steps = []
    worst_reflection_history_db = []

    def current_profile(current_raw_fraction_axis, current_raw_thicknesses):
        return raw_parameters_to_multi_material_profile(
            current_raw_fraction_axis,
            current_raw_thicknesses,
            total_thickness_m,
            len(material_sequence),
        )

    def record_history(step, current_raw_fraction_axis, current_raw_thicknesses):
        fraction_axis, local_fraction, segment_indices, thicknesses_m = current_profile(
            current_raw_fraction_axis, current_raw_thicknesses
        )

        if (
            return_history
            and fraction_history_interval
            and (
                step == 0
                or step == int(num_steps)
                or step % int(fraction_history_interval) == 0
            )
        ):
            fraction_history_steps.append(int(step))
            fraction_axis_history.append(onp.asarray(fraction_axis))
            local_fraction_history.append(onp.asarray(local_fraction))
            fraction_thickness_history_m.append(onp.asarray(thicknesses_m))

        if (
            return_history
            and worst_history_interval
            and (
                step == 0
                or step == int(num_steps)
                or step % int(worst_history_interval) == 0
            )
        ):
            reflection_linear = reflection_linear_multi_material_graded_stack(
                material_sequence,
                frequencies,
                total_thickness_m,
                local_fraction,
                segment_indices,
                thicknesses_m,
                inc_angle_deg,
                polarization=polarization,
            )
            worst_reflection_db = 10 * jnp.log10(jnp.max(reflection_linear))
            worst_reflection_history_steps.append(int(step))
            worst_reflection_history_db.append(float(worst_reflection_db))

    record_history(0, raw_fraction_axis, raw_thicknesses)

    try:
        import optax

        optimizer = optax.adam(learning_rate)
        params = (raw_fraction_axis, raw_thicknesses)
        opt_state = optimizer.init(params)
        for step in range(1, int(num_steps) + 1):
            value, grads = value_and_grad(params[0], params[1])
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            record_history(step, params[0], params[1])
        raw_fraction_axis, raw_thicknesses = params
        objective_value = value
    except ImportError:
        objective_value = None
        for step in range(1, int(num_steps) + 1):
            objective_value, grads = value_and_grad(raw_fraction_axis, raw_thicknesses)
            raw_fraction_axis = raw_fraction_axis - learning_rate * grads[0]
            raw_thicknesses = raw_thicknesses - learning_rate * grads[1]
            record_history(step, raw_fraction_axis, raw_thicknesses)

    fraction_axis, local_fraction, segment_indices, thicknesses_m = current_profile(
        raw_fraction_axis, raw_thicknesses
    )
    reflection_linear = reflection_linear_multi_material_graded_stack(
        material_sequence,
        frequencies,
        total_thickness_m,
        local_fraction,
        segment_indices,
        thicknesses_m,
        inc_angle_deg,
        polarization=polarization,
    )
    objective_value = jnp.mean(reflection_linear**p_norm) ** (1.0 / p_norm)

    return {
        "material_sequence": material_sequence,
        "raw_fraction_axis": raw_fraction_axis,
        "raw_thicknesses": raw_thicknesses,
        "fraction_axis": fraction_axis,
        "local_fraction": local_fraction,
        "segment_indices": segment_indices,
        "thicknesses_m": thicknesses_m,
        "reflection_linear": reflection_linear,
        "lp_linear": objective_value,
        "fraction_history_steps": onp.asarray(fraction_history_steps, dtype=int),
        "fraction_axis_history": onp.asarray(fraction_axis_history),
        "local_fraction_history": onp.asarray(local_fraction_history),
        "fraction_thickness_history_m": onp.asarray(fraction_thickness_history_m),
        "worst_reflection_history_steps": onp.asarray(
            worst_reflection_history_steps, dtype=int
        ),
        "worst_reflection_history_db": onp.asarray(worst_reflection_history_db),
    }
