from jax import numpy as jnp

num_mixing_layers = 10

def get_ratios(method="linear"):
            thicknesses = jnp.linspace(0, 1, num_mixing_layers)

            if method == "linear":
                fun_ratio = lambda x: 1 - x
            elif method == "sine":
                fun_ratio = lambda x: 1 - jnp.sin(x * jnp.pi / 2)
            else:
                raise ValueError

            return fun_ratio(thicknesses)

def get_n_k_optimized():
            n_k_simulated_ = jnp.concatenate([jnp.ones((1, n_k_simulated.shape[1]), dtype=jnp.complex128), n_k_simulated], axis=0)

            n_k_graded = []
            for ind in range(0, len(thicknesses_intermediate) - 1):
                n_k_thickness = []

                for ratio in get_ratios():
                    n_k_thickness.append(n_k_simulated_[ind] * ratio + n_k_simulated_[ind + 1] * (1 - ratio))
                n_k_graded.append(n_k_thickness)

            n_k_optimized = jnp.concatenate([
                jnp.array(n_k_graded[ind] + [n_k_simulated_[ind + 1]]) for ind in range(0, len(thicknesses_intermediate) - 1)
            ] + [
                jnp.expand_dims(n_k_simulated_[-1], axis=0)
            ], axis=0)
            return n_k_optimized