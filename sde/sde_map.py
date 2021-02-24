from typing import Tuple

import haiku as hk
import jax.numpy as jnp
import numpyro
from numpyro.distributions.continuous import MultivariateNormal

import aux
from jax_aux import aux_math
from sde import mappings, solvers

Array = jnp.ndarray


class SDEMap:
    def __init__(self, delta_t: float, solver_x: solvers.BaseSolver, mapping: mappings.Map, likelihood: aux.Likelihood):
        self.delta_t = delta_t
        self.solver_x = solver_x
        self.mapping = mapping
        self.likelihood = likelihood

    def brownian_noise(self) -> Array:
        key = hk.next_rng_key()

        delta_beta = numpyro.sample("delta_beta",
                                    MultivariateNormal(
                                        loc=jnp.zeros(self.likelihood().shape[-1]),
                                        covariance_matrix=self.delta_t * jnp.eye(self.likelihood().shape[-1])),
                                    rng_key=key)
        return aux_math.diag_part(self.likelihood()) * delta_beta

    def __call__(self,
                 x_0: Array,
                 y_values: Array,
                 t_mask: Array,
                 training: int,
                 t_0: float = 0.) -> Tuple[Array, Array, Array, Array]:

        def scan_fn(carry, it):
            x_t, y_t_path, y_t_generated, t = carry
            y_t_true, mask = it

            x_t_new = self.solver_x(x_t, t)
            d_x = x_t_new - x_t

            first_derivative = self.mapping.first_derivative(x_t, t)

            d_y = \
                self.mapping.time_derivative(x_t, t)(jnp.array([1.])) * self.delta_t + \
                first_derivative(d_x) + \
                0.5 * jnp.einsum("bc,c->b", self.mapping.hessian(x_t, t)(d_x), d_x)

            y_to_use = \
                (mask * y_t_true + jnp.abs(mask - 1) * y_t_path) * training + \
                jnp.abs(training - 1) * y_t_generated

            noise = self.brownian_noise()
            y_t_new = y_to_use + d_y + jnp.abs(training - 1) * noise
            y_t_generated_new = y_t_generated + d_y + noise

            t = t + self.delta_t
            return (x_t_new, y_t_new, y_t_generated_new, t), \
                   (x_t_new, y_t_new, y_t_generated_new, t)

        _, (final_paths_x, final_paths_y, final_paths_y_generated, final_t_seq) = \
            hk.scan(scan_fn, (x_0, y_values[0], y_values[0], t_0), (y_values[:-1], t_mask[:-1]))

        t_seq = jnp.tile(jnp.array([t_0]), [y_values.shape[0]])
        t_seq = t_seq.at[1:].set(final_t_seq)
        paths_x = jnp.tile(jnp.expand_dims(x_0, axis=0), [y_values.shape[0], 1])
        paths_x = paths_x.at[1:].set(final_paths_x)
        paths_y = jnp.tile(jnp.expand_dims(y_values[0], axis=0), [y_values.shape[0], 1])
        paths_y_generated = jnp.tile(jnp.expand_dims(y_values[0], axis=0), [y_values.shape[0], 1])
        paths_y = paths_y.at[1:].set(final_paths_y)
        paths_y_generated = paths_y_generated.at[1:].set(final_paths_y_generated)

        return t_seq, paths_x, paths_y, paths_y_generated
