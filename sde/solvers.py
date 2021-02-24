from typing import Tuple

import haiku as hk
import jax.numpy as jnp
import numpyro
from absl import flags
from numpyro.distributions.continuous import MultivariateNormal

from sde import drifts, diffusions

Array = jnp.ndarray

flags.DEFINE_enum("solver", "strong_3_halfs", ["euler_maruyama", "strong_3_halfs"], "Solver to use.")
FLAGS = flags.FLAGS


class BaseSolver:
    def __init__(self, delta_t: float, beta_dims: int, drift: drifts.BaseDrift, diffusion: diffusions.BaseDiffusion):
        self.delta_t = delta_t
        self.beta_dims = beta_dims
        self.drift = drift
        self.diffusion = diffusion

    def __call__(self, x_0: Array, time: float) -> Array:
        raise NotImplementedError


class EulerMaruyamaSolver(BaseSolver):
    def __init__(self, delta_t: float, beta_dims: int, drift: drifts.BaseDrift, diffusion: diffusions.BaseDiffusion):
        super().__init__(delta_t, beta_dims, drift, diffusion)

    def __call__(self, x_0: Array, time: float) -> Array:
        rng_beta = hk.next_rng_key()
        delta_beta = numpyro.sample("delta_beta",
                                    MultivariateNormal(
                                        loc=jnp.zeros(self.beta_dims),
                                        scale_tril=jnp.sqrt(self.delta_t) * jnp.eye(self.beta_dims)),
                                    rng_key=rng_beta)
        drift = self.drift(x_0, time)
        diff = self.diffusion(x_0, time)
        x_1 = x_0 + drift * self.delta_t + jnp.matmul(diff, delta_beta)

        return x_1


class Strong3HalfsSolver(BaseSolver):
    def __init__(self, delta_t: float, beta_dims: int, drift: drifts.BaseDrift, diffusion: diffusions.BaseDiffusion):
        super().__init__(delta_t, beta_dims, drift, diffusion)

    def __call__(self, x_0: Array, time: float) -> Array:
        rng_beta = hk.next_rng_key()

        # Vector of zeros
        beta_mean_vector = jnp.zeros((self.beta_dims*2, ))

        # Covariance matrix for the betas and gammas
        beta_covariance_top_left = self.delta_t ** 3 / 3 * jnp.eye(self.beta_dims)
        beta_covariance_top_right = self.delta_t ** 2 / 2 * jnp.eye(self.beta_dims)
        beta_covariance_bottom_right = self.delta_t * jnp.eye(self.beta_dims)
        beta_covariance_top = jnp.concatenate((beta_covariance_top_left, beta_covariance_top_right), axis=1)
        beta_covariance_bottom = jnp.concatenate((beta_covariance_top_right, beta_covariance_bottom_right),
                                                 axis=1)
        beta_covariance = jnp.concatenate((beta_covariance_top, beta_covariance_bottom), axis=0)

        delta_gamma_beta = numpyro.sample("delta_gamma_beta",
                                          MultivariateNormal(loc=beta_mean_vector,
                                                             covariance_matrix=beta_covariance),
                                          rng_key=rng_beta)

        delta_gamma = delta_gamma_beta[:self.beta_dims]
        delta_beta = delta_gamma_beta[self.beta_dims:]

        drift_0 = self.drift(x_0, time)
        diff = self.diffusion(x_0, time)
        diff_plus = self.diffusion(x_0, time + self.delta_t)

        init_x_1 = x_0 + drift_0 * self.delta_t + jnp.matmul(diff, delta_beta)
        init_x_1 += 1. / self.delta_t * jnp.matmul(diff_plus - diff, delta_beta * self.delta_t - delta_gamma)

        def scan_fn(carry, s):
            x_1 = carry
            x_0_plus = \
                x_0 + drift_0 * self.delta_t / self.beta_dims + \
                diff[:, s] * jnp.sqrt(self.delta_t)
            x_0_minus = \
                x_0 + drift_0 * self.delta_t / self.beta_dims - \
                diff[:, s] * jnp.sqrt(self.delta_t)

            drift_0_plus = self.drift(x_0_plus, time + self.delta_t)
            drift_0_minus = self.drift(x_0_minus, time + self.delta_t)

            x_1 += 0.25 * self.delta_t * (drift_0_plus + drift_0_minus)
            x_1 -= 0.5 * drift_0 * self.delta_t
            x_1 += \
                1. / (2 * jnp.sqrt(self.delta_t)) * (drift_0_plus - drift_0_minus) * delta_gamma[s]
            return x_1, None

        final_x_1, _ = hk.scan(scan_fn, init_x_1, jnp.arange(self.beta_dims))

        return final_x_1


solvers_dict = {
    "euler_maruyama": EulerMaruyamaSolver,
    "strong_3_halfs": Strong3HalfsSolver
}
