import haiku as hk
import jax
import jax.numpy as jnp
from jax_aux import aux_math

Array = jnp.ndarray


class InitialLatent(hk.Module):
    def __init__(self, output_size: int, name: str = "initial_latents"):
        super().__init__(name)
        self.output_size = output_size

    def __call__(self) -> Array:
        initial_values = hk.get_parameter("initial_values", [self.output_size], init=hk.initializers.Constant(0.))
        return initial_values


class Likelihood(hk.Module):
    def __init__(self, output_size: int, name: str = "likelihood"):
        super().__init__(name)
        self.output_size = output_size

    def __call__(self) -> Array:
        likelihood = hk.get_parameter("likelihood", [self.output_size], init=hk.initializers.RandomNormal(mean=-5))
        return aux_math.diag(jax.nn.softplus(likelihood))
