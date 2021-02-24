import haiku as hk
import jax
import jax.numpy as jnp
from absl import flags

from jax_aux import aux_math
from sde import mappings, diffusions

Array = jnp.ndarray

flags.DEFINE_enum("drift", "linear_constant",
                  ["linear_constant", "linear_nn_time", "nn_space", "nn_general"],
                  "Drift to use.")
FLAGS = flags.FLAGS


class BaseDrift(hk.Module):
    def __init__(self, output_size: int, name: str = "base"):
        super().__init__(f"drift_{name}")
        self.output_size = output_size

    def __call__(self, x: Array, t: float) -> Array:
        raise NotImplementedError


class LinearConstant(BaseDrift):
    def __init__(self, output_size: int, name: str = "linear_constant"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        j = self.output_size
        matrix = hk.get_parameter("matrix", shape=[j, j], dtype=x.dtype, init=hk.initializers.Constant(0.))
        b = hk.get_parameter("bias", shape=[j], dtype=x.dtype, init=hk.initializers.Constant(0.))
        return jnp.einsum('ab,b->a', matrix, x) + b


class LinearNNTime(BaseDrift):
    def __init__(self, output_size: int, name: str = "linear_nn_time"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        intermediate_dims = 3 * self.output_size**2
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(self.output_size**2, w_init=hk.initializers.Constant(0.), b_init=hk.initializers.Constant(0.))
        ])
        return jnp.matmul(jnp.reshape(mlp(jnp.array([t])), (self.output_size, self.output_size)), x)


class NNSpace(BaseDrift):
    def __init__(self, output_size: int, name: str = "nn_space"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        intermediate_dims = 3 * self.output_size
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(self.output_size, w_init=hk.initializers.Constant(0.), b_init=hk.initializers.Constant(0.))
        ])
        return mlp(x)


class NNGeneral(BaseDrift):
    def __init__(self, output_size: int, name: str = "nn_space"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        intermediate_dims = 3 * self.output_size
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(self.output_size, w_init=hk.initializers.Constant(0.), b_init=hk.initializers.Constant(0.))
        ])
        return mlp(jnp.append(x, t))


drifts_dict = {
    "linear_constant": LinearConstant,
    "linear_nn_time": LinearNNTime,
    "nn_space": NNSpace,
    "nn_general": NNGeneral
}


class DriftIto:
    def __init__(self, mapping: mappings.Map, drift_x: BaseDrift, diffusion_x: diffusions.BaseDiffusion):
        self.mapping = mapping
        self.drift_x = drift_x
        self.diffusion_x = diffusion_x

    def __call__(self, x: Array, t: float) -> Array:
        first_derivative = self.mapping.first_derivative(x, t)
        hessian = self.mapping.hessian(x, t)
        time_derivative = self.mapping.time_derivative(x, t)

        drift_x = self.drift_x(x, t)
        diffusion_x = self.diffusion_x(x, t)

        hessian_term = 0.5 * jnp.sum(
            aux_math.diag_part(jnp.einsum("ce,bed->bcd", diffusion_x, hessian(jnp.transpose(diffusion_x)))),
            axis=-1)

        drift = time_derivative(jnp.array([1.])) + first_derivative(drift_x) + hessian_term

        return drift
