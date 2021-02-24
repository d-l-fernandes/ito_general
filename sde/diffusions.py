import haiku as hk
import jax
import jax.numpy as jnp
from absl import flags

import aux
from jax_aux import aux_math
from sde import mappings

Array = jnp.ndarray

flags.DEFINE_enum("diffusion", "constant_full",
                  ["constant_diagonal", "constant_full", "nn_time", "nn_space", "nn_general"],
                  "Diffusion to use.")
FLAGS = flags.FLAGS


class BaseDiffusion(hk.Module):
    def __init__(self, output_size: int, name: str = "base"):
        super().__init__(f"diffusion_{name}")
        self.output_size = output_size

    def __call__(self, x: Array, t: float) -> Array:
        raise NotImplementedError


class ConstantDiagonal(BaseDiffusion):
    def __init__(self, output_size: int, name: str = "constant_diagonal"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        j = self.output_size
        diag = hk.get_parameter("diag", shape=[j], dtype=x.dtype, init=hk.initializers.RandomNormal())
        return aux_math.diag(jax.nn.softplus(diag))


class ConstantFull(BaseDiffusion):
    def __init__(self, output_size: int, name: str = "constant_full"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        j = self.output_size
        matrix = hk.get_parameter("matrix", shape=[j, j], dtype=x.dtype, init=hk.initializers.RandomNormal())
        return aux_math.matrix_diag_transform(matrix, jax.nn.softplus)


class NNTime(BaseDiffusion):
    def __init__(self, output_size: int, name: str = "nn_time"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        intermediate_dims = 3 * self.output_size**2
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(self.output_size**2, w_init=hk.initializers.Constant(1.), b_init=hk.initializers.Constant(1.))
        ])

        output = jnp.reshape(mlp(jnp.array([t])), (self.output_size, self.output_size))
        return aux_math.matrix_diag_transform(output, jax.nn.softplus)


class NNSpace(BaseDiffusion):
    def __init__(self, output_size: int, name: str = "nn_space"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        intermediate_dims = 3 * self.output_size**2
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(self.output_size**2, w_init=hk.initializers.Constant(1.), b_init=hk.initializers.Constant(1.))
        ])
        output = jnp.reshape(mlp(x), (self.output_size, self.output_size))
        return aux_math.matrix_diag_transform(output, jax.nn.softplus)


class NNGeneral(BaseDiffusion):
    def __init__(self, output_size: int, name: str = "nn_general"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        intermediate_dims = 3 * self.output_size
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(self.output_size, w_init=hk.initializers.Constant(1.), b_init=hk.initializers.Constant(1.))
        ])
        output = jnp.reshape(mlp(jnp.append(x, t)), (self.output_size, self.output_size))
        return aux_math.matrix_diag_transform(output, jax.nn.softplus)


diffusions_dict = {
    "constant_diagonal": ConstantDiagonal,
    "constant_full": ConstantFull,
    "nn_time": NNTime,
    "nn_space": NNSpace,
    "nn_general": NNGeneral
}


class DiffusionIto:
    def __init__(self, mapping: mappings.Map, diffusion_x: BaseDiffusion, likelihood: aux.Likelihood):
        self.mapping = mapping
        self.diffusion_x = diffusion_x
        self.likelihood = likelihood

    def __call__(self, x: Array, t: float) -> Array:
        first_derivative = self.mapping.first_derivative(x, t)
        diffusion_x = self.diffusion_x(x, t)
        diff_y = first_derivative(diffusion_x)
        diffusion_y = aux_math.diag_part(jnp.einsum("cd,ed->ce", diff_y, diff_y))

        wiener_term = \
            diffusion_y + \
            aux_math.diag_part(self.likelihood())**2

        return jnp.sqrt(wiener_term)
