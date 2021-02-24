from typing import Callable, Union

import haiku as hk
import jax
import jax.numpy as jnp
from absl import flags

Array = jnp.ndarray

flags.DEFINE_enum("mapping", "linear_combination",
                  ["identity", "linear_combination", "linear_combination_with_time", "nn", "nn_with_time",
                   "neural_ode", "neural_ode_with_time"],
                  "Mapping to use.")
FLAGS = flags.FLAGS


class BaseMap(hk.Module):
    def __init__(self, output_size: int, name: str = "base"):
        super().__init__(f"mapping_{name}")
        self.output_size = output_size
        self.derivative_map = False

    def __call__(self, x: Array, t: float) -> Array:
        raise NotImplementedError

    def time_derivative(self, x: Array, t: float) -> Callable[[Array], Array]:
        def aux(v: Array):
            return jax.jvp(lambda t_int: self.__call__(x, t_int), (jnp.array([t]),), (v, ))[1]
        return aux

    def first_derivative(self, x: Array, t: float) -> Callable[[Array], Array]:
        def aux(v: Array):
            if len(v.shape) == 1:
                return jax.jvp(lambda x_int: self.__call__(x_int, t), (x,), (v,))[1]
            elif len(v.shape) == 2:
                return jax.vmap(lambda vs: jax.jvp(lambda x_int: self.__call__(x_int, t),
                                                   (x,), (vs,))[1], 1, 1)(v)
            else:
                raise RuntimeError("first derivative vector product works with vector with at most two dimensions.")
        return aux

    def hessian(self, x: Array, t: float) -> Callable[[Array], Array]:
        def aux(v: Array):
            if len(v.shape) == 1:
                return jax.jvp(jax.jacfwd(lambda x_int: self.__call__(x_int, t), 0), (x,), (v,))[1]
            elif len(v.shape) == 2:
                return jax.vmap(lambda vs: jax.jvp(jax.jacfwd(lambda x_int: self.__call__(x_int, t), 0),
                                                   (x,), (vs,))[1], 1, 1)(v)
            else:
                raise RuntimeError("hessian vector product works with vector with at most two dimensions.")
        return aux


class Identity(BaseMap):
    def __init__(self, output_size: int, name: str = "identity"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        if x.shape[-1] != self.output_size:
            raise RuntimeError("For identity mapping, input dims must be equal to output dims")
        return x


class LinearCombination(BaseMap):
    def __init__(self, output_size: int, name: str = "linear_combination"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        j, i = self.output_size, x.shape[-1]
        matrix_a = hk.get_parameter("matrix_a", shape=[j, i], dtype=x.dtype, init=hk.initializers.RandomNormal())
        b = hk.get_parameter("b", shape=[j], dtype=x.dtype, init=hk.initializers.RandomNormal())
        return jnp.einsum('ab,b->a', matrix_a, x) + b


class LinearCombinationWithTime(BaseMap):
    def __init__(self, output_size: int, name: str = "linear_combination_with_time"):
        super().__init__(output_size, name)

    def __call__(self, x: Array, t: float) -> Array:
        j, i = self.output_size, x.shape[-1]
        matrix_a = hk.get_parameter("matrix_a", shape=[j, i+1], dtype=x.dtype, init=hk.initializers.RandomNormal())
        b = hk.get_parameter("b", shape=[j], dtype=x.dtype, init=hk.initializers.RandomNormal())
        return jnp.einsum('ab,b->a', matrix_a, jnp.append(x, t)) + b


class NN(BaseMap):
    def __init__(self, output_size: int, name: str = "nn"):
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


class NNWithTime(BaseMap):
    def __init__(self, output_size: int, name: str = "nn_with_time"):
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


class BaseDerivativeMap(hk.Module):
    def __init__(self, output_size: int, name: str = "base_derivative"):
        super().__init__(f"mapping_{name}")
        self.output_size = output_size
        self.derivative_map = True

    def _time_derivative(self, x: Array, t: float) -> Array:
        raise NotImplementedError

    def _first_derivative(self, x: Array, t: float) -> Array:
        raise NotImplementedError

    def time_derivative(self, x: Array, t: float) -> Callable[[Array], Array]:
        return lambda v: jnp.matmul(self._time_derivative(x, t), v)

    def first_derivative(self, x: Array, t: float) -> Callable[[Array], Array]:
        return lambda v: jnp.matmul(self._first_derivative(x, t), v)

    def hessian(self, x: Array, t: float) -> Callable[[Array], Array]:
        def aux(v: Array):
            if len(v.shape) == 1:
                return jax.jvp(lambda x_int: self._first_derivative(x_int, t), (x,), (v,))[1]
            elif len(v.shape) == 2:
                return jax.vmap(lambda vs: jax.jvp(lambda x_int: self._first_derivative(x_int, t),
                                                   (x,), (vs,))[1], 1, 1)(v)
            else:
                raise RuntimeError("hessian vector product works with vector with at most two dimensions.")
        return aux


class NeuralODE(BaseDerivativeMap):
    def __init__(self, output_size: int, name: str = "neural_ode"):
        super().__init__(output_size, name)

    def _time_derivative(self, x: Array, t: float) -> Array:
        return jnp.zeros((self.output_size, 1))

    def _first_derivative(self, x: Array, t: float) -> Array:
        intermediate_dims = 3 * self.output_size * x.shape[-1]
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(self.output_size * x.shape[-1],
                      w_init=hk.initializers.Constant(0.), b_init=hk.initializers.Constant(0.))
        ])
        return mlp(x).reshape(self.output_size, x.shape[-1])


class NeuralODEWithTime(BaseDerivativeMap):
    def __init__(self, output_size: int, name: str = "neural_ode_with_time"):
        super().__init__(output_size, name)

    def _time_derivative(self, x: Array, t: float) -> Array:
        intermediate_dims = 3 * self.output_size
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(self.output_size, w_init=hk.initializers.Constant(0.), b_init=hk.initializers.Constant(0.))
        ])
        return mlp(jnp.append(x, t)).reshape(self.output_size, 1)

    def _first_derivative(self, x: Array, t: float) -> Array:
        intermediate_dims = 3 * self.output_size * x.shape[-1]
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(intermediate_dims), jax.nn.sigmoid,
            hk.Linear(self.output_size * x.shape[-1],
                      w_init=hk.initializers.Constant(0.), b_init=hk.initializers.Constant(0.))
        ])
        return mlp(jnp.append(x, t)).reshape(self.output_size, x.shape[-1])


Map = Union[BaseMap, BaseDerivativeMap]

mappings_dict = {
    "identity": Identity,
    "linear_combination": LinearCombination,
    "linear_combination_with_time": LinearCombinationWithTime,
    "nn": NN,
    "nn_with_time": NNWithTime,
    "neural_ode": NeuralODE,
    "neural_ode_with_time": NeuralODEWithTime
}
