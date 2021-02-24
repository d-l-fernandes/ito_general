from typing import Tuple, Optional, Sequence

import jax.numpy as jnp
from absl import flags
from numpy import random
import numpy as np

Array = jnp.ndarray
Batch = Tuple[Array, Array, Optional[Array], Optional[Array]]

flags.DEFINE_enum("dataset", "line",
                  ["line", "noisy_line", "quadratic", "sde_1d_non_linear", "sde_2d_linear_extrapolation",
                   "sde_2d_linear_interpolation", "sde_2d_to_5d_non_linear_mapping", "exchange_rate_extrapolation",
                   "exchange_rate_interpolation"],
                  "Dataset to use.")


class BaseDataGenerator:
    def __init__(self):
        # Data properties
        self.n_train: int = 0
        self.n_test: int = 0
        self.observed_dims: int = 0
        self.t_steps: int = 0
        self.t_steps_test: int = 0
        self.delta_t: float = 0.01
        self.known_drift_diffusion: bool = False

        # Plotting properties
        self.draw_y_axis: bool = False
        self.y_lims: Optional[Sequence[float]] = None

        # Train arrays
        self.ys_train: Array = jnp.ones((self.n_train, self.t_steps, self.observed_dims))
        self.masks_train: Array = jnp.ones((self.n_train, self.t_steps_test, self.observed_dims))
        self.drifts_train: Optional[Array] = None
        self.diffusions_train: Optional[Array] = None
        # Test arrays
        self.ys_test: Array = jnp.ones((self.n_train, self.t_steps, self.observed_dims))
        self.masks_test: Array = jnp.ones((self.n_train, self.t_steps_test, self.observed_dims))
        self.drifts_test: Optional[Array] = None
        self.diffusions_test: Optional[Array] = None

    def single_sample(self) -> Batch:
        if self.known_drift_diffusion:
            return self.ys_train[0], self.masks_train[0], self.drifts_train[0], self.diffusions_train[0]
        else:
            return self.ys_train[0], self.masks_train[0], self.drifts_train, self.diffusions_train

    def train_generator(self, batch_size: int) -> Batch:
        while True:
            idx = random.choice(self.n_train, batch_size)
            if self.known_drift_diffusion:
                yield self.ys_train[idx], self.masks_train[idx], self.drifts_train[idx], self.diffusions_train[idx]
            else:
                yield self.ys_train[idx], self.masks_train[idx], self.drifts_train, self.diffusions_train

    def test_generator(self, batch_size: int) -> Batch:
        if batch_size > self.n_test:
            raise ValueError("Batch size can't be bigger than number of test points.")

        n_iter_test = self.n_test // batch_size
        for b in range(n_iter_test):
            if self.known_drift_diffusion:
                yield \
                    self.ys_test[b * batch_size:(b + 1) * batch_size], \
                    self.masks_test[b * batch_size:(b + 1) * batch_size], \
                    self.drifts_test[b * batch_size:(b + 1) * batch_size], \
                    self.diffusions_test[b * batch_size:(b + 1) * batch_size]
            else:
                yield \
                    self.ys_test[b * batch_size:(b + 1) * batch_size], \
                    self.masks_test[b * batch_size:(b + 1) * batch_size], \
                    self.drifts_test, \
                    self.diffusions_test


class Line(BaseDataGenerator):
    def __init__(self):
        super().__init__()
        # Data properties
        self.n_train: int = 5
        self.n_test: int = 100
        self.observed_dims: int = 1
        self.t_steps: int = 200
        self.t_steps_test: int = 200
        self.delta_t: float = 0.005
        self.known_drift_diffusion: bool = True

        ts_train = jnp.linspace(0, self.delta_t * self.t_steps, self.t_steps)
        ts_test = jnp.linspace(0, self.delta_t * self.t_steps_test, self.t_steps_test)
        ys_train = 5.6 * ts_train + 2
        ys_test = 5.6 * ts_test + 2

        self.ys_train = jnp.tile(jnp.expand_dims(jnp.expand_dims(ys_train, 0), -1), (self.n_train, 1, 1))
        self.ys_test = jnp.tile(jnp.expand_dims(jnp.expand_dims(ys_test, 0), -1), (self.n_test, 1, 1))

        self.masks_train = jnp.ones((self.n_train, self.t_steps, self.observed_dims), dtype=jnp.int32)
        self.masks_test = jnp.ones((self.n_test, self.t_steps_test, self.observed_dims), dtype=jnp.int32)

        self.drifts_train = jnp.ones([self.n_train, self.t_steps, self.observed_dims]) * 5.6
        self.drifts_test = jnp.ones([self.n_test, self.t_steps_test, self.observed_dims]) * 5.6
        self.diffusions_train = jnp.zeros([self.n_train, self.t_steps, self.observed_dims])
        self.diffusions_test = jnp.zeros([self.n_test, self.t_steps_test, self.observed_dims])


class NoisyLine(BaseDataGenerator):
    def __init__(self):
        super().__init__()
        # Data properties
        self.n_train: int = 15
        self.n_test: int = 100
        self.observed_dims: int = 1
        self.t_steps: int = 200
        self.t_steps_test: int = 200
        self.delta_t: float = 0.005
        self.known_drift_diffusion: bool = True

        np.random.seed(42)

        x_train = np.zeros(shape=(self.t_steps, self.n_train, 1)) + 2
        x_test = np.zeros(shape=(self.t_steps, self.n_test, 1)) + 2

        for i in range(0, self.t_steps - 1):
            x_train[i + 1] = \
                x_train[i] + self.drift(x_train[i]) * self.delta_t + \
                np.sqrt(self.delta_t) * self.diffusion(x_train[i]) * np.random.normal(size=(self.n_train, 1))
        for i in range(0, self.t_steps - 1):
            x_test[i + 1] = \
                x_test[i] + self.drift(x_test[i]) * self.delta_t + \
                np.sqrt(self.delta_t) * self.diffusion(x_test[i]) * np.random.normal(size=(self.n_test, 1))

        self.ys_train = jnp.transpose(jnp.array(x_train), (1, 0, 2))
        self.ys_test = jnp.transpose(jnp.array(x_test), (1, 0, 2))

        self.masks_train = jnp.ones((self.n_train, self.t_steps, self.observed_dims), dtype=jnp.int32)
        self.masks_test = jnp.ones((self.n_test, self.t_steps_test, self.observed_dims), dtype=jnp.int32)

        self.drifts_train = jnp.ones([self.n_train, self.t_steps, self.observed_dims]) * 5.6
        self.drifts_test = jnp.ones([self.n_test, self.t_steps_test, self.observed_dims]) * 5.6
        self.diffusions_train = jnp.ones([self.n_train, self.t_steps, self.observed_dims])
        self.diffusions_test = jnp.ones([self.n_test, self.t_steps_test, self.observed_dims])

    @staticmethod
    def drift(state):
        return 5.6

    @staticmethod
    def diffusion(state):
        return 1.


class Quadratic(BaseDataGenerator):
    def __init__(self):
        super().__init__()
        # Data properties
        self.n_train: int = 15
        self.n_test: int = 100
        self.observed_dims: int = 1
        self.t_steps: int = 200
        self.t_steps_test: int = 200
        self.delta_t: float = 0.005
        self.known_drift_diffusion: bool = False

        ts_train = jnp.linspace(0, self.delta_t * self.t_steps, self.t_steps)
        ts_test = jnp.linspace(0, self.delta_t * self.t_steps_test, self.t_steps_test)
        ys_train = 3.1 * (ts_train - 0.5) ** 2 + 2
        ys_test = 3.1 * (ts_test - 0.5) ** 2 + 2

        self.ys_train = jnp.tile(jnp.expand_dims(jnp.expand_dims(ys_train, 0), -1), (self.n_train, 1, 1))
        self.ys_test = jnp.tile(jnp.expand_dims(jnp.expand_dims(ys_test, 0), -1), (self.n_test, 1, 1))

        self.masks_train = jnp.ones((self.n_train, self.t_steps, self.observed_dims), dtype=jnp.int32)
        self.masks_test = jnp.ones((self.n_test, self.t_steps_test, self.observed_dims), dtype=jnp.int32)


# Paper datasets
class SDE1DNonlinear(BaseDataGenerator):
    def __init__(self):
        super().__init__()
        # Data properties
        self.n_train: int = 1
        self.n_test: int = 1000
        self.observed_dims: int = 1
        self.t_steps: int = 800
        self.t_steps_test: int = 800
        self.delta_t: float = 0.01
        self.known_drift_diffusion: bool = True
        self.y_lims: Optional[Sequence[Sequence[float]]] = [[-2, 2]]

        np.random.seed(0)
        scale = 500

        x_train = np.zeros(shape=(self.t_steps*scale, 1))
        x_test = np.zeros(shape=(self.t_steps_test*scale, 1))

        drift_train = np.zeros(shape=(self.t_steps*scale, 1))
        drift_test = np.zeros(shape=(self.t_steps_test*scale, 1))

        diffusion_train = np.zeros(shape=(self.t_steps*scale, 1))
        diffusion_test = np.zeros(shape=(self.t_steps_test*scale, 1))

        for i in range(0, self.t_steps*scale-1):
            drift_train[i] = self.drift(x_train[i])
            drift_test[i] = self.drift(x_test[i])
            diffusion_train[i] = self.diffusion(x_train[i])
            diffusion_test[i] = self.diffusion(x_test[i])

            noise = np.random.normal(size=(1,))
            x_train[i+1] = \
                x_train[i] + drift_train[i] * self.delta_t/scale + \
                np.sqrt(self.delta_t/scale) * diffusion_train[i] * noise
            x_test[i+1] = \
                x_test[i] + drift_test[i] * self.delta_t/scale + \
                np.sqrt(self.delta_t/scale) * diffusion_test[i] * noise

        x_train = np.tile(x_train[None, ::scale], (self.n_train, 1, 1))
        x_test = np.tile(x_test[None, ::scale], (self.n_test, 1, 1))

        drift_train = np.tile(drift_train[None, ::scale], (self.n_train, 1, 1))
        drift_test = np.tile(drift_test[None, ::scale], (self.n_test, 1, 1))

        diffusion_train = np.tile(diffusion_train[None, ::scale], (self.n_train, 1, 1))
        diffusion_test = np.tile(diffusion_test[None, ::scale], (self.n_test, 1, 1))

        self.ys_train = jnp.array(x_train)
        self.ys_test = jnp.array(x_test)

        self.drifts_train = jnp.array(drift_train)
        self.drifts_test = jnp.array(drift_test)

        self.diffusions_train = jnp.array(diffusion_train)
        self.diffusions_test = jnp.array(diffusion_test)

        self.masks_train = jnp.ones((self.n_train, self.t_steps, self.observed_dims), dtype=jnp.int32)
        self.masks_test = jnp.ones((self.n_test, self.t_steps_test, self.observed_dims), dtype=jnp.int32)

    @staticmethod
    def drift(state):
        return 4 * (state - state ** 3)

    @staticmethod
    def diffusion(state):
        return 1.


class SDE2DLinearExtrapolation(BaseDataGenerator):
    def __init__(self):
        super().__init__()
        # Data properties
        self.n_train: int = 1
        self.n_test: int = 1000
        self.observed_dims: int = 2
        self.t_steps: int = 500
        self.t_steps_test: int = 800
        self.delta_t: float = 0.01
        self.known_drift_diffusion: bool = True

        np.random.seed(42)

        scale = 500
        x_train = np.zeros(shape=(self.t_steps_test*scale, 2))
        x_test = np.zeros(shape=(self.t_steps_test*scale, 2))

        drift_train = np.zeros(shape=(self.t_steps_test*scale, 2))
        drift_test = np.zeros(shape=(self.t_steps_test*scale, 2))

        diffusion_train = np.zeros(shape=(self.t_steps_test*scale, 2))
        diffusion_test = np.zeros(shape=(self.t_steps_test*scale, 2))

        x_train[0, 1] = 1
        x_test[0, 1] = 1

        for i in range(0, self.t_steps_test*scale-1):
            drift_train[i] = self.drift(x_train[i])
            drift_test[i] = self.drift(x_test[i])
            diffusion_train[i] = self.diffusion(x_train[i])
            diffusion_test[i] = self.diffusion(x_test[i])

            noise = np.random.normal(size=(2,))
            x_train[i+1] = \
                x_train[i] + drift_train[i] * self.delta_t/scale + \
                np.sqrt(self.delta_t/scale) * diffusion_train[i] * noise
            x_test[i+1] = \
                x_test[i] + drift_test[i] * self.delta_t/scale + \
                np.sqrt(self.delta_t/scale) * diffusion_test[i] * noise

        x_train = np.tile(x_train[None, :self.t_steps*scale:scale], (self.n_train, 1, 1))
        x_test = np.tile(x_test[None, ::scale], (self.n_test, 1, 1))

        drift_train = np.tile(drift_train[None, :self.t_steps*scale:scale], (self.n_train, 1, 1))
        drift_test = np.tile(drift_test[None, ::scale], (self.n_test, 1, 1))

        diffusion_train = np.tile(diffusion_train[None, :self.t_steps*scale:scale], (self.n_train, 1, 1))
        diffusion_test = np.tile(diffusion_test[None, ::scale], (self.n_test, 1, 1))

        self.ys_train = jnp.array(x_train)
        self.ys_test = jnp.array(x_test)

        self.drifts_train = jnp.array(drift_train)
        self.drifts_test = jnp.array(drift_test)

        self.diffusions_train = jnp.array(diffusion_train)
        self.diffusions_test = jnp.array(diffusion_test)

        self.masks_train = jnp.ones((self.n_train, self.t_steps, self.observed_dims), dtype=jnp.int32)
        self.masks_test = np.ones((self.n_test, self.t_steps_test, self.observed_dims), dtype=np.int32)
        self.masks_test[:, self.t_steps:] = 0
        self.masks_test = jnp.array(self.masks_test)

    @staticmethod
    def drift(state):
        d = np.zeros_like(state)
        d[0] = 4 * state[1]
        d[1] = -16 * state[0]
        return d

    @staticmethod
    def diffusion(state):
        return np.array([1., 1.])


class SDE2DLinearInterpolation(BaseDataGenerator):
    def __init__(self):
        super().__init__()
        # Data properties
        self.n_train: int = 1
        self.n_test: int = 1000
        self.observed_dims: int = 2
        self.t_steps: int = 500
        self.t_steps_test: int = 500
        self.delta_t: float = 0.01
        self.known_drift_diffusion: bool = True

        np.random.seed(42)

        scale = 500
        x_train = np.zeros(shape=(self.t_steps_test*scale, 2))
        x_test = np.zeros(shape=(self.t_steps_test*scale, 2))

        drift_train = np.zeros(shape=(self.t_steps_test*scale, 2))
        drift_test = np.zeros(shape=(self.t_steps_test*scale, 2))

        diffusion_train = np.zeros(shape=(self.t_steps_test*scale, 2))
        diffusion_test = np.zeros(shape=(self.t_steps_test*scale, 2))

        x_train[0, 1] = 1
        x_test[0, 1] = 1

        for i in range(0, self.t_steps_test*scale-1):
            drift_train[i] = self.drift(x_train[i])
            drift_test[i] = self.drift(x_test[i])
            diffusion_train[i] = self.diffusion(x_train[i])
            diffusion_test[i] = self.diffusion(x_test[i])

            noise = np.random.normal(size=(2,))
            x_train[i+1] = \
                x_train[i] + drift_train[i] * self.delta_t/scale + \
                np.sqrt(self.delta_t/scale) * diffusion_train[i] * noise
            x_test[i+1] = \
                x_test[i] + drift_test[i] * self.delta_t/scale + \
                np.sqrt(self.delta_t/scale) * diffusion_test[i] * noise

        x_train = np.tile(x_train[None, :self.t_steps*scale:scale], (self.n_train, 1, 1))
        x_test = np.tile(x_test[None, ::scale], (self.n_test, 1, 1))

        drift_train = np.tile(drift_train[None, :self.t_steps*scale:scale], (self.n_train, 1, 1))
        drift_test = np.tile(drift_test[None, ::scale], (self.n_test, 1, 1))

        diffusion_train = np.tile(diffusion_train[None, :self.t_steps*scale:scale], (self.n_train, 1, 1))
        diffusion_test = np.tile(diffusion_test[None, ::scale], (self.n_test, 1, 1))

        self.ys_train = jnp.array(x_train)
        self.ys_test = jnp.array(x_test)

        self.drifts_train = jnp.array(drift_train)
        self.drifts_test = jnp.array(drift_test)

        self.diffusions_train = jnp.array(diffusion_train)
        self.diffusions_test = jnp.array(diffusion_test)

        self.masks_train = jnp.ones((self.n_train, self.t_steps, self.observed_dims), dtype=jnp.int32)
        self.masks_test = np.ones((self.n_test, self.t_steps_test, self.observed_dims), dtype=np.int32)
        self.masks_test[:, self.t_steps:] = 0
        self.masks_test = jnp.array(self.masks_test)
        self.masks_train = np.ones((self.n_train, self.t_steps, self.observed_dims), dtype=np.int32)
        self.masks_train[:, self.t_steps // 5: 2 * self.t_steps // 5, 0] = 0
        self.masks_train[:, 3 * self.t_steps // 5: 4 * self.t_steps // 5, 1] = 0

        self.masks_test = np.ones((self.n_test, self.t_steps_test, self.observed_dims), dtype=np.int32)
        self.masks_test[:, self.t_steps_test // 5: 2 * self.t_steps_test // 5, 0] = 0
        self.masks_test[:, 3 * self.t_steps_test // 5: 4 * self.t_steps_test // 5, 1] = 0

        self.masks_train = jnp.array(self.masks_train)
        self.masks_test = jnp.array(self.masks_test)

    @staticmethod
    def drift(state):
        d = np.zeros_like(state)
        d[0] = 4 * state[1]
        d[1] = -16 * state[0]
        return d

    @staticmethod
    def diffusion(state):
        return np.array([1., 1.])


class SDE2DTo5DNonLinearMapping(BaseDataGenerator):
    def __init__(self):
        super().__init__()
        # Data properties
        self.n_train: int = 1
        self.n_test: int = 1000
        self.observed_dims: int = 5
        self.t_steps: int = 300
        self.t_steps_test: int = 300
        self.delta_t: float = 0.01
        self.known_drift_diffusion: bool = True

        scale = 500
        np.random.seed(42)

        x_train = np.zeros(shape=(self.t_steps*scale, 2))
        x_test = np.zeros(shape=(self.t_steps_test*scale, 2))
        y_train = np.zeros(shape=(self.t_steps*scale, 5))
        y_test = np.zeros(shape=(self.t_steps_test*scale, 5))

        drift_train = np.zeros(shape=(self.t_steps*scale, 5))
        drift_test = np.zeros(shape=(self.t_steps_test*scale, 5))

        diffusion_train = np.zeros(shape=(self.t_steps*scale, 5))
        diffusion_test = np.zeros(shape=(self.t_steps_test*scale, 5))

        x_train[0, 1] = 1
        x_test[0, 1] = 1

        y_train[0] = self.drift_y(x_train[0])
        y_test[0] = self.drift_y(x_test[0])

        for i in range(0, self.t_steps*scale-1):
            noise = np.random.normal(size=(2, ))
            noise_y = np.random.normal(size=(5, ))

            drift_train[i] = self.drift_y(x_train[i])
            drift_test[i] = self.drift_y(x_test[i])
            diffusion_train[i] = self.diffusion_y(x_train[i])
            diffusion_test[i] = self.diffusion_y(x_test[i])

            x_train[i+1] = \
                x_train[i] + self.drift(x_train[i]) * self.delta_t/scale + \
                np.sqrt(self.delta_t/scale) * self.diffusion(x_train[i]) * noise
            y_train[i+1] = \
                y_train[i] + self.drift_y(x_train[i]) * self.delta_t / scale + \
                np.sqrt(self.delta_t/scale) * self.diffusion_y(x_train[i]) * noise_y
            x_test[i+1] = \
                x_test[i] + self.drift(x_test[i]) * self.delta_t/scale + \
                np.sqrt(self.delta_t/scale) * self.diffusion(x_test[i]) * noise
            y_test[i+1] = \
                y_test[i] + self.drift_y(x_test[i]) * self.delta_t / scale + \
                np.sqrt(self.delta_t/scale) * self.diffusion_y(x_test[i]) * noise_y

        y_train = np.tile(y_train[None, ::scale], (self.n_train, 1, 1))
        y_test = np.tile(y_test[None, ::scale], (self.n_test, 1, 1))

        drift_train = np.tile(drift_train[None, ::scale], (self.n_train, 1, 1))
        drift_test = np.tile(drift_test[None, ::scale], (self.n_test, 1, 1))

        diffusion_train = np.tile(diffusion_train[None, ::scale], (self.n_train, 1, 1))
        diffusion_test = np.tile(diffusion_test[None, ::scale], (self.n_test, 1, 1))

        self.ys_train = jnp.array(y_train)
        self.ys_test = jnp.array(y_test)

        self.drifts_train = jnp.array(drift_train)
        self.drifts_test = jnp.array(drift_test)

        self.diffusions_train = jnp.array(diffusion_train)
        self.diffusions_test = jnp.array(diffusion_test)

        self.masks_train = jnp.ones((self.n_train, self.t_steps, self.observed_dims), dtype=jnp.int32)
        self.masks_test = jnp.ones((self.n_test, self.t_steps_test, self.observed_dims), dtype=jnp.int32)

    @staticmethod
    def drift(state):
        d = np.zeros_like(state)
        d[0] = 4. * state[1]
        d[1] = -16. * state[0]
        return d

    @staticmethod
    def diffusion(state):
        return np.array([1., 1.])

    @staticmethod
    def drift_y(state):
        d = np.zeros(5)
        d[0] = np.cos(state[0])
        d[1] = np.cos(state[1])
        d[2] = np.exp(-(state[0]**2 + state[1]**2))
        d[3] = 0.1 * (state[1] - state[1]**3)
        d[4] = 1.5
        return d

    @staticmethod
    def diffusion_y(state):
        d = np.zeros(5)
        d[0] = 1.5
        d[1] = 1.0
        d[2] = 0.1
        d[3] = np.sqrt(np.maximum(4 - 1.25 * state[1]**2, 0))
        d[4] = 1 / (1 + np.exp(-state[0]))
        return d

    @staticmethod
    def mapping(state):
        d = np.zeros((state.shape[0], 5))
        d[:, 0] = np.sqrt(state[:, 0]**2 + state[:, 1]**2)
        d[:, 1] = np.abs(state[:, 0])
        d[:, 2] = state[:, 0] - state[:, 1]
        d[:, 3] = 1 / (1 + np.exp(-state[:, 0]))
        d[:, 4] = 1.5
        return d


class ExchangeRateExtrapolation(BaseDataGenerator):
    def __init__(self):
        super().__init__()
        # Data properties
        self.n_train: int = 1
        self.n_test: int = 1000
        self.observed_dims: int = 8
        self.t_steps: int = 500
        self.t_steps_test: int = 700
        self.delta_t: float = 0.01
        self.known_drift_diffusion: bool = False

        # https://github.com/laiguokun/multivariate-time-series-data

        data = np.loadtxt("Data/time_series_data/exchange_rate.txt.gz", delimiter=",")

        data_train = data[:self.t_steps]
        data_test = data[:self.t_steps_test]

        self.ys_train = jnp.array(np.tile(data_train[None], (self.n_train, 1, 1)))
        self.ys_test = jnp.array(np.tile(data_test[None], (self.n_test, 1, 1)))

        self.masks_train = jnp.ones((self.n_train, self.t_steps, self.observed_dims), dtype=jnp.int32)
        self.masks_test = np.ones((self.n_test, self.t_steps_test, self.observed_dims), dtype=np.int32)
        self.masks_test[:, self.t_steps:] = 0
        self.masks_test = jnp.array(self.masks_test)


class ExchangeRateInterpolation(BaseDataGenerator):
    def __init__(self):
        super().__init__()
        # Data properties
        self.n_train: int = 1
        self.n_test: int = 1000
        self.observed_dims: int = 8
        self.t_steps: int = 500
        self.t_steps_test: int = 500
        self.delta_t: float = 0.01
        self.known_drift_diffusion: bool = False

        # https://github.com/laiguokun/multivariate-time-series-data

        data = np.loadtxt("Data/time_series_data/exchange_rate.txt.gz", delimiter=",")

        data_train = data[:self.t_steps]
        data_test = data[:self.t_steps_test]
        self.ys_train = jnp.array(np.tile(data_train[None], (self.n_train, 1, 1)))
        self.ys_test = jnp.array(np.tile(data_test[None], (self.n_test, 1, 1)))

        self.masks_train = np.ones((self.t_steps, self.observed_dims), dtype=np.int32)

        self.masks_train[190:250, 4] = 0
        self.masks_train[100:180, 1] = 0
        self.masks_train[350:400, 6] = 0
        self.masks_train[40:100, 6] = 0
        self.masks_train[400:475, 2] = 0
        self.masks_train[150:250, 0] = 0
        self.masks_train[10:80, 3] = 0
        self.masks_train[300:350, 7] = 0

        self.ts_test = jnp.tile(jnp.array(self.masks_train)[None], (self.n_test, 1, 1))
        self.masks_train = jnp.tile(jnp.array(self.masks_train)[None], (self.n_train, 1, 1))


datasets_dict = {
    "line": Line,
    "noisy_line": NoisyLine,
    "quadratic": Quadratic,
    "sde_1d_non_linear": SDE1DNonlinear,
    "sde_2d_linear_extrapolation": SDE2DLinearExtrapolation,
    "sde_2d_linear_interpolation": SDE2DLinearInterpolation,
    "sde_2d_to_5d_non_linear_mapping": SDE2DTo5DNonLinearMapping,
    "exchange_rate_extrapolation": ExchangeRateExtrapolation,
    "exchange_rate_interpolation": ExchangeRateInterpolation,
}
