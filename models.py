from __future__ import annotations

from typing import Tuple, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
from absl import flags

import aux
from datasets import BaseDataGenerator
from jax_aux import aux_math
from sde.diffusions import diffusions_dict, DiffusionIto
from sde.drifts import drifts_dict, DriftIto
from sde.mappings import mappings_dict
from sde.sde_map import SDEMap
from sde.solvers import solvers_dict

flags.DEFINE_integer("latent_dims", 2, "Number of latent dimensions.", lower_bound=1)
FLAGS = flags.FLAGS

Array = jnp.ndarray


class Metrics(NamedTuple):
    elbo: Array
    elbo_generated: Array
    mse: Array

    def __add__(self, other: Metrics) -> Metrics:
        return Metrics(self.elbo + other.elbo,
                       self.elbo_generated + other.elbo_generated,
                       self.mse + other.mse)


class ItoGeneralOutput(NamedTuple):
    t_seq: Array
    paths_x: Array
    drift_x: Array
    diffusion_x: Array
    paths_y: Array
    drift_y: Array
    diffusion_y: Array
    mask: Array

    def __add__(self, other: ItoGeneralOutput) -> ItoGeneralOutput:
        return ItoGeneralOutput(jnp.concatenate([self.t_seq, other.t_seq]),
                                jnp.concatenate([self.paths_x, other.paths_x]),
                                jnp.concatenate([self.drift_x, other.drift_x]),
                                jnp.concatenate([self.diffusion_x, other.diffusion_x]),
                                jnp.concatenate([self.paths_y, other.paths_y]),
                                jnp.concatenate([self.drift_y, other.drift_y]),
                                jnp.concatenate([self.diffusion_y, other.diffusion_y]),
                                jnp.concatenate([self.mask, other.mask]))


class ItoGeneral(hk.Module):
    def __init__(self, data: BaseDataGenerator, name: str = "ito_general"):
        super().__init__(name)
        self.data = data
        self.mapping = mappings_dict[FLAGS.mapping](self.data.observed_dims)
        self.initial_latents = aux.InitialLatent(FLAGS.latent_dims)

        # Latent
        self.drift_x = drifts_dict[FLAGS.drift](FLAGS.latent_dims)
        self.diffusion_x = diffusions_dict[FLAGS.diffusion](FLAGS.latent_dims)

        self.likelihood = aux.Likelihood(self.data.observed_dims)

        solver_x = solvers_dict[FLAGS.solver](delta_t=self.data.delta_t,
                                              beta_dims=FLAGS.latent_dims,
                                              drift=self.drift_x,
                                              diffusion=self.diffusion_x
                                              )
        self.sde = SDEMap(delta_t=self.data.delta_t,
                          solver_x=solver_x,
                          mapping=self.mapping,
                          likelihood=self.likelihood)

        # Observed
        self.drift_y = DriftIto(self.mapping, self.drift_x, self.diffusion_x)
        self.diffusion_y = DiffusionIto(self.mapping, self.diffusion_x, self.likelihood)

    def __call__(self, y_input: Array, t_mask: Array, training: int) -> Tuple[Metrics, ItoGeneralOutput]:
        x_0 = self.initial_latents()
        likelihood = self.likelihood()

        t_seq, paths_x, paths_y, paths_y_generated = self.sde(x_0, y_input, t_mask, training)

        # Drifts and diffusions
        drift_y = jax.vmap(lambda x, t: self.drift_y(x, t), (0, 0))(paths_x, t_seq)
        diffusion_y = jax.vmap(lambda x, t: self.diffusion_y(x, t), (0, 0))(paths_x, t_seq)

        drift_x = jax.vmap(lambda x, t: self.drift_x(x, t), (0, 0))(paths_x, t_seq)
        diff_x = aux_math.diag_part(jax.vmap(lambda x, t: self.diffusion_x(x, t), (0, 0))(paths_x, t_seq))

        # Objectives
        y_objective = y_input * t_mask + jnp.abs(t_mask - 1) * paths_y

        elbo = jnp.mean(aux_math.log_prob_multivariate_normal(
            paths_y[1:],
            aux_math.diag(diffusion_y[:-1] * jnp.sqrt(self.data.delta_t)),
            y_objective[1:]))
        elbo_generated = jnp.mean(aux_math.log_prob_multivariate_normal(
            paths_y_generated[1:],
            aux_math.diag(diffusion_y[:-1] * jnp.sqrt(self.data.delta_t)),
            y_objective[1:]))
        mse = jnp.mean(jnp.sum((paths_y[1:] - y_objective[1:])**2, axis=-1))
        paths_y = paths_y_generated * training + jnp.abs(1 - training) * paths_y

        return \
            Metrics(elbo, elbo_generated, mse), \
            ItoGeneralOutput(t_seq, paths_x, drift_x, diff_x, paths_y, drift_y, diffusion_y, t_mask)
