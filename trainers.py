import os
import time
import shutil
from functools import partial
from typing import Any, Tuple, List
import gc

import haiku as hk
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import jsonpickle
import optax
import tensorboardX
from absl import flags, logging
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from datasets import datasets_dict, Batch
from models import ItoGeneral, ItoGeneralOutput, Metrics
import aux_plot

flags.DEFINE_integer("batch_size", 1, "Batch Size.")
flags.DEFINE_integer("batch_size_eval", 1, "Batch Size for evaluation.")
flags.DEFINE_integer("training_steps", 150000, "Number of training steps.")
flags.DEFINE_integer("max_step_diff", 20000, "Number of training steps without improvement before it stops.")
flags.DEFINE_integer("eval_frequency", 1000, "How often to evaluate the model.")
flags.DEFINE_integer("predict", 0, "Do Predictions (0 - only train, 1 - train and predict, 2 - only predict",
                     lower_bound=0, upper_bound=2)

flags.DEFINE_bool("restore", False, "Whether to restore previous params from checkpoint.", short_name="r")
flags.DEFINE_bool("erase", False, "Whether to erase previous checkpoints and summaries.", short_name="e")
flags.DEFINE_bool("debug", False, "Whether to evaluate model every eval_frequency", short_name="d")

flags.DEFINE_float("learning_rate", 0.001, "Learning rate of the optimizer.")

flags.DEFINE_multi_enum("not_to_train", [], ["drift", "diffusion", "mapping", "initial_latents", "likelihood"],
                        "Parts of the model not to train.")

FLAGS = flags.FLAGS

PRNGKey = jnp.ndarray
Array = jnp.ndarray
OptState = Any


class ItoGeneralTrainer:
    def __init__(self):
        # Folders
        parent_folder = f"results/{FLAGS.dataset}/{FLAGS.mapping}/drift_{FLAGS.drift}/diffusion_{FLAGS.diffusion}/"
        parent_folder += f"latent_{FLAGS.latent_dims}D"
        self.summary_folder = parent_folder + "/summary/"
        self.results_folder = parent_folder + "/results/"
        self.checkpoint_folder = parent_folder + "/checkpoint/"

        # Erase summary and checkpoint folders
        if FLAGS.erase and (FLAGS.restore or FLAGS.predict == 2):
            raise RuntimeError("Can't erase previous checkpoints and then restore params from them.")

        if FLAGS.erase:
            if os.path.exists(self.summary_folder):
                shutil.rmtree(self.summary_folder, ignore_errors=True)
                shutil.rmtree(self.checkpoint_folder, ignore_errors=True)

        # Create Folders
        for d in [self.summary_folder, f"{self.results_folder}latent", f"{self.results_folder}observed",
                  self.checkpoint_folder]:
            if not os.path.exists(d):
                os.makedirs(d)

        # Logging
        logging.get_absl_handler().use_absl_log_file("log", self.summary_folder)
        logging.info(f"CURRENT SETUP: {parent_folder}")
        print(f"CURRENT SETUP: {parent_folder}")

        # Data
        self.data = datasets_dict[FLAGS.dataset]()
        self.train_generator = self.data.train_generator(FLAGS.batch_size)

        # Model
        self.model = hk.transform(lambda y_input, t_mask, training: ItoGeneral(self.data)(y_input, t_mask, training))

        # Train RNG sequence
        self.rng_seq = hk.PRNGSequence(42)

        # Test RNG sequence
        self.rng_seq_test = hk.PRNGSequence(43)

        # Optimizer
        self.optim = optax.adam(FLAGS.learning_rate)

        # Params
        if not (FLAGS.restore or FLAGS.predict == 2):
            single_sample = self.data.single_sample()
            params = self.model.init(rng=next(self.rng_seq), y_input=single_sample[0], t_mask=single_sample[1],
                                     training=1)

            trainable_params, non_trainable_params = hk.data_structures.partition(
                lambda m, n, p: not any(i in m for i in FLAGS.not_to_train), params)

            self.opt_state = self.optim.init(trainable_params)
            objective = -jnp.inf
            train_step = 0
        else:
            objective, params, self.opt_state, train_step = self.load()

        # Best values
        self.best_params = params
        self.best_objective = objective
        self.cur_train_step = train_step
        self.cur_train_diff = 0

    def train(self) -> None:
        test_summary_writer = tensorboardX.SummaryWriter(f"{self.summary_folder}test")
        if FLAGS.predict < 2:
            # Summary writer
            train_summary_writer = tensorboardX.SummaryWriter(f"{self.summary_folder}train")

            # Initial params
            params = self.best_params

            # Train cycle
            for step in range(self.cur_train_step, FLAGS.training_steps):
                # Train step
                tic = time.time()
                objective, metrics, params, self.opt_state = self.update(params,
                                                                         self.opt_state,
                                                                         next(self.rng_seq),
                                                                         next(self.train_generator))
                # Log metrics
                for k, v in metrics._asdict().items():
                    train_summary_writer.add_scalar(f"Metrics/{k}", v, step)

                # Check if there was improvement and save if there was
                if objective > self.best_objective:
                    self.best_objective = objective
                    self.best_params = params
                    self.cur_train_diff = 0
                    self.save()
                else:
                    self.cur_train_diff += 1
                    # Stops training if it hasn't been improving for a while
                    if self.cur_train_diff == FLAGS.max_step_diff:
                        logging.info(f"Training stopped as it was not improving (Step {step})")
                        break

                if step % FLAGS.eval_frequency == 0:
                    logging.info(f"Step {step:6d}: {np.array(objective):.4f} ({time.time() - tic:.3f} sec)")
                    self.evaluate(test_summary_writer, step=step)

                self.cur_train_step += 1

            # Close summary writer
            train_summary_writer.close()

        # Do final evaluation
        if FLAGS.predict > 0:
            self.evaluate(test_summary_writer, self.cur_train_step, True)

        test_summary_writer.close()

    @partial(jax.jit, static_argnums=(0,))
    def loss(self,
             trainable_params: hk.Params,
             non_trainable_params: hk.Params,
             rng_key: PRNGKey,
             batch: Batch,
             training: int) -> Tuple[Array, Tuple[Metrics, ItoGeneralOutput]]:
        params = hk.data_structures.merge(trainable_params, non_trainable_params)
        rng_key_list = random.split(rng_key, batch[0].shape[0])

        metrics: Metrics
        output: ItoGeneralOutput
        metrics, output = \
            jax.vmap(self.model.apply, (None, 0, 0, 0, None))(params, rng_key_list, batch[0], batch[1], training)

        mean_metrics = Metrics(jnp.mean(metrics.elbo), jnp.mean(metrics.elbo_generated), jnp.mean(metrics.mse))

        return -mean_metrics.elbo, (mean_metrics, output)

    @partial(jax.jit, static_argnums=(0,))
    def update(self,
               params: hk.Params,
               opt_state: OptState,
               rng_key: PRNGKey,
               batch: Batch) -> Tuple[Array, Metrics, hk.Params, OptState]:
        trainable_params, non_trainable_params = hk.data_structures.partition(
            lambda m, n, p: not any(i in m for i in FLAGS.not_to_train), params)

        metrics: Metrics
        output: ItoGeneralOutput
        grads, (metrics, output) = jax.grad(self.loss, 0, has_aux=True)(trainable_params,
                                                                        non_trainable_params,
                                                                        rng_key,
                                                                        batch, 1)
        updates, new_opt_state = self.optim.update(grads, opt_state, None)
        trainable_params = optax.apply_updates(trainable_params, updates)

        params = hk.data_structures.merge(trainable_params, non_trainable_params)
        return metrics.elbo, metrics, params, new_opt_state

    def evaluate(self, summary_writer: tensorboardX.SummaryWriter, step: int,
                 save_in_folder: bool = False):
        test_generator = self.data.test_generator(FLAGS.batch_size_eval)
        trainable_params, non_trainable_params = hk.data_structures.partition(
            lambda m, n, p: not any(model_part in m for model_part in FLAGS.not_to_train), self.best_params)

        _, (metrics, output) = self.loss(trainable_params, non_trainable_params, next(self.rng_seq_test),
                                         next(test_generator), 0)
        count = 1

        n_iter = self.data.n_test // FLAGS.batch_size_eval
        if n_iter > 1:
            for i in range(n_iter-1):
                count += 1
                _, (metrics_step, output_step) = self.loss(trainable_params, non_trainable_params,
                                                           next(self.rng_seq_test),
                                                           next(test_generator), 0)
                metrics += metrics_step
                output += output_step

        for k, v in metrics._asdict().items():
            summary_writer.add_scalar(f"Metrics/{k}", v / count, step)

        if FLAGS.debug:
            self.make_plot_figs(output, summary_writer, step, save_in_folder)

    def make_plot_figs(self, output: ItoGeneralOutput,
                       summary_writer: tensorboardX.SummaryWriter,
                       step: int,
                       save_in_folder: bool = False) -> None:
        plt.ioff()

        ax_list: List[Axes] = []
        fig_list: List[Figure] = []
        name_list: List[str] = []

        for t in ["paths", "drifts", "diffusions"]:

            # Latent space
            for i in range(FLAGS.latent_dims):
                fig = plt.figure(figsize=(15, 4))
                fig.tight_layout()
                ax = fig.add_subplot()

                ax_list.append(ax)
                fig_list.append(fig)
                name_list.append(f"latent/{t}_{i}")

            if t == "paths":
                aux_plot.plot_predictions(self.data, ax_list, "x", output.t_seq, output.paths_x)
            elif t == "drifts":
                aux_plot.plot_predictions(self.data, ax_list, "f(x, t)", output.t_seq, output.drift_x)
            elif t == "diffusions":
                aux_plot.plot_predictions(self.data, ax_list, "L(x, t)", output.t_seq, output.diffusion_x)

            self.save_figs(name_list, fig_list, summary_writer, step, save_in_folder)
            ax_list.clear()
            fig_list.clear()
            name_list.clear()

            # Observed space
            for i in range(self.data.observed_dims):
                fig = plt.figure(figsize=(15, 4))
                fig.tight_layout()
                ax = fig.add_subplot()

                ax_list.append(ax)
                fig_list.append(fig)
                name_list.append(f"observed/{t}_{i}")

            if t == "paths":
                aux_plot.plot_predictions(self.data, ax_list, "y", output.t_seq, output.paths_y,
                                          output.mask, self.data.ys_test, self.data.y_lims)
            elif t == "drifts":
                aux_plot.plot_predictions(self.data, ax_list, "f_y(x, t)", output.t_seq, output.drift_y,
                                          output.mask, self.data.drifts_test)
            elif t == "diffusions":
                aux_plot.plot_predictions(self.data, ax_list, "L_y(x, t)", output.t_seq, output.diffusion_y,
                                          output.mask, self.data.diffusions_test)

            self.save_figs(name_list, fig_list, summary_writer, step, save_in_folder)
            ax_list.clear()
            fig_list.clear()
            name_list.clear()

        gc.collect()

    def save_figs(self, fig_names: List[str],
                  figs: List[Figure],
                  summary_writer: tensorboardX.SummaryWriter,
                  step: int,
                  save_in_folder: bool = False) -> None:
        if save_in_folder:
            for i in range(len(fig_names)):
                figs[i].savefig(f"{self.results_folder}{fig_names[i]}.pdf")
                plt.close(figs[i])
        else:
            for i in range(len(fig_names)):
                summary_writer.add_figure(fig_names[i], figs[i], step)

    def save(self) -> None:
        save_dict = {"params": self.best_params,
                     "opt_state": self.opt_state,
                     "objective": self.best_objective,
                     "train_step": self.cur_train_step}
        with open(f"{self.checkpoint_folder}checkpoint.json", "w") as f:
            f.write(jsonpickle.encode(save_dict))

    def load(self) -> Tuple[Array, hk.Params, OptState, int]:
        checkpoints = os.listdir(self.checkpoint_folder)
        if checkpoints:
            print(f"Loading model checkpoint...")
            with open(f"{self.checkpoint_folder}checkpoint.json", 'r') as f:
                s = jsonpickle.decode(f.read())
            print("Model loaded")
            return s["objective"], s["params"], s["opt_state"], s["train_step"]
        else:
            raise RuntimeError("No checkpoint to load from.")
