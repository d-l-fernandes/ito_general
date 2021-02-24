from typing import Sequence, Optional

import numpy as np
from matplotlib.pyplot import Axes

import datasets


def plot_predictions(data: datasets.BaseDataGenerator,
                     ax_arr: Sequence[Axes],
                     name: str,
                     t_seq: np.ndarray,
                     paths: np.ndarray,
                     mask: Optional[np.ndarray] = None,
                     true_paths: Optional[np.ndarray] = None,
                     y_lims: Optional[Sequence[Sequence[float]]] = None
                     ):

    for i in range(len(ax_arr)):
        ax = ax_arr[i]
        color = 'b'
        color_true = 'r'
        for j in range(min(paths.shape[0], 15)):
            ax.plot(t_seq[j], paths[j, :, i], color=color, alpha=0.2, linewidth=0.5)
        ax.plot(t_seq[0], np.mean(paths[:, :, i], axis=0), color=color)

        if true_paths is not None and mask is not None:
            mask_step = np.array(mask[0, :, i], dtype=bool)
            inverted_mask = np.array(np.abs(mask - 1, dtype=np.int32)[0, :, i], dtype=bool)
            true_paths_step = np.mean(true_paths[:, :, i], axis=0)
            true_paths_step_nan = np.array(true_paths_step)
            true_paths_step_nan[inverted_mask] = np.nan
            true_paths_step_nan_inverted = np.array(true_paths_step)
            true_paths_step_nan_inverted[mask_step] = np.nan
            ax.plot(t_seq[0], true_paths_step_nan,
                    color=color_true)
            ax.plot(t_seq[0], true_paths_step_nan_inverted, "--",
                    color=color_true)

        if y_lims is not None:
            ax.set_ylim(y_lims[i][0], y_lims[i][1])
        ax.set_ylabel(f"${name}_{i + 1}$")
        ax.set_xlabel("$t$")
        # if not data.draw_y_axis:
        #     ax.set_yticks([])
        # else:
        #     ax.set_ylabel(f"${name}_{i + 1}$")

        # if i != len(ax_arr):
        #     ax.set_xticks([])
        # else:
        #     ax.set_xlabel("$t$")
