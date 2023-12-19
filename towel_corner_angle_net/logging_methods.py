from io import BytesIO
from typing import List, Dict

import numpy as np
import torch
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor

from utils.corporatecolors import BLUE, GREEN, SECONDARY_COLORS
from utils.tools import pyout


def plot_orientation_distribution(
        X: Tensor, distributions: Dict[str, Tensor],
        N_max: int = 4) -> Image.Image:
    N = min(N_max, X.shape[0])
    fig, axes = plt.subplots(2, N, figsize=(5 * N, 8))
    N = min(N, X.shape[0])
    for ii in range(min(N, X.shape[0])):
        img = X[ii].permute(1, 2, 0)
        if len(axes.shape) == 1:
            ax_img, ax_graph = axes[0], axes[1]
        else:
            ax_img, ax_graph = axes[0, ii], axes[1, ii]

        ax_img.imshow(img.cpu().numpy())
        ax_img.set_title(f"Image {ii + 1}")
        ax_img.axis("off")

        for jj, (label, logits) in enumerate(distributions.items()):
            P = logits / torch.sum(logits, dim=-1, keepdim=True)

            theta = np.linspace(-np.pi, np.pi, num=P.shape[1], endpoint=False)
            ax_graph.plot(theta, P[ii].cpu().detach().numpy(),
                          label=label, color=SECONDARY_COLORS[jj])

        ax_graph.set_title(f"Output vs Target for Sample {ii + 1}")
        ax_graph.set_xlabel("Index")
        ax_graph.set_ylabel("Value")
        if ii == 0:  # Only add legend to the first graph
            ax_graph.legend()

    plt.tight_layout()

    # Instead of plt.show(), save the plot to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)  # Close the figure to free up memory
    buffer.seek(0)  # Go to the beginning of the IO stream

    # Log the buffer content as an image to wandb
    return Image.open(buffer)


def plot_to_wandb(X: Tensor, y: Tensor, t: Tensor,
                  N: int = 3, title: str = "sample_plot") -> None:
    N = min(N, X.shape[0])
    img = plot_orientation_distribution(X[:N], {"y": y[:N], "t": t[:N]},
                                        N_max=N)
    wandb.log({title: wandb.Image(img)})
