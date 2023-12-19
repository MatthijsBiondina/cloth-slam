from io import BytesIO

import numpy as np
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor

from utils.corporatecolors import BLUE, GREEN
from utils.tools import pyout


def plot_to_wandb(X: Tensor, y: Tensor, t: Tensor,
                  N: int = 3, title: str = "sample_plot") -> None:
    fig, axes = plt.subplots(2, N, figsize=(5 * N, 8))
    N = min(N, X.shape[0])
    for i in range(min(N, X.shape[0])):
        img = X[i].permute(1, 2, 0)
        if len(axes.shape) == 1:
            ax_img, ax_graph = axes[0], axes[1]
        else:
            ax_img, ax_graph = axes[0, i], axes[1, i]

        ax_img.imshow(img.cpu().numpy())
        ax_img.set_title(f"Image {i + 1}")
        ax_img.axis("off")

        theta = np.linspace(-np.pi, np.pi, num=y.shape[1], endpoint=False)
        ax_graph.plot(theta, y[i].cpu().detach().numpy(), label="y",
                      color=BLUE)
        ax_graph.plot(theta, t[i].cpu().detach().numpy(), label="t",
                      color=GREEN)

        ax_graph.set_title(f"Output vs Target for Sample {i + 1}")
        ax_graph.set_xlabel("Index")
        ax_graph.set_ylabel("Value")
        if i == 0:  # Only add legend to the first graph
            ax_graph.legend()

    plt.tight_layout()

    # Instead of plt.show(), save the plot to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)  # Close the figure to free up memory
    buffer.seek(0)  # Go to the beginning of the IO stream

    # Log the buffer content as an image to wandb
    wandb.log({title: wandb.Image(Image.open(buffer))})
