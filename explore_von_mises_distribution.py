import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data import DataLoader

from towel_corner_angle_net.logging_methods import plot_to_wandb, \
    plot_orientation_distribution
from towel_corner_angle_net.towel_corner_dataset import TowelCornerDataset
from towel_corner_angle_net.towel_corner_network import TowelCornerResNet
from towel_corner_angle_net.wrapped_gaussian_distribution import \
    WrappedGaussian1D
from utils.tools import pyout, pbar

ROOT = "/home/matt/Datasets/towels/angles"
DEVICE = torch.device("cuda:2")

# hyperparameters
SIZE: int = 128  # in pixels
HEATMAP_SIZE: int = 100  # number of values between (-pi, pi)
HEATMAP_SIGMA: float = 0.1 * np.pi
BATCH_SIZE: int = 4  # int

dataset = TowelCornerDataset(
    f"{ROOT}/eval", SIZE, HEATMAP_SIGMA, HEATMAP_SIZE, augment=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model = TowelCornerResNet(HEATMAP_SIZE).to(DEVICE)
checkpoint = torch.load("/home/matt/Models/angles.pth")
model.load_state_dict(checkpoint['model_state_dict'])

for X, t in pbar(dataloader):
    X, t = X.to(DEVICE), t.to(DEVICE)
    with torch.no_grad():
        y = model(X)

    gaussians = [WrappedGaussian1D(logits=y[ii]) for ii in range(y.shape[0])]
    G = FloatTensor(np.stack([g.heatmap() for g in gaussians], axis=0))

    plot_orientation_distribution(X, {"y": y, "t": t, "y'": G}).show()

    pyout()
