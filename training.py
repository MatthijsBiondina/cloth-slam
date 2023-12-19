import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from towel_corner_angle_net.logging_methods import plot_to_wandb
from towel_corner_angle_net.towel_corner_dataset import TowelCornerDataset
from towel_corner_angle_net.towel_corner_network import TowelCornerResNet
from utils.tools import pyout, pbar, poem

ROOT = "/home/matt/Datasets/towels/angles"
DEVICE = torch.device("cuda:0")
EPOCHS = 100
PATIENCE = 10

# hyperparameters
SIZE: int = 128  # in pixels
HEATMAP_SIGMA: float = 0.1 * np.pi  # in radians
HEATMAP_SIZE: int = 100  # number of values between (-pi, pi)
BATCH_SIZE: int = 32  # int
LEARNING_RATE: float = 1e-3  # float

wandb.init(project='towel_corner_angle_net', config={
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "size": SIZE,
    "heatmap_sigma": HEATMAP_SIGMA,
    "heatmap_size": HEATMAP_SIZE,
    "device": str(DEVICE)
})

train_set = TowelCornerDataset(
    f"{ROOT}/train", SIZE, HEATMAP_SIGMA, HEATMAP_SIZE, augment=True)
valid_set = TowelCornerDataset(
    f"{ROOT}/eval", SIZE, HEATMAP_SIGMA, HEATMAP_SIZE, augment=False)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

model = TowelCornerResNet(HEATMAP_SIZE).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

step = 0
best_loss, best_epoch = None, 0
for epoch in (bar1 := pbar(range(EPOCHS))):
    if epoch - best_epoch > PATIENCE:
        break
    model.train()
    ema_loss = None
    for ii, (X, t) in (bar := pbar(enumerate(train_loader),
                                   total=len(train_loader))):
        optim.zero_grad()
        X, t = X.to(DEVICE), t.to(DEVICE)
        y = model(X)
        loss = torch.nn.functional.mse_loss(y, t)
        loss.backward()
        optim.step()
        ema_loss = loss if ema_loss is None else .1 * loss + .9 * ema_loss
        bar.desc = poem(f"Train:  {ema_loss:5f}")

        step += 1
        wandb.log({"train_step_loss": loss.item(), "step": step})

        if step % 100 == 0:
            plot_to_wandb(X, y, t, title="train_plot")
            break

    wandb.log({"train_loss": ema_loss.item(), "epoch": epoch})

    model.eval()
    with torch.no_grad():
        cum_loss, num = 0., 0.
        for X, t in (bar := pbar(valid_loader)):
            X, t = X.to(DEVICE), t.to(DEVICE)
            y = model(X)
            loss = torch.nn.functional.mse_loss(y, t)
            cum_loss += loss.cpu().item() * X.shape[0]
            num += X.shape[0]
            bar.desc = poem(f"Eval: {cum_loss / num:.5f}")
    wandb.log({"eval_loss": cum_loss / num, "epoch": epoch})
    plot_to_wandb(X, y, t, title="eval_plot")

    new_loss = cum_loss / num
    if best_loss is None or new_loss < best_loss:
        best_loss = new_loss
        best_epoch = epoch
        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': new_loss}
        torch.save(checkpoint, "/home/matt/Models/angles.pth")
        wandb.save("/home/matt/Models/angles.pth")
        bar1.desc = poem(f"Best: {best_loss:.3f}")

wandb.finish()
pyout(f"Process finished after {epoch} epochs. Loss: {best_loss:.5f}")
