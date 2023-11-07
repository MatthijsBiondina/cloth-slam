"""example script for inference on local image with a saved model checkpoint"""

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from keypoint_detection.utils.heatmap import \
    get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.load_checkpoints import \
    get_model_from_wandb_checkpoint, load_from_checkpoint

import requests


def local_inference(model, image: np.ndarray, device="cuda"):
    """
    inference on a single image as if you would load the image from disk or
    get it from a camera. Returns a list of the extracted keypoints for each
    channel.
    """
    # assert model is in eval mode! (important for batch norm layers)
    assert model.training == False, \
        "model should be in eval mode for inference"

    # convert image to tensor with correct shape (channels, height, width)
    # and convert to floats in range [0,1], add batch dimension, and move to
    # device
    image = to_tensor(image).unsqueeze(0).to(device)

    # pass through model
    with torch.no_grad():
        heatmaps = model(image).squeeze(0)

    # extract keypoints from heatmaps
    predicted_keypoints = \
        get_keypoints_from_heatmap_batch_maxpool(heatmaps.unsqueeze(0))[0]

    return predicted_keypoints, heatmaps


class PretrainedModel:
    def __init__(self, checkpoint_name, gpu=None):
        # self.model = get_model_from_wandb_checkpoint(checkpoint_name)
        self.model = load_from_checkpoint(checkpoint_name)
        self.model.eval()
        if gpu is None:
            self.model.cuda()
        else:
            self.model.cuda(gpu)
        self.device = self.model.device

    def __call__(self, img):
        img = to_tensor(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmaps = self.model(img).squeeze(0)

        predicted_keypoints = get_keypoints_from_heatmap_batch_maxpool(
            heatmaps.unsqueeze(0))[0]

        return predicted_keypoints, heatmaps
