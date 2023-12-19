import json
from random import random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor, FloatTensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose

from towel_corner_angle_net.PODs import SampleMetaData, Coord
from utils.tools import pyout, listdir, pbar

DEBUG_MODE = False


class TowelCornerDataset(Dataset):
    def __init__(self, root: str,
                 size: int = 128,  # in pixels
                 heatmap_sigma: float = 0.1 * np.pi,  # in radians
                 heatmap_size: int = 100,
                 augment: bool = False):
        self.root: str = root
        self.width: int = size
        self.height: int = size
        self.heatmap_sigma: float = heatmap_sigma
        self.heatmap_size: int = heatmap_size

        self.transform: Compose = self.__init_transform(augment)

        self.samples: List[SampleMetaData] = self.__init_data()

    def __len__(self):
        """
        Returns: length of dataset
        """
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]

        x = self.__open_and_preprocess_image(meta.path, meta.center)
        y = self.__wrapped_gaussian_heatmap(
            meta.angle, self.heatmap_sigma, self.heatmap_size)

        return x, y

    def __init_transform(self, data_augmentation: bool) -> Compose:
        if data_augmentation:
            return transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomRotation(3),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor()
            ])

    def __init_data(self) -> List[SampleMetaData]:
        """
        Initializes and loads the dataset by extracting metadata from
        annotation files.

        This method iterates through the dataset directories located in
        `self.root`. For each trial (subdirectory) in the dataset, it
        attempts to load the 'annotations_aoa.json' file. If this file is
        found, the method parses it and processes each annotation to extract
        frame metadata. The extracted metadata is then accumulated in a list.

        In DEBUG_MODE, the method processes only the first trial for quicker
        testing.

        Returns:
            List[SampleMetaData]: A list containing the metadata for
            each frame in the dataset. Each item in the list is an instance
            of `CornerSampleMetadata`, which holds the relevant metadata for
            a single frame.

        Notes:
            - The method uses a progress bar (pbar) to display the dataset loading status.
            - If an annotation file is not found in a trial directory, that trial is skipped.
            - The actual structure and content of `CornerSampleMetadata` depend on the
              implementation of the `__extract_frame_metadata` method.
        """
        # Function implementation remains the same

        samples = []

        for trial in pbar(listdir(self.root), desc="Loading Datset"):
            try:
                with open(f"{trial}/annotations_aoa.json", "r") as f:
                    annotations = json.load(f)
            except FileNotFoundError:
                continue

            for rel_path, meta in annotations.items():
                samples.extend(
                    self.__extract_frame_metadata(trial, rel_path, meta))

            if DEBUG_MODE:
                break

        return sorted(samples, key=lambda _: random())

    def __extract_frame_metadata(self,
                                 trial: str,
                                 rel_path: str,
                                 metadata: Dict[str, Any]
                                 ) -> List[SampleMetaData]:
        """
            Extracts frame metadata from given metadata for a specific trial and relative path.

            Args:
                trial (str): The trial identifier.
                rel_path (str): Relative path to the frame within the trial.
                metadata (Dict[str, Any]): The metadata dictionary containing 'theta_rel' and 'uv_coco'.

            Returns:
                List[SampleMetaData]: A list of `CornerSampleMetadata` instances, each representing
                metadata for a frame. Only frames where the last element of 'uv_coco' is close to 2.0 are included.

            Note:
                The method constructs absolute paths for each frame and extracts the center coordinates and
                theta values from the metadata.
        """
        frame_samples = []

        abs_path = f"{trial}/{rel_path}"
        for theta, uv in zip(metadata['theta_rel'], metadata['uv_coco']):
            if np.isclose(uv[-1], 2.0):
                x_center, y_center = [int(x) for x in uv[:2]]
                frame_samples.append(SampleMetaData(
                    abs_path, Coord(x_center, y_center), theta))
        return frame_samples

    def __open_and_preprocess_image(self, path: str, center: Coord
                                    ) -> Tensor:

        img = Image.open(path)

        crop_box = (
            max(0, center.x - self.width // 2),  # left
            max(0, center.y - self.height // 2),  # top
            min(img.width, center.x + (self.width + 1) // 2),  # right
            min(img.height, center.y + (self.height + 1) // 2))  # bottom

        cropped_img = img.crop(crop_box)

        black_background = Image.new(
            'RGB', (self.width, self.height), (0, 0, 0))
        black_background.paste(cropped_img, (0, 0))

        return self.transform(black_background)

    def __wrapped_gaussian_heatmap(self, mean: float, std_dev: float, N: int
                                   ) -> Tensor:  # todo utils?
        x = np.linspace(-np.pi, np.pi, num=N, endpoint=False)

        wrapped_distance = np.minimum(
            np.abs(x - mean), 2 * np.pi - np.abs(x - mean))

        logits = np.exp(-.5 * (wrapped_distance / std_dev) ** 2) \
                 / (std_dev * np.sqrt(2 * np.pi))
        logits = logits / np.max(logits)

        return FloatTensor(logits)


if __name__ == "__main__":
    dataset = TowelCornerDataset("/home/matt/Datasets/towels/img")
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for X, t in data_loader:
        pyout()

    pyout()
