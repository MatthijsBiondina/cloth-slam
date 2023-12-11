import json
import numpy as np
from PIL import Image
from utils.tools import makedirs, listdir, pyout, pbar
from utils.utils import deserialize_ndarray


def process_trial_data(root):
    """
    Processes trial data stored in JSON format and saves the images and TCP data
    in separate directories.

    Parameters:
    root (str): The root directory where the datasets are stored.
    """

    makedirs(f"{root}/img")
    makedirs(f"{root}/tcp")

    for trial in pbar(listdir(f"{root}/json"), desc="Processing Trials"):
        makedirs(trial.replace("/json/", "/img/"))
        makedirs(trial.replace("/json/", "/tcp/"))

        for frame_path in pbar(listdir(trial), desc=trial.split("/")[-1]):
            with open(frame_path, "r") as f:
                d = json.load(f)

            img = deserialize_ndarray(*d['img'])
            img = Image.fromarray(img)
            img = img.rotate(90, expand=True)
            img.save(frame_path.replace("/json/", "/img/").replace(".json",
                                                                   ".jpg"))

            tcp = deserialize_ndarray(*d['tcp'], dtype=np.float64)
            np.save(frame_path.replace("/json/", "/tcp/").replace(".json",
                                                                  ".npy"),
                    tcp)


if __name__ == '__main__':
    root = "/home/matt/Datasets/towels"
    process_trial_data(root)
