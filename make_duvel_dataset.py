import json
import os
from typing import List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw

from utils.tools import pyout, makedirs, pbar


def init_coco_dataset():
    # Initialize COCO dataset structure
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"supercategory": "cloth",
                        "id": 0,
                        "name": "towel",
                        "keypoints": ["corner0", "corner1"],
                        "skeleton": []}]  # Define your categories here
    }
    return coco_dataset


def preprocess(img: Image.Image, A: List[List[float]]):
    W, H = img.width, img.height

    box1 = (0, 0, W, W)
    box2 = (0, H - W, W, H)

    img1 = img.crop(box1)
    img2 = img.crop(box2)

    A1 = [x for x in A if x[1] < W]
    A2 = [[x[0], x[1] - (H - W)] for x in A if x[1] > H - W]

    img1 = img1.resize((512, 512))
    img2 = img2.resize((512, 512))

    A1 = (np.array(A1) * (512 / W)).tolist()
    A2 = (np.array(A2) * (512 / W)).tolist()

    return img1, img2, A1, A2


def annotate_and_save(coco: Dict[str, Any], img: Image.Image,
                      A: List[List[float]], id: int, path: str):
    coco["images"].append({
        "id": id,
        "width": img.width,
        "height": img.height,
        "file_name": f"images/{str(id).zfill(4)}.jpg"})
    coco["annotations"].append({
        "id": id,
        "image_id": id,
        "category_id": 0,
        "keypoints": [x for a in A for x in (a[0], a[1], 2.0)],
        "num_keypoints": len(A)
    })
    img.save(f"{path}/{str(id).zfill(4)}.jpg")

    return coco, id + 1


def make_dataset(subset: str, trials: List[int]):
    img_dir = f"{OU_ROOT}/{subset}/images"
    makedirs(img_dir)
    coco = init_coco_dataset()

    idx = 1
    for trial in pbar(trials):
        trial_dir = f"{IN_ROOT}/{str(trial).zfill(3)}"
        with open(f"{trial_dir}/opticalflow_annotations.json", "r") as f:
            flow = json.load(f)
        for fname in pbar(flow.keys()):
            img = Image.open(f"{trial_dir}/images/{fname}")
            A = flow[fname]
            img1, img2, A1, A2 = preprocess(img, A)
            coco, idx = annotate_and_save(coco, img1, A1, idx, img_dir)
            coco, idx = annotate_and_save(coco, img2, A2, idx, img_dir)

    with open(f"{OU_ROOT}/{subset}/annotations.json", "w+") as f:
        json.dump(coco, f)



IN_ROOT = "/home/matt/Datasets/real"
OU_ROOT = "/home/matt/Datasets/duvel"

if __name__ == "__main__":
    make_dataset("train", list(range(17)))
    make_dataset("eval", list(range(17, 21)))

