import json
import shutil
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.tools import makedirs, pyout, listdir, fname, pbar


class COCOMaker:
    def __init__(self, root: str):
        self.root = root
        makedirs(f"{root}/coco/")

    def run(self):
        train_paths, eval_paths = self.__make_train_eval_split()

        self.make_subset("train", train_paths)
        self.make_subset("eval", eval_paths)
        pyout("Done making datasets!")

    def make_subset(self, subset, res_folders):
        data_folder = f"{self.root}/coco/{subset}"
        shutil.rmtree(data_folder, ignore_errors=True)
        makedirs(f"{data_folder}/images")
        coco = self.__init_coco_dataset()

        idx = 0
        for folder in pbar(res_folders, desc=subset):
            try:
                with open(f"{folder}/annotations_aoa.json", "r") as f:
                    A = json.load(f)
            except FileNotFoundError:
                continue
            for rel_img_path in pbar(A.keys(), desc=fname(folder)):
                img_480x640 = cv2.imread(f"{folder}/{rel_img_path}")
                img_top, img_bot, ann_top, ann_bot = \
                    self.__crop_image_top(
                        img_480x640, A[rel_img_path], idx)

                filename = f"{str(idx).zfill(5)}.jpg"
                cv2.imwrite(f"{data_folder}/images/{filename}", img_top)
                coco["annotations"].append(ann_top)
                coco["images"].append({"id": idx, "width": 512, "height": 512,
                                       "file_name": f"images/{filename}"})
                idx += 1

                filename = f"{str(idx).zfill(5)}.jpg"
                cv2.imwrite(f"{data_folder}/images/{filename}", img_bot)
                coco["annotations"].append(ann_bot)
                coco["images"].append({"id": idx, "width": 512, "height": 512,
                                       "file_name": f"images/{filename}"})
                idx += 1
        with open(f"{data_folder}/annotations.json", "w+") as f:
            json.dump(coco, f, indent=2)

    def __crop_image_top(self, img: np.ndarray, annotations: Dict[str, Any],
                         idx: int, size: int = 512
                         ) -> Tuple[
        np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:

        h, w, _ = img.shape
        img_top, img_bot = img[:w], img[-w:]
        img_top = cv2.resize(img_top, (size, size), cv2.INTER_CUBIC)
        img_bot = cv2.resize(img_bot, (size, size), cv2.INTER_CUBIC)

        uv_top = np.array(annotations['uv_coco'])
        uv_top[:, :2] = uv_top[:, :2] / w * size
        uv_bot = np.array(annotations['uv_coco'])
        uv_bot[:, 1] -= (h - w)
        uv_bot[:, :2] = uv_bot[:, :2] / w * size

        nr_of_keypoints = len(annotations['uv_coco'])

        uv_bot[np.any(uv_bot[:, :2] < 0, axis=-1), 2] = 0.
        uv_bot[np.any(uv_bot[:, :2] > size, axis=-1), 2] = 0.
        uv_top[np.any(uv_top[:, :2] < 0, axis=-1), 2] = 0.
        uv_top[np.any(uv_top[:, :2] > size, axis=-1), 2] = 0.

        ann_top = {"id": idx, "image_id": idx, "category_id": 0,
                   "keypoints": uv_top.reshape(-1).tolist(),
                   "num_keypoints": len(uv_top),
                   "theta": annotations["theta_rel"]}
        ann_bot = {"id": idx + 1, "image_id": idx + 1, "category_id": 0,
                   "keypoints": uv_bot.reshape(-1).tolist(),
                   "num_keypoints": len(uv_bot),
                   "theta": annotations["theta_rel"]}

        return img_top, img_bot, ann_top, ann_bot

    def __make_train_eval_split(self, cutoff=8):
        trials = listdir(f"{self.root}/img")
        train, eval = [], []
        for path in trials:
            towel_nr = int(fname(path).split("_")[1])
            if towel_nr < cutoff:
                train.append(path)
            else:
                eval.append(path)
        return train, eval

    def __init_coco_dataset(self) -> Dict[str, Any]:
        coco = {"categories": [{"supercategory": "cloth",
                                "id": 0,
                                "name": "towel",
                                "keypoints": ["corner0", "corner1"],
                                "skeleton": []}],
                "images": [],
                "annotations": []}
        return coco
