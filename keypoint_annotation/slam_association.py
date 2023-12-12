import json
import os
from multiprocessing import Queue

import cv2
import numpy as np
from PIL import Image

from slave_slam import SLAMSlave
from utils.tools import pyout, makedirs, pbar, listdir


class KeypointAssociation:
    def __init__(self, root):
        self.root = root

    def run(self):
        for trial_name in pbar(sorted(os.listdir(f"{self.root}/img"))):
            pyout(trial_name)
            annotations_3d = self.__run_slam_on_manual_annotations(trial_name)
            self.__3d_to_camera(trial_name, annotations_3d)

    def __run_slam_on_manual_annotations(self, trial_name):
        slam = SLAMSlave(render=False)
        img_dir = f"{self.root}/img/{trial_name}"
        tcp_dir = f"{self.root}/tcp/{trial_name}"

        with open(f"{img_dir}/annotations.json", "r") as f:
            annotations = json.load(f)

        slam_output = []
        for img_rel_path in pbar(sorted(list(annotations.keys())),
                                 desc=trial_name):
            img_nr = img_rel_path.split("/")[-1].split(".")[0]

            img = Image.open(f"{img_dir}/{img_rel_path}")
            tcp = np.load(f"{tcp_dir}/{img_nr}.npy")
            kps = annotations[img_rel_path]
            if len(kps) > 0:
                slam_output = slam.process_measurement(tcp, img, kps)
        slam.shutdown()
        return slam_output

    def __3d_to_camera(self, trial_name, annotations_3d):
        slam = SLAMSlave(render=False)
        img_dir = f"{self.root}/img/{trial_name}"
        tcp_dir = f"{self.root}/tcp/{trial_name}"

        X = np.array([pt['X'] + [0, 0, 0] for pt in annotations_3d])

        with open(f"{img_dir}/annotations.json", "r") as f:
            annotations = json.load(f)
        annotations_new = {}
        # for img_rel_path in pbar(sorted(list(annotations.keys())),
        #                          desc=trial_name):
        for img_path in pbar(listdir(f"{img_dir}/images")[1:],
                             desc=trial_name):
            img_nr = img_path.split("/")[-1].split(".")[0]
            img_rel_path = f"images/{img_nr}.jpg"

            img = Image.open(f"{img_dir}/{img_rel_path}")
            tcp = np.load(f"{tcp_dir}/{img_nr}.npy")
            try:
                kps = annotations[img_rel_path]
            except KeyError:
                kps = []
            Y, P_cam = slam.compute_expected_measurement(
                X, slam.tcp2ccp(tcp), return_3d=True)
            Y = Y.reshape((-1, 2))
            img = np.array(img)[..., ::-1].copy()
            Y_coco = self.__add_visibility_flag(Y, kps, *img.shape[:2])

            annotations_new[img_rel_path] = {
                "uv_coco": Y_coco.tolist(),
                "xyz_abs": X[..., :3].tolist(),
                "xyz_rel": P_cam.tolist()
            }

            for y in Y_coco:
                x1, x2, v = int(round(y[0])), int(round(y[1])), y[2]
                if v == 2.:
                    cv2.circle(img, (x1, x2), 5, (0, 255, 0), -1)
                elif v == 1:
                    cv2.circle(img, (x1, x2), 5, (0, 127, 0), -1)
            cv2.imshow("image", img)
            cv2.waitKey(int(1000 / 60))

        with open(f"{img_dir}/annotations_slam.json", "w+") as f:
            json.dump(annotations_new, f, indent=2)

    def __add_visibility_flag(self, y, kps, img_height, img_width):
        # By default, flag=1 (on image, but not visible)
        coco = np.concatenate((y, np.ones((y.shape[0], 1))), axis=-1)

        # If outside image bounds, flag=0
        coco[(coco[:, 0] < 0) | (coco[:, 0] > img_width), -1] = 0.
        coco[(coco[:, 1] < 0) | (coco[:, 1] > img_height), -1] = 0.

        for kp in kps:
            kp = np.array(kp)[None, ...]
            distance = np.sum((coco[:, :2] - kp) ** 2, axis=-1) ** .5
            idx = np.argmin(distance)
            # If keypoint visibible, flag=2
            coco[idx, -1] = 2.

        return coco
