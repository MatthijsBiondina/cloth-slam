import json

import cv2
import numpy as np

from slave_slam import SLAMSlave
from utils.tools import listdir, pbar, pyout, fname


def show(img):
    img = cv2.resize(np.copy(img), img.shape[:2][::-1])
    cv2.imshow("Frame", img)
    return cv2.waitKey(-1)


class AngleOfApproachLabeler:
    def __init__(self, root):
        self.root = root

    def run(self):
        slam = SLAMSlave()
        for trial_path in pbar(listdir(f"{self.root}/img")):
            with open(f"{trial_path}/annotations_slam.json", "r") as f:
                annotations = json.load(f)
            annotations, nr_visible_keypoints = self.__init_theta_annotation(
                annotations)
            imgs, tcps = self.__load_images_and_tcp(trial_path, annotations)
            frames = list(imgs.keys())

            for kp_idx in range(nr_visible_keypoints):
                ii, theta = 0, 0.5 * np.pi
                while True:
                    img = imgs[frames[ii]].copy()
                    origin = np.array(
                        annotations[frames[ii]]['xyz_abs'][kp_idx])
                    tcp = tcps[frames[ii]]
                    ccp = slam.tcp2ccp(tcp)

                    x_axis, y_axis, z_axis = self.__compute_axes(
                        theta, origin)

                    origin_uv = slam.compute_expected_measurement(
                        np.concatenate((origin, np.zeros((3,)))), ccp)
                    x_uv = slam.compute_expected_measurement(
                        np.concatenate((x_axis, np.zeros((3, 1)))), ccp)
                    y_uv = slam.compute_expected_measurement(
                        np.concatenate((y_axis, np.zeros((3)))), ccp)
                    z_uv = slam.compute_expected_measurement(
                        np.concatenate((z_axis, np.zeros((3, 1)))), ccp)

                    origin_uv = tuple(int(round(x)) for x in origin_uv[:, 0])
                    x_uv = tuple(int(round(x)) for x in x_uv[:, 0])
                    y_uv = tuple(int(round(x)) for x in y_uv[:, 0])
                    z_uv = tuple(int(round(x)) for x in z_uv[:, 0])

                    img = cv2.line(img, origin_uv, y_uv, (0, 255, 0), 2)
                    img = cv2.line(img, origin_uv, x_uv, (0, 0, 255), 2)
                    img = cv2.line(img, origin_uv, z_uv, (255, 0, 0), 2)

                    k = show(img)
                    if k == 27:
                        break
                    elif k == ord('d'):
                        ii = min(ii + 1, len(frames) - 1)
                    elif k == ord('a'):
                        ii = max(1, ii - 1)
                    elif k == ord('j'):
                        theta = (theta - np.pi / 180) % (2 * np.pi)
                    elif k == ord('l'):
                        theta = (theta + np.pi / 180) % (2 * np.pi)
                    elif k == ord('j'):
                        pyout()

    def __compute_axes(self, theta, origin, length=0.1):
        th = theta
        R = np.array([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th), np.cos(th), 0],
                      [0, 0, 1]])
        x_axis = R @ np.array([0, -1, 0])[:, None] * length + origin[:, None]
        y_axis = np.array([0, 0, -1]) * length + origin
        z_axis = R @ np.array([1, 0, 0])[:, None] * length + origin[:, None]

        return x_axis, y_axis, z_axis

    def __init_theta_annotation(self, annotations):
        nr_visible_keypoints = None
        for key in annotations.keys():
            d = annotations[key]
            if nr_visible_keypoints is None:
                nr_visible_keypoints = len(d['uv_coco'])
            assert nr_visible_keypoints == len(d['uv_coco'])
            annotations[key]['theta_rel'] = [
                None for _ in range(nr_visible_keypoints)]
        return annotations, nr_visible_keypoints

    def __load_images_and_tcp(self, trial_path, annotations):
        imgs, tcps = {}, {}
        tcp_path = trial_path.replace("/img/", "/tcp/")

        for rel_path in pbar(annotations.keys(), desc="Loading Images"):
            abs_path = f"{trial_path}/{rel_path}"
            imgs[rel_path] = cv2.imread(abs_path)

            tcp_file = fname(rel_path).replace('.jpg', '.npy')
            tcps[rel_path] = np.load(f"{tcp_path}/{tcp_file}")

        return imgs, tcps
