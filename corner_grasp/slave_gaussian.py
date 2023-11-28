import json
import pickle
import time
from multiprocessing import Queue, Process
from queue import Empty
from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.mixture import GaussianMixture
from torch.nn.utils._expanded_weights.conv_utils import THRESHOLD

from corner_grasp.grasp_config import CAMERA_TCP
from utils.exceptions import BreakException
from utils.tools import pyout
from utils.utils import wait_for_next_cycle, deserialize_ndarray


class GaussianMixtureModel:
    THRESHOLD = 0.1
    GMM_CONFIDENCE_THRESHOLD = 0.2
    ROUNDNESS_THRESHOLD = 2
    SIZE_THRESHOLD = 0.1  # Size threshold in terms of portion of width
    N_SAMPLES = 100

    def __init__(self, in_queue: Queue):
        self.in_queue = in_queue
        self.ou_queue = Queue()
        self.sys_queue = Queue()

        self.process = Process(
            target=self.run,
            args=(self.in_queue, self.ou_queue, self.sys_queue))
        self.process.start()

    def __preprocess_heatmap(self, data: Tuple[Any, str]):
        heat = deserialize_ndarray(*data).astype(float) / 255

        heat[heat < self.THRESHOLD] = 0.

        if np.sum(heat) < 1e-6:
            return None
        else:
            return heat / np.sum(heat)

    def __fit_gaussian_mixture(self, distribution: np.ndarray):
        h, w = distribution.shape
        points = np.stack(np.mgrid[0:h - 1:h * 1j, 0:w - 1:w * 1j], axis=-1)
        II = np.random.choice(np.arange(h * w), self.N_SAMPLES,
                              p=distribution.reshape(-1))
        samples = points.reshape((-1, 2))[II][..., ::-1]

        lowest_bic = np.inf
        best_gmm = None
        bic = []
        n_components_range = range(1, 3)
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(samples)
            bic.append(gmm.bic(samples))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

        mu, C, a = best_gmm.means_, best_gmm.covariances_, best_gmm.weights_

        # Check that we are certain enough about each component
        mu = mu[a > self.GMM_CONFIDENCE_THRESHOLD]
        C = C[a > self.GMM_CONFIDENCE_THRESHOLD]
        a = a[a > self.GMM_CONFIDENCE_THRESHOLD]

        # Compute roundness of the measurements
        valid_indices = np.empty(a.shape, dtype=bool)
        for i, covariance in enumerate(C):
            eigenvalues = np.linalg.eigvalsh(covariance)
            ratio = max(eigenvalues) / min(eigenvalues)
            size_check = all(eigen ** .5 <= self.SIZE_THRESHOLD * w
                             for eigen in eigenvalues)

            valid_indices[i] = ((ratio <= self.ROUNDNESS_THRESHOLD)
                                & size_check)

        mu = mu[valid_indices]
        C = C[valid_indices]

        return mu, C

    def __preprocess_image(self,
                           tcp_end_effector: np.ndarray,
                           img: np.ndarray):
        tcp = CAMERA_TCP @ tcp_end_effector
        rotation_3d = tcp[:3, :3]
        theta = np.arctan2(rotation_3d[2, 0], rotation_3d[1, 0])

        if theta > 0:  # rotated 90 degrees
            img = img.transpose(1, 0, 2)[::-1]
        else:
            img = img.transpose(1, 0, 2)[:, ::-1]

        return img[..., ::-1]

    def run(self, in_queue, ou_queue, sys_queue):
        while True:
            t_start = time.time()
            try:
                self.__process_sys_command(sys_queue)
                tcp_str, img_data, depth_data, heat_data = \
                    in_queue.get(timeout=1)
                tcp = pickle.loads(tcp_str)
                img = deserialize_ndarray(*img_data)
                # img = self.__preprocess_image(tcp, img).astype(np.uint8)
                heat = self.__preprocess_heatmap(heat_data)
                if heat is None:
                    continue
                else:
                    mu, C = self.__fit_gaussian_mixture(heat)
                    if mu is None:
                        continue

                    kp = {'mu': mu.tolist(), 'C': C.tolist()}
                    ou_queue.put((tcp_str, img_data, depth_data,
                                  json.dumps(kp)))

            except BreakException:
                break
            except Empty:
                pass
            finally:
                wait_for_next_cycle(t_start)

    def __process_sys_command(self, sys_queue: Queue):
        if not sys_queue.empty():
            msg = sys_queue.get()
            if msg == "shutdown":
                raise BreakException()
            else:
                raise ValueError(f"Message {msg} unknown.")

    def shutdown(self):
        self.sys_queue.put("shutdown")
        self.process.join()
