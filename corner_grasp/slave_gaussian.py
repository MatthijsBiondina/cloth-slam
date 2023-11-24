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

        pyout(np.min(heat), np.max(heat))

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

        gmm = GaussianMixture(n_components=2)
        gmm.fit(samples)
        mu, C, a = gmm.means_, gmm.covariances_, gmm.weights_

        # Check that we are certain enough about each component
        mu = mu[a > self.GMM_CONFIDENCE_THRESHOLD]
        C = C[a > self.GMM_CONFIDENCE_THRESHOLD]
        a = a[a > self.GMM_CONFIDENCE_THRESHOLD]

        # Consider that both are the same gaussian
        # d1 = (mu[0] - mu[1]).T @ np.linalg.inv(C[0]) @ (mu[0] - mu[1])

        return mu, C, a

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
                tcp_str, img_data, heat_data = in_queue.get(timeout=1)
                tcp = pickle.loads(tcp_str)
                img = deserialize_ndarray(*img_data)
                img = self.__preprocess_image(tcp, img).astype(np.uint8)
                heat = self.__preprocess_heatmap(heat_data)
                if heat is None:
                    means = []
                else:
                    mu, C, a = self.__fit_gaussian_mixture(heat)

                    for ii in range(a.size):
                        eigenvalues, eigenvectors = np.linalg.eigh(C[ii])
                        scale = np.sqrt(eigenvalues)

                        angle = np.arctan2(eigenvectors[0, 1],
                                           eigenvectors[0, 0]) * (180 / np.pi)

                        # Center of the ellipse (mean)
                        center = (int(mu[ii][0]), int(mu[ii][1]))

                        pyout(center)

                        # Width and height of the ellipse
                        width, height = scale * 2  # 2 for 1 std deviation

                        img = cv2.ellipse(
                            img.copy(), center, (int(width), int(height)),
                            angle, 0, 360, (255, 0, 0), 2)

                    cv2.imshow("frame", img)
                    cv2.waitKey(50)

                    # pyout()









            except BreakException:
                break
            except Empty:
                pass
            finally:
                wait_for_next_cycle(t_start)

        pyout()

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
