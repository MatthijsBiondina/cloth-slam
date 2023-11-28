import json
import pickle
import time
from multiprocessing import Queue, Process
from queue import Empty

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
from matplotlib import pyplot as plt

from corner_grasp.grasp_config import CAMERA_TCP
from utils.exceptions import BreakException
from utils.tools import pyout
from utils.utils import wait_for_next_cycle, deserialize_ndarray


class OrientationSlave:
    KERNEL_SIZE = 64

    def __init__(self, in_queue: Queue):
        self.in_queue = in_queue
        self.ou_queue = Queue()
        self.sys_queue = Queue()

        self.process = Process(
            target=self.run,
            args=(self.in_queue, self.ou_queue, self.sys_queue))
        self.process.start()

    def show(self, d: np.ndarray):
        cv2.imshow("Corner", d)
        cv2.waitKey(50)

    def determine_corner_facing(self, dep: np.ndarray, pt: np.ndarray):
        H, W = dep.shape
        try:
            x_frm = int(max(0, pt[0] - self.KERNEL_SIZE // 2))
            x_to = int(min(W, pt[0] + self.KERNEL_SIZE // 2))
            y_frm = int(max(0, pt[1] - self.KERNEL_SIZE // 2))
            y_to = int(min(H, pt[1] + self.KERNEL_SIZE // 2))

            slice_3d = dep[y_frm:y_to, x_frm:x_to]

            edges = cv2.Canny((slice_3d * 255).astype(np.uint8),
                              50, 100)
            contours, _ = contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)

            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            idx_sort = np.argsort(approx[:, 0, 1])
            approx = np.concatenate(
                (np.array([[[0, 0]]]),
                 approx[idx_sort],
                 np.array([[[0, edges.shape[0] - 1]]])))

            mask = np.zeros(slice_3d.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [approx], 255)
            mask = mask > 125
            slice_3d[slice_3d < 1e-6] = 1000.

            L = np.sum(slice_3d[mask]) / np.sum(mask)
            R = np.sum(slice_3d[~mask]) / np.sum(~mask)

            return 'True' if L < R else 'False'
        except Exception:
            return 'None'

    def run(self, in_queue: Queue, ou_queue: Queue, sys_queue: Queue):
        while True:
            t_start = time.time()
            try:
                self.__process_sys_command(sys_queue)
                tcp_str, img_data, depth_data, kp_str = \
                    in_queue.get(timeout=1)
                tcp = pickle.loads(tcp_str)
                img = deserialize_ndarray(*img_data)
                img = self.__preprocess_image(tcp, img)

                dep = deserialize_ndarray(*depth_data, dtype=np.uint16)
                dep = self.__preprocess_depth_map(tcp, dep)
                keypoints = json.loads(kp_str)
                keypoints['x-facing'] = []
                for pt in keypoints['mu']:
                    keypoints['x-facing'].append(self.determine_corner_facing(
                        dep, pt))

                kp_str = json.dumps(keypoints)
                ou_queue.put((tcp_str, img_data, depth_data, kp_str))


            except BreakException:
                break
            except Empty:
                pass
            finally:
                wait_for_next_cycle(t_start)

    def __preprocess_image(self, tool_tcp: np.ndarray, img: np.ndarray):
        tcp = CAMERA_TCP @ tool_tcp
        R = tcp[:3, :3]
        theta = np.arctan2(R[2, 0], R[1, 0])
        if theta > 0:
            img = img.transpose(1, 0, 2)[::-1].copy()
        else:
            img = img.transpose(1, 0, 2)[:, ::-1].copy()

        img = cv2.cvtColor(img[..., ::-1], cv2.COLOR_BGR2GRAY)

        return img

    def __preprocess_depth_map(self, tool_tcp: np.ndarray, dep: np.ndarray):
        tcp = CAMERA_TCP @ tool_tcp
        R = tcp[:3, :3]
        theta = np.arctan2(R[2, 0], R[1, 0])
        if theta > 0:
            dep = dep.transpose(1, 0)[::-1].copy()
        else:
            dep = dep.transpose(1, 0)[:, ::-1].copy()

        dep[dep > 800] = 0.

        dep = dep / 1000

        return dep

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
