import json
import pickle
import time
from multiprocessing import Queue, Process
from queue import Empty
from typing import List, Dict, Optional

import PIL
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from corner_grasp.grasp_config import CAMERA_TCP, PRINCIPAL_POINT_X, \
    PRINCIPAL_POINT_Y, FOCAL_LENGTH_Y, FOCAL_LENGTH_X, CAMERA_OFFSET_NOISE, \
    MAHALANOBIS_THRESHOLD, MIN_DISTANCE_THRESHOLD
from utils.exceptions import BreakException
from utils.tools import pyout, UGENT
from utils.utils import wait_for_next_cycle, deserialize_ndarray, \
    serialize_ndarray


class SLAMSlave:
    ORIENTATION_STD = 0.5

    def __init__(self, in_queue: Optional[Queue] = None, render: bool = True):
        self.in_queue = Queue() if in_queue is None else in_queue
        self.ou_queue = Queue()
        self.sys_queue = Queue()
        self.render = render

        self.process = Process(
            target=self.run,
            args=(self.in_queue, self.ou_queue, self.sys_queue))
        self.process.start()

    def __unpack_data(self, tcp_str, img_data, kp_str):
        tcp = pickle.loads(tcp_str)
        img = deserialize_ndarray(*img_data)
        kp = json.loads(kp_str)

        return tcp, img, kp

    def process_measurement(self, tcp, img, keypoints):
        N = len(keypoints)
        Sigma = np.stack((np.eye(2) * 10,) * N, axis=0)
        y = {'mu': keypoints, 'C': Sigma.tolist(), 'x-facing': ["None", ] * N}

        self.in_queue.put((pickle.dumps(tcp),
                           serialize_ndarray(np.array(img)),
                           None,
                           json.dumps(y)))
        return json.loads(self.ou_queue.get())

    def __add_trajectory(self, tcp, ax, color):
        ax.plot(tcp[:, 0, -1], tcp[:, 1, -1], 0., '-', color=color)
        ax.plot(tcp[:, 0, -1], tcp[:, 1, -1], tcp[:, 2, -1], 'o-',
                color=color)

        # Plot the direction vector
        last_tcp = tcp[-1]
        position = last_tcp[:3, 3]
        vector_length = 0.1  # Adjust as needed
        for ii, c in ((0, UGENT.RED), (1, UGENT.GREEN), (2, UGENT.BLUE)):
            direction = last_tcp[:3, ii]
            line_end = position + direction * vector_length
            ax.plot([position[0], line_end[0]],
                    [position[1], line_end[1]],
                    [position[2], line_end[2]], '-', color=c)

    def __plot_TCPs(self, TCPs: List[np.ndarray],
                    mu: np.ndarray,
                    nr: np.ndarray):
        TCPs_array = np.array(TCPs)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim([-0.65, 0.1])
        ax.set_ylim([0., 0.75])
        ax.set_zlim([0., 0.55])
        ax.set_title("TCP")
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        if nr.size > 2:
            count_threshold = np.sort(nr)[-2]
        else:
            count_threshold = 0

        for x, count in zip(mu.reshape((-1, 6)), nr):
            if count >= count_threshold:
                ax.plot([x[0], x[0], x[0] + x[3]],
                        [x[1], x[1], x[1] + x[4]],
                        [x[2], 0., 0.],
                        'o-', color=UGENT.GREEN)

        self.__add_trajectory(TCPs_array, ax, UGENT.BLUE)

        # Convert Matplotlib figure to an OpenCV image
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8'
                              ).reshape(int(height), int(width), 3)[..., ::-1]

        # Display using OpenCV
        cv2.imshow("Trajectory", image)
        cv2.waitKey(50)

        # Close the figure to free up memory
        plt.close(fig)

    def tcp2ccp(self, tcp):
        ccp = tcp @ CAMERA_TCP

        direction = ccp[:3, 0]
        if ccp[2, 0] > 0:
            ccp = ccp @ np.array([[0., -1., 0., 0.],
                                  [1., 0., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
        else:
            ccp = ccp @ np.array([[0., 1., 0., 0.],
                                  [-1., 0., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])

        return ccp

    def compute_expected_measurement(self, P, ccp, return_3d=False):
        P = P.reshape((-1, 6, 1))
        C = ccp[:3, 3]
        R = np.linalg.inv(ccp[:3, :3])

        # Transform the 3D point from world coordinates to camera coordinates
        P_cam = R[None, ...] @ (P[:, :3, :] - C[None, :, None])

        # X and Y were flipped on the frame to get it upright, hence
        # focal length and ppy are associated with the x-axis of the camera
        # tcp
        u = FOCAL_LENGTH_Y * P_cam[:, 0] / P_cam[:, 2] + PRINCIPAL_POINT_Y
        v = FOCAL_LENGTH_X * P_cam[:, 1] / P_cam[:, 2] + PRINCIPAL_POINT_X

        y = np.concatenate((u, v), axis=1).reshape(-1, 1)
        if return_3d:
            return y, P_cam
        else:
            return y

    def compute_measurement_jacobian(self, P, ccp):
        P = P.reshape((-1, 6, 1))
        n = P.shape[0]
        C = ccp[:3, 3]
        R = np.linalg.inv(ccp[:3, :3])

        P_cam = R[None, ...] @ (P[:, :3, :] - C[None, :, None])

        # Compute the derivative of u with respect to the world coordinates
        c_u = FOCAL_LENGTH_Y * (
                R[0][None, :] / P_cam[:, 2] -
                R[2][None, :] * P_cam[:, 0] / P_cam[:, 2] ** 2)
        c_v = FOCAL_LENGTH_X * (
                R[1][None, :] / P_cam[:, 2] -
                R[2][None, :] * P_cam[:, 1] / P_cam[:, 2] ** 2)

        J = np.zeros((n * 2, n * 6))
        for ii in range(n):
            J[ii * 2, ii * 6:ii * 6 + 3] = c_u[ii]
            J[ii * 2 + 1, ii * 6:ii * 6 + 3] = c_v[ii]

        return J

    def calculate_y_bar_and_C(self, mu, ccp):
        y_bar = self.compute_expected_measurement(mu, ccp)
        C = self.compute_measurement_jacobian(mu, ccp)
        return y_bar, C

    def make_Q_matrix(self, estimation_noise):
        nr_of_measurements = estimation_noise.shape[0]
        Q = np.concatenate([q for q in estimation_noise], axis=0)
        Q = np.concatenate(
            (Q, np.zeros((Q.shape[0], (nr_of_measurements - 1) * 2))), axis=1)
        for ii in range(1, nr_of_measurements):
            Q[ii * 2:] = np.roll(Q[ii * 2:], 2, axis=1)

        Q += np.eye(Q.shape[0]) * CAMERA_OFFSET_NOISE ** 2
        if Q.shape[0] == 4:
            Q[:2, 2:] = np.eye(2) * CAMERA_OFFSET_NOISE ** 2
            Q[2:, :2] = np.eye(2) * CAMERA_OFFSET_NOISE ** 2

        return Q

    def kalman_update_formula(self, mu, Sigma, y, y_bar, C, Q):
        # Calculate the Kalman Gain
        K = Sigma @ C.T @ np.linalg.inv(C @ Sigma @ C.T + Q)

        # Update the state estimate
        mu_ = mu + K @ (y - y_bar)

        # Update the state covariance
        Sigma_ = (np.eye(mu.size) - K @ C) @ Sigma

        return mu_, Sigma_

    def fuse(self, mu, Sigma, *args):
        ii_from, ii_to, jj_from, jj_to = args
        y_size = ii_to - ii_from

        y = np.zeros(y_size)[:, None]
        y_bar = mu[ii_from:ii_to] - mu[jj_from:jj_to]
        C = np.zeros((y_size, mu.size))
        C[:, ii_from:ii_to] = np.eye(y_size)
        C[:, jj_from:jj_to] = -np.eye(y_size)
        Q = np.eye(y_size) * 1e-6
        return self.kalman_update_formula(mu, Sigma, y, y_bar, C, Q)

    def mahalanobis_distance(self, mu1, Sigma1, mu2):
        # We're only interested in positions for distance
        mu1_xyz = mu1.reshape(-1, 6)[:, :3].reshape(-1, 1)
        mu2_xyz = mu2.reshape(-1, 6)[:, :3].reshape(-1, 1)
        mu_diff = mu1_xyz - mu2_xyz

        Sigma_ = Sigma1.reshape(-1, 6)[:, :3]
        Sigma_ = Sigma_.reshape(mu1.shape[0], mu1_xyz.shape[0])
        Sigma_ = Sigma_.T.reshape(-1, 6)[:, :3]
        Sigma_ = Sigma_.reshape(mu1_xyz.shape[0], mu1_xyz.shape[0]).T

        D = (mu_diff.T @ np.linalg.inv(Sigma_) @ mu_diff) ** .5
        return float(D)

    def check_point_infront_of_cameras(self, x, CCP):
        for ccp in CCP:
            C = ccp[:3, 3][:, None]
            R = np.linalg.inv(ccp[:3, :3])
            P = R @ (x - C)

            if P[-1] < MIN_DISTANCE_THRESHOLD:
                return False
        return True

    def delete(self, mu, Sigma, jj_from, jj_to):
        mu = np.concatenate((mu[:jj_from], mu[jj_to:]))
        Sigma = np.concatenate((Sigma[:jj_from], Sigma[jj_to:]), axis=0)
        Sigma = np.concatenate((Sigma[:, :jj_from], Sigma[:, jj_to:]), axis=1)

        return mu, Sigma

    def sensor_fusion(self, mu, Sigma, mu_new, Sigma_new, CCP, nr):
        n = mu_new.size // 6
        nr = np.concatenate((nr, np.ones(n, dtype=int)))
        mu = np.concatenate((mu, mu_new))
        Sigma_tmp = np.zeros((Sigma.shape[0] + Sigma_new.shape[0],) * 2)
        Sigma_tmp[:Sigma.shape[0], :Sigma.shape[1]] = Sigma
        Sigma_tmp[-Sigma_new.shape[0]:, -Sigma_new.shape[1]:] = Sigma_new
        Sigma = np.copy(Sigma_tmp)
        del Sigma_tmp

        modified_flag = True
        while modified_flag:
            modified_flag = False
            for ii in range(mu.shape[0] // 6):
                if modified_flag:
                    break
                for jj in range(mu.shape[0] // 6):
                    if modified_flag:
                        break
                    if ii == jj:
                        continue

                    # check hypothesis of merging these keypoints
                    ii_from, ii_to = ii * 6, (ii + 1) * 6
                    jj_from, jj_to = jj * 6, (jj + 1) * 6

                    mu_new, Sigma_new = self.fuse(
                        mu, Sigma, ii_from, ii_to, jj_from, jj_to)

                    # Check Mahalanobis Distance
                    distance = self.mahalanobis_distance(mu, Sigma, mu_new)
                    if distance < MAHALANOBIS_THRESHOLD:
                        # Check that we are not merging outside bounding box
                        if self.check_point_infront_of_cameras(
                                mu_new[ii_from:ii_to - 3], CCP):
                            mu, Sigma = self.delete(
                                mu_new, Sigma_new, jj_from, jj_to)
                            nr[ii] += 1
                            nr = np.concatenate((nr[:jj], nr[jj + 1:]))
                            modified_flag = True
        return mu, Sigma, nr

    def measurement_update(self,
                           mu: np.ndarray,
                           Sigma: np.ndarray,
                           measurement: Dict[str, List[float]],
                           ccp: np.ndarray,
                           CCP: np.ndarray,
                           nr: np.ndarray,
                           n_iterations: int = 2):
        CCP = np.concatenate((CCP, ccp[None, :]), axis=0)
        n = len(measurement['mu'])
        if n == 0:
            return mu, Sigma, CCP, nr
        y = np.array(measurement['mu']).reshape(-1, 1)
        Q = self.make_Q_matrix(np.array(measurement['C']))

        z_axis = ccp[:3, 2] / np.linalg.norm(ccp[:3, 2])
        x_axis = ccp[:3, 0] / np.linalg.norm(ccp[:3, 0])
        mu_ = np.empty((0, 1))
        Sigma_ = np.empty((0,))
        for x_aligned in measurement['x-facing']:
            if x_aligned == "True":
                corner_axis = (x_axis * 0.1)[:, None]
                uncertainty = np.full_like(
                    x_axis, self.ORIENTATION_STD ** 2)
            elif x_aligned == "False":
                corner_axis = (-x_axis * 0.1)[:, None]
                uncertainty = np.full_like(
                    x_axis, self.ORIENTATION_STD ** 2)
            elif x_aligned == "None":
                corner_axis = (np.zeros_like(x_axis))[:, None]
                uncertainty = np.full_like(x_axis, 10 ** 2)
            else:
                raise ValueError(f"Alignment {x_aligned} unknown.")

            mu_ = np.concatenate((mu_,
                                  ccp[:3, 3][:, None] + .5 * z_axis[:, None],
                                  corner_axis))
            Sigma_ = np.concatenate(
                (Sigma_, np.full((3,), 10 ** 2), uncertainty))
        Sigma_ = np.diag(Sigma_)

        for _ in range(n_iterations):
            for ii in range(n):
                Sigma_[(6 * ii):(6 * ii) + 3, (6 * ii):(6 * ii) + 3] = \
                    np.eye(3) * 10 ** 2
            y_bar, C = self.calculate_y_bar_and_C(mu_, ccp)
            mu_, Sigma_ = self.kalman_update_formula(
                mu_, Sigma_, y, y_bar, C, Q)

        mu, Sigma, nr = self.sensor_fusion(mu, Sigma, mu_, Sigma_, CCP, nr)
        return mu, Sigma, CCP, nr

    def generate_output(self, mu, nr):
        pts = mu.reshape(-1, 6)
        II = np.argsort(nr)[::-1]

        kp = []
        for ii, idx in enumerate(II):
            if ii == 2:
                break

            kp.append({'X': pts[idx, :3].tolist(),
                       'theta': pts[idx, 3:].tolist(),
                       'count': int(nr[idx])})

        return kp

    def run(self, in_queue: Queue, ou_queue: Queue, sys_queue: Queue):
        TCPs = []

        # Initialize the state estimate mu to zeros
        mu = np.zeros((0, 1), dtype=float)
        nr = np.empty((0,), dtype=int)

        # Initialize the state covariance matrix Sigma to a large value
        # (indicating high uncertainty)
        Sigma = np.eye(0, dtype=float)

        # Initialize the camera locations (required for fusion)
        CCP = np.empty((0, 4, 4))

        ii = 0
        while True:
            ii += 1
            t_start = time.time()
            try:
                self.__process_sys_command(sys_queue)
                tcp_str, img_data, depth_data, kp_str = \
                    in_queue.get(timeout=1)
                tcp, img, y = self.__unpack_data(tcp_str, img_data, kp_str)
                ccp = self.tcp2ccp(tcp)

                mu, Sigma, CCP, nr = self.measurement_update(
                    mu, Sigma, y, ccp, CCP, nr)

                kpts = self.generate_output(mu, nr)
                ou_queue.put(json.dumps(kpts))

                TCPs.append(ccp)
                if ii % 10 == 0 and self.render:
                    self.__plot_TCPs(TCPs, mu, nr)



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
