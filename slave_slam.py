import json
import pickle
import time
from multiprocessing import Queue, Process
from queue import Empty
from typing import List, Dict

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from corner_grasp.grasp_config import CAMERA_TCP, PRINCIPAL_POINT_X, \
    PRINCIPAL_POINT_Y, FOCAL_LENGTH_Y, FOCAL_LENGTH_X, CAMERA_OFFSET_NOISE, \
    MAHALANOBIS_THRESHOLD, MIN_DISTANCE_THRESHOLD
from utils.exceptions import BreakException
from utils.tools import pyout, UGENT
from utils.utils import wait_for_next_cycle, deserialize_ndarray


class SLAMSlave:
    def __init__(self, in_queue: Queue):
        self.in_queue = in_queue
        self.ou_queue = Queue()
        self.sys_queue = Queue()

        self.process = Process(
            target=self.run,
            args=(self.in_queue, self.ou_queue, self.sys_queue))
        self.process.start()

    def __unpack_data(self, tcp_str, img_data, kp_str):
        tcp = pickle.loads(tcp_str)
        img = deserialize_ndarray(*img_data)
        kp = json.loads(kp_str)

        return tcp, img, kp

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

    def __plot_TCPs(self, TCPs: List[np.ndarray], mu: np.ndarray):
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

        for x in mu.reshape((-1, 3)):
            ax.plot([x[0], x[0]], [x[1], x[1]], [x[2], 0.],
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

    def __tcp2ccp(self, tcp):
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

    def compute_expected_measurement(self, P, ccp):
        P = P.reshape((-1, 3, 1))
        C = ccp[:3, 3]
        R = np.linalg.inv(ccp[:3, :3])

        # Transform the 3D point from world coordinates to camera coordinates
        P_cam = R[None, ...] @ (P - C[None, :, None])

        # X and Y were flipped on the frame to get it upright, hence
        # focal length and ppy are associated with the x-axis of the camera
        # tcp
        u = FOCAL_LENGTH_Y * P_cam[:, 0] / P_cam[:, 2] + PRINCIPAL_POINT_Y
        v = FOCAL_LENGTH_X * P_cam[:, 1] / P_cam[:, 2] + PRINCIPAL_POINT_X

        return np.concatenate((u, v), axis=1).reshape(-1, 1)

    def compute_measurement_jacobian(self, P, ccp):
        P = P.reshape((-1, 3, 1))
        J = ccp[:3, 3]
        R = np.linalg.inv(ccp[:3, :3])

        P_cam = R[None, ...] @ (P - J[None, :, None])

        # Compute the derivative of u with respect to the world coordinates
        c_u = FOCAL_LENGTH_Y * (
                R[0][None, :] / P_cam[:, 2] -
                R[2][None, :] * P_cam[:, 0] / P_cam[:, 2] ** 2)
        c_v = FOCAL_LENGTH_X * (
                R[1][None, :] / P_cam[:, 2] -
                R[2][None, :] * P_cam[:, 1] / P_cam[:, 2] ** 2)

        J = np.concatenate((c_u, c_v), axis=1).reshape(-1, 3)
        J = np.concatenate((J, np.zeros((J.shape[0],
                                         (J.shape[0] // 2 - 1) * 3))),
                           axis=1)
        for ii in range(J.shape[0] // 2 - 1):
            J[(ii + 1) * 2:] = np.roll(J[(ii + 1) * 2:], 3, axis=1)

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

        pyout()

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

        y = np.array([0., 0., 0.])[:, None]
        y_bar = mu[ii_from:ii_to] - mu[jj_from:jj_to]
        C = np.zeros((3, mu.size))
        C[:, ii_from:ii_to] = np.eye(3)
        C[:, jj_from:jj_to] = -np.eye(3)
        Q = np.eye(3) * 1e-6
        return self.kalman_update_formula(mu, Sigma, y, y_bar, C, Q)

    def mahalanobis_distance(self, mu1, Sigma1, mu2):
        D = ((mu1 - mu2).T @ np.linalg.inv(Sigma1) @ (mu1 - mu2)) ** .5
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
        nr = np.concatenate((nr, np.ones(mu.size // 2, dtype=int)))
        mu = np.concatenate((mu, mu_new))
        Sigma_tmp = np.zeros((Sigma.shape[0] + Sigma_new.shape[0],) * 2)
        Sigma_tmp[:Sigma.shape[0], :Sigma.shape[1]] = Sigma
        Sigma_tmp[-Sigma_new.shape[0]:, -Sigma_new.shape[1]:] = Sigma_new
        Sigma = np.copy(Sigma_tmp)
        del Sigma_tmp

        modified_flag = True
        while modified_flag:
            modified_flag = False
            for ii in range(mu.shape[0] // 3):
                if modified_flag:
                    break
                for jj in range(mu.shape[0] // 3):
                    if modified_flag:
                        break
                    if ii == jj:
                        continue

                    # check hypothesis of merging these keypoints
                    ii_from, ii_to = ii * 3, (ii + 1) * 3
                    jj_from, jj_to = jj * 3, (jj + 1) * 3

                    mu_new, Sigma_new = self.fuse(
                        mu, Sigma, ii_from, ii_to, jj_from, jj_to)

                    # Check Mahalanobis Distance
                    distance = self.mahalanobis_distance(mu, Sigma, mu_new)
                    if distance < MAHALANOBIS_THRESHOLD:
                        # Check that we are not merging outside bounding box
                        if self.check_point_infront_of_cameras(
                                mu_new[ii_from:ii_to], CCP):
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
        mu_ = np.concatenate((ccp[:3, 3] + .5 * z_axis,) * n, axis=0)[:, None]
        for _ in range(n_iterations):
            Sigma_ = np.eye(3 * n) * 10 ** 2
            y_bar, C = self.calculate_y_bar_and_C(mu_, ccp)
            mu_, Sigma_ = self.kalman_update_formula(
                mu_, Sigma_, y, y_bar, C, Q)

        mu, Sigma, nr = self.sensor_fusion(mu, Sigma, mu_, Sigma_, CCP, nr)
        return mu, Sigma, CCP, nr

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

        while True:
            t_start = time.time()
            try:
                self.__process_sys_command(sys_queue)
                tcp_str, img_data, kp_str = in_queue.get(timeout=1)
                tcp, img, y = self.__unpack_data(tcp_str, img_data, kp_str)
                ccp = self.__tcp2ccp(tcp)


                mu, Sigma, CCP, nr = self.measurement_update(
                    mu, Sigma, y, ccp, CCP, nr)

                TCPs.append(ccp)
                self.__plot_TCPs(TCPs, mu)

                # pyout()


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
