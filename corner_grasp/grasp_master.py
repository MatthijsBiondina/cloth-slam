import json
import pickle
import sys
import time
import warnings
from multiprocessing import Queue
from typing import Optional

import numpy as np
from airo_robots.grippers import Robotiq2F85
from airo_robots.manipulators.hardware.ur_rtde import URrtde

from corner_grasp.grasp_config import POSE_LEFT_SAFE, POSE_LEFT_PRESENT, \
    POSE_RIGHT_REST, EXPLORATION_TRAJECTORY, EXPLORATION_RECORD_FLAG, \
    POSE_RIGHT_GRAB_INIT, POSE_LEFT_MESS1, POSE_LEFT_MESS2, POSE_LEFT_MESS3, \
    POSE_RIGHT_GRAB_LEFT, POSE_RIGHT_GRAB_RIGHT, POSE_LEFT_REST
from corner_grasp.slave_alignment import TemporalAligner
from corner_grasp.slave_gaussian import GaussianMixtureModel
from corner_grasp.slave_keypoints import InferenceModel
from corner_grasp.slave_orientation import OrientationSlave
from corner_grasp.slave_realsense import RealsenseSlave
from slave_slam import SLAMSlave
from utils.tools import pyout
from utils.utils import serialize_ndarray, wait_for_next_cycle

np.set_printoptions(precision=3, suppress=True)
warnings.simplefilter('once', RuntimeWarning)


def angle_between_vectors(v1, v2):
    """
    Computes the direction-specific angle between two 2D vectors.

    Args:
    v1 (array-like): The first vector.
    v2 (array-like): The second vector.

    Returns:
    float: The angle in degrees from v1 to v2.
    """
    # Normalize the vectors
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)

    # Angle in radians
    angle_rad = np.arctan2(v2_u[1], v2_u[0]) - np.arctan2(v1_u[1], v1_u[0])

    angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi

    return angle_rad


class RobotMaster:
    CONTROL_LOOP_FREQUENCY = 60
    MAX_QUEUE_SIZE = 1000

    def __init__(self,
                 arm_right: URrtde,
                 arm_left: URrtde,
                 trial_name: Optional[str] = None,
                 ):

        self.arm_right = arm_right
        self.arm_left = arm_left

        # self.go_to_safe_pose()

        self.camera = RealsenseSlave()
        self.tcp_queue = Queue()
        self.aligner = TemporalAligner(
            self.tcp_queue, self.camera.image_queue, trial_name)
        self.neural_network = InferenceModel(self.aligner.pair_queue)
        self.gaussian_mixture = GaussianMixtureModel(
            self.neural_network.out_queue)
        self.orientation = OrientationSlave(self.gaussian_mixture.ou_queue)
        self.slam = SLAMSlave(self.orientation.ou_queue)
        self.corners = self.slam.ou_queue

        self.workers = {"camera": self.camera,
                        "aligner": self.aligner,
                        "nn": self.neural_network,
                        "gmm": self.gaussian_mixture,
                        "orientation": self.orientation,
                        "slam": self.slam}

    def go_to_safe_pose(self):
        self.__move_arm_to_joint_pose(
            self.arm_left, POSE_LEFT_PRESENT[::-1], tcp=False)

        self.arm_left.move_to_joint_configuration(POSE_LEFT_SAFE).wait()
        self.arm_right.move_to_joint_configuration(POSE_RIGHT_REST).wait()

    def go_to_rest_pose(self):
        self.arm_left.move_to_joint_configuration(POSE_LEFT_REST).wait()
        self.arm_right.move_to_joint_configuration(POSE_RIGHT_REST).wait()

    def rearrange_towel(self, N=1):
        self.__move_arm_to_joint_pose(
            self.arm_right, POSE_RIGHT_REST, tcp=False)
        self.__move_arm_to_joint_pose(
            self.arm_left, POSE_LEFT_SAFE, tcp=False)
        for _ in range(N):
            for pose in (POSE_LEFT_MESS1, POSE_LEFT_MESS3,
                         POSE_LEFT_MESS3, POSE_LEFT_MESS1):
                pose[0] = np.random.uniform(-.5 * np.pi, .5 * np.pi)
                # pose[-1] = np.random.uniform(0. * np.pi, 1. * np.pi)
                self.__move_arm_to_joint_pose(self.arm_left, pose, tcp=False)
        self.__move_arm_to_joint_pose(
            self.arm_left, POSE_LEFT_SAFE, tcp=False)

    def scan_towel(self):
        self.__move_arm_to_joint_pose(self.arm_left, POSE_LEFT_REST)
        self.__move_arm_to_joint_pose(self.arm_right, POSE_RIGHT_REST)
        # self.arm_left.gripper.open()
        # sys.exit(0)
        self.arm_left.default_leading_axis_joint_acceleration = 0.012
        self.__move_arm_to_joint_pose(self.arm_left, POSE_LEFT_PRESENT,
                                      joint_speed=0.1)
        self.arm_left.default_leading_axis_joint_acceleration = 1.2
        self.__move_arm_to_joint_pose(self.arm_right, POSE_RIGHT_REST)

        for pose, record_segment in (
                zip(EXPLORATION_TRAJECTORY[:5], EXPLORATION_RECORD_FLAG[:5])):
            self.__move_arm_to_joint_pose(
                self.arm_right, pose, record=record_segment)

        self.arm_right.move_to_joint_configuration(POSE_RIGHT_REST).wait()
        # self.arm_left.move_to_joint_configuration(POSE_LEFT_REST).wait()
        # self.arm_left.gripper.open()

        # self.__move_arm_to_joint_pose(self.arm_right, POSE_RIGHT_GRAB_INIT)

        while not self.corners.empty():
            kpts = json.loads(self.corners.get())

        return kpts

    def grasp(self, corners):

        corner = min(corners, key=lambda c: np.sum(np.array(c['X']) ** 2))
        X_corner = np.array(corner['X'])
        theta_corner = np.array([max(0, corner['theta'][0]),
                                 min(0, corner['theta'][1]),
                                 0])
        theta_norm = np.linalg.norm(theta_corner)
        if np.isclose(theta_norm, 0., 1e-6):
            return False
        theta_corner = theta_corner / theta_norm
        tcp_approach = np.eye(4)
        tcp_approach[:3, 2] = -1 * theta_corner
        tcp_approach[:3, 1] = np.array([0., 0., -1.])
        tcp_approach[:3, 0] = np.cross(tcp_approach[:3, 1],
                                       tcp_approach[:3, 2])
        tcp_approach[:3, 3] = X_corner + 0.1 * theta_corner

        tcp_open = tcp_approach.copy()
        tcp_open[:3, 3] = X_corner + 0.02 * theta_corner
        tcp_grab = tcp_approach.copy()
        tcp_grab[:3, 3] = X_corner - 0.03 * theta_corner

        self.arm_right.gripper.close()
        if angle_between_vectors(tcp_approach[:2, 2], X_corner[:2]) > 0:
            self.__move_arm_to_joint_pose(self.arm_right,
                                          POSE_RIGHT_GRAB_LEFT, tcp=False)
            self.arm_right.move_to_tcp_pose(tcp_pose=tcp_approach).wait()
        else:
            self.__move_arm_to_joint_pose(self.arm_right,
                                          POSE_RIGHT_GRAB_RIGHT, tcp=False)
            self.arm_right.move_to_tcp_pose(tcp_pose=tcp_approach).wait()
        self.arm_right.move_to_tcp_pose(tcp_pose=tcp_open).wait()

        self.arm_right.gripper.move(0.05).wait()
        self.arm_right.move_to_tcp_pose(tcp_pose=tcp_grab).wait()
        self.arm_right.gripper.close().wait()
        self.arm_right.move_to_tcp_pose(tcp_pose=tcp_approach).wait()

    def __move_arm_to_joint_pose(self,
                                 arm: URrtde,
                                 pose_array: np.ndarray,
                                 joint_speed: Optional[float] = None,
                                 record: bool = False,
                                 tcp: bool = True):
        loop_duration = 1 / self.CONTROL_LOOP_FREQUENCY

        if record:
            self.camera.start_recording()

        if len(pose_array.shape) == 1:
            pose_array = pose_array[None, :]

        for pose in pose_array:
            action = arm.move_to_joint_configuration(
                pose, joint_speed=joint_speed)

            while not action.is_action_done():
                start_time = time.time()

                if tcp:
                    # Any logic while waiting for the arm
                    if self.tcp_queue.qsize() < self.MAX_QUEUE_SIZE:
                        self.tcp_queue.put(
                            (time.time(), pickle.dumps(arm.get_tcp_pose())))

                wait_for_next_cycle(start_time, self.CONTROL_LOOP_FREQUENCY)

        if record:
            self.camera.stop_recording()

    def shutdown(self):
        for w_name, worker in self.workers.items():
            pyout(f"Shutting down {w_name}")
            worker.shutdown()
