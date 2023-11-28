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

from corner_grasp.grasp_config import POSE_LEFT_REST, POSE_LEFT_PRESENT, \
    POSE_RIGHT_REST, EXPLORATION_TRAJECTORY, EXPLORATION_RECORD_FLAG, \
    POSE_RIGHT_GRAB
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


class RobotMaster:
    CONTROL_LOOP_FREQUENCY = 60
    MAX_QUEUE_SIZE = 1000

    def __init__(self,
                 ip_right: str = "10.42.0.162",
                 ip_left: str = "10.42.0.163"):

        self.arm_right = URrtde(ip_right, URrtde.UR3E_CONFIG)
        self.arm_left = URrtde(ip_left, URrtde.UR3E_CONFIG)
        self.arm_left.gripper = Robotiq2F85(ip_left)
        self.arm_left.default_linear_acceleration = 0.12

        A = self.arm_left.move_to_joint_configuration(POSE_LEFT_PRESENT)

        self.camera = RealsenseSlave()
        self.tcp_queue = Queue()
        self.aligner = TemporalAligner(
            self.tcp_queue, self.camera.image_queue)
        self.neural_network = InferenceModel(self.aligner.pair_queue)
        self.gaussian_mixture = GaussianMixtureModel(
            self.neural_network.out_queue)
        self.orientation = OrientationSlave(self.gaussian_mixture.ou_queue)
        self.slam = SLAMSlave(self.orientation.ou_queue)

        self.corners = self.slam.ou_queue

        A.wait()

    def scan_towel(self):
        self.__move_arm_to_joint_pose(self.arm_right, POSE_RIGHT_REST)
        # self.__move_arm_to_joint_pose(self.arm_left, POSE_LEFT_REST)
        # self.arm_left.gripper.open()
        # sys.exit(0)
        self.__move_arm_to_joint_pose(self.arm_left, POSE_LEFT_PRESENT)
        self.__move_arm_to_joint_pose(self.arm_right, POSE_RIGHT_REST)

        for pose, record_segment in (
                zip(EXPLORATION_TRAJECTORY[:5], EXPLORATION_RECORD_FLAG[:5])):
            self.__move_arm_to_joint_pose(
                self.arm_right, pose, record=record_segment)

        self.arm_right.move_to_joint_configuration(POSE_RIGHT_REST).wait()
        # self.arm_left.move_to_joint_configuration(POSE_LEFT_REST).wait()
        # self.arm_left.gripper.open()

        self.__move_arm_to_joint_pose(self.arm_right, POSE_RIGHT_GRAB)
        while self.corners.empty():
            time.sleep(1)
        while not self.corners.empty():
            kpts = json.loads(self.corners.get())
        pyout(kpts)


        self.camera.shutdown()
        self.aligner.shutdown()
        self.neural_network.shutdown()
        self.gaussian_mixture.shutdown()
        self.orientation.shutdown()
        self.slam.shutdown()

        time.sleep(3600)

    def __move_arm_to_joint_pose(self,
                                 arm: URrtde,
                                 pose: np.ndarray,
                                 joint_speed: Optional[float] = None,
                                 record: bool = False):
        loop_duration = 1 / self.CONTROL_LOOP_FREQUENCY

        if record:
            self.camera.start_recording()

        action = arm.move_to_joint_configuration(
            pose, joint_speed=joint_speed)

        while not action.is_action_done():
            start_time = time.time()

            # Any logic while waiting for the arm
            if self.tcp_queue.qsize() < self.MAX_QUEUE_SIZE:
                self.tcp_queue.put(
                    (time.time(), pickle.dumps(arm.get_tcp_pose())))

            wait_for_next_cycle(start_time, self.CONTROL_LOOP_FREQUENCY)

        if record:
            self.camera.stop_recording()
