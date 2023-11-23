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
    POSE_RIGHT_REST, EXPLORATION_TRAJECTORY, EXPLORATION_RECORD_FLAG
from corner_grasp.slave_alignment import TemporalAligner
from corner_grasp.slave_keypoints import InferenceModel
from corner_grasp.slave_realsense import RealsenseSlave
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
        self.keypoint_detector = InferenceModel(self.aligner.pair_queue)

        A.wait()

    def scan_towel(self):
        self.__move_arm_to_joint_pose(self.arm_right, POSE_RIGHT_REST)
        # self.__move_arm_to_joint_pose(self.arm_left, POSE_LEFT_REST)
        self.__move_arm_to_joint_pose(self.arm_left, POSE_LEFT_PRESENT,
                                      joint_speed=0.1)
        self.__move_arm_to_joint_pose(self.arm_right, POSE_RIGHT_REST)

        pyout(self.arm_right.get_tcp_pose())

        for pose, record_segment in (
                zip(EXPLORATION_TRAJECTORY[:], EXPLORATION_RECORD_FLAG[:])):
            self.__move_arm_to_joint_pose(
                self.arm_right, pose, record=record_segment)

        self.arm_right.move_to_joint_configuration(POSE_RIGHT_REST).wait()
        self.arm_left.move_to_joint_configuration(POSE_LEFT_REST).wait()
        self.arm_left.gripper.open()

        self.camera.shutdown()
        self.aligner.shutdown()

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
