import sys
import time
import warnings
from typing import Optional

import numpy as np
from airo_robots.manipulators.hardware.ur_rtde import URrtde

from corner_grasp.grasp_config import POSE_LEFT_REST, POSE_LEFT_PRESENT, \
    POSE_RIGHT_REST, EXPLORATION_TRAJECTORY, EXPLORATION_RECORD_FLAG
from corner_grasp.slave_realsense import RealsenseSlave
from utils.tools import pyout

np.set_printoptions(precision=3, suppress=True)
warnings.simplefilter('once', RuntimeWarning)


class RobotMaster:
    CONTROL_LOOP_FREQUENCY = 60

    def __init__(self,
                 ip_right: str = "10.42.0.162",
                 ip_left: str = "10.42.0.163"):

        self.camera = RealsenseSlave()

        self.arm_right = URrtde(ip_right, URrtde.UR3E_CONFIG)
        self.arm_left = URrtde(ip_left, URrtde.UR3E_CONFIG)

    def scan_towel(self):
        self.__move_arm_to_joint_pose(self.arm_left, POSE_LEFT_REST)
        # self.__move_arm_to_joint_pose(self.arm_left, POSE_LEFT_PRESENT)
        self.camera.shutdown()
        sys.exit(0)

        self.__move_arm_to_joint_pose(self.arm_right, POSE_RIGHT_REST)

        for pose, record_segment in (
                zip(EXPLORATION_TRAJECTORY, EXPLORATION_RECORD_FLAG)):
            self.__move_arm_to_joint_pose(
                self.arm_right, pose, record=record_segment)

        self.arm_left.move_to_joint_configuration(POSE_RIGHT_REST)

        self.camera.shutdown()

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

            self.__wait_for_next_cycle(start_time)

        if record:
            self.camera.stop_recording()

    def __wait_for_next_cycle(self, start_time: float):
        loop_duration = 1 / self.CONTROL_LOOP_FREQUENCY
        end_time = time.time()
        elapsed_time = end_time - start_time

        if elapsed_time < loop_duration:
            time.sleep(loop_duration - elapsed_time)
        else:
            warnings.warn(f"Control loop frequency dropped below desired "
                          f"{self.CONTROL_LOOP_FREQUENCY} Hz", RuntimeWarning)
