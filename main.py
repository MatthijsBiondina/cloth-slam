import atexit
import multiprocessing
import os
import sys
import time
from typing import List

import cv2
import pyrealsense2 as rs

import numpy as np
from airo_robots.grippers import Robotiq2F85
from airo_robots.manipulators import URrtde

from corner_grasp.grasp_config import CAMERA_TCP, POSE_LEFT_DROP, \
    POSE_LEFT_SAFE, POSE_LEFT_REST, POSE_RIGHT_REST
from corner_grasp.grasp_master import RobotMaster
from utils.tools import pyout, pbar

np.set_printoptions(precision=3, suppress=True)

# multiprocessing.set_start_method('spawn')

# robot_master.go_to_rest_pose()
# sys.exit(0)

TOWEL = 9

ip_right: str = "10.42.0.162"
ip_left: str = "10.42.0.163"

arm_right = URrtde(ip_right, URrtde.UR3E_CONFIG)
arm_right.gripper = Robotiq2F85(ip_right)
arm_right.gripper.open()
arm_left = URrtde(ip_left, URrtde.UR3E_CONFIG)
arm_left.gripper = Robotiq2F85(ip_left)

for ii in range(10):
    trial_name = f"towel_{str(TOWEL).zfill(2)}_trial_{ii}"


    robot_master = RobotMaster(arm_right, arm_left, trial_name)

    # arm_left.move_to_joint_configuration(POSE_LEFT_DROP).wait()
    # arm_left.gripper.open()
    # arm_left.move_to_joint_configuration(POSE_LEFT_SAFE).wait()
    time.sleep(10)
    # robot_master.rearrange_towel(N=1)
    corners = robot_master.scan_towel()
    arm_left.move_to_joint_configuration(POSE_LEFT_REST).wait()
    arm_left.gripper.open()
    arm_right.move_to_joint_configuration(POSE_RIGHT_REST).wait()


    # robot_master.grasp(corners)
    # time.sleep(5)
    # robot_master.arm_right.gripper.open()
    robot_master.shutdown()
