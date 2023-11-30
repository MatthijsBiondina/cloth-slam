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

from corner_grasp.grasp_config import CAMERA_TCP
from corner_grasp.grasp_master import RobotMaster
from utils.tools import pyout, pbar

np.set_printoptions(precision=3, suppress=True)

# multiprocessing.set_start_method('spawn')


robot_master = RobotMaster()

robot_master.go_to_rest_pose()
sys.exit(0)

robot_master.rearrange_towel(N=1)
corners = robot_master.scan_towel()
robot_master.grasp(corners)
