import atexit
import os
import sys
import time
from typing import List

import cv2
import pyrealsense2 as rs

import numpy as np

from utils.tools import pyout, pbar

np.set_printoptions(precision=3, suppress=True)

from airo_robots.manipulators.hardware.ur_rtde import URrtde

Rarm = URrtde("10.42.0.162", URrtde.UR3E_CONFIG)
Larm = URrtde("10.42.0.163", URrtde.UR3E_CONFIG)


# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
# profile = pipeline.start(config)
#
# sensor = profile.get_device().query_sensors()[1]  # Get the color sensor
# sensor.set_option(rs.option.exposure,
#                   166)  # Adjust exposure (value may need tweaking)
# sensor.set_option(rs.option.gain, 16)  # Adjust gain (value may need tweaking)


def move_to_joint_configuration_blocking(
        arm: URrtde, pose: np.array,
        eps: float = 0.001, slow=False,
        record=False):
    if slow:
        arm.move_to_joint_configuration(pose, joint_speed=0.05)
    else:
        arm.move_to_joint_configuration(pose)
    while not np.all(np.isclose(np.array(arm.get_joint_configuration()),
                                np.array(pose), atol=eps)):
        if record:
            fname = arm.get_tcp_pose().reshape(-1).tolist()
            fname = ' '.join([f"{v:.3f}" for v in fname])

            take_camera_frame(fname)
        else:
            time.sleep(0.01)
    time.sleep(0.1)


def get_xyz(tcp_pose):
    return tcp_pose[:-1, -1]


def move_arms():
    # Move left arm to rest
    poseL_rest = np.array([0, -.25, -0.5, -.75, .5, 0]) * np.pi
    poseL_present = [np.pi / 2, - 3 / 4 * np.pi, 0, -np.pi / 2, 0, 0]
    # move_to_joint_configuration_blocking(Larm, poseL_rest)
    move_to_joint_configuration_blocking(Larm, poseL_present)
    # move_to_joint_configuration_blocking(Larm, poseL_rest)
    # sys.exit(0)
    # Move right arm to rest
    poseR_rest = np.array([0., -.75, 0.5, -.25, -1., 0.]) * np.pi
    move_to_joint_configuration_blocking(Rarm, poseR_rest)

    posesR_scan = [
        # [0., -.33, 0.7, np.nan, -1., 0., True],
        # [0., -.33, 0.7, np.nan, -1.05, 0., True],
        # [0.04, -.36, 0.77, np.nan, -1., 0., True],
        # [0.07, -.4, 0.85, np.nan, -1., 0., True],
        # [0.13, -.75, 0.85, np.nan, -1., 0., True],
        # [0.18, -1.05, 0.85, np.nan, -1., 0., True],
        # [0.28, -1.05, 0.66, np.nan, -1., 0., True],
        [0.35, -1.05, 0.55, np.nan, -1., 0., True],
        # [0.35, -1.05, 0.55, np.nan, -1., 0., False],
        [0.35, -.373, -0.681, np.nan, -1., 0., False],
        # [0.48, -.66, -0.7, np.nan, -1., 0.],
        # [0.35, -.6, -0.85, np.nan, -1., 0.],
        # [0.35, -.33, -0.85, np.nan, -1., 0.],
    ]
    poses = []

    for ii, pose_tgt in enumerate(posesR_scan):
        # Rarm.mov
        condition = pose_tgt[-1]
        pose_tgt = np.array(pose_tgt[:-1]) * np.pi
        if condition:
            pose_tgt[3] = -0.5 * np.pi - np.sum(pose_tgt[1:3])
        else:
            pose_tgt[3] = -1.5 * np.pi - np.sum(pose_tgt[1:3])

        move_to_joint_configuration_blocking(Rarm, pose_tgt, record=False)
        poses.append(get_xyz(Rarm.get_tcp_pose()))

    # todo: We're goint to need this once we know the camera parameters
    # tgt = poses[0]
    # pose = np.array(posesR_scan[-1][:-1]) * np.pi
    # loc = get_xyz(Rarm.get_tcp_pose())
    #
    # dist = lambda: np.sum((tgt - get_xyz(Rarm.get_tcp_pose())) ** 2) ** .5
    # current_dist = dist()
    # current_pose = np.copy(pose)
    # dp = 0.01
    # while True:
    #     changed = False
    #     while True:
    #         pose[1] += dp
    #         pose[3] = -1.5 * np.pi - np.sum(pose[1:3])
    #         move_to_joint_configuration_blocking(Rarm, pose)
    #         if dist() < current_dist:
    #             changed = True
    #             current_dist = dist()
    #             current_pose = np.copy(pose)
    #         else:
    #             pose = np.copy(current_pose)
    #             move_to_joint_configuration_blocking(Rarm, pose)
    #             break
    #
    #     while True:
    #         pose[1] -= dp
    #         pose[3] = -1.5 * np.pi - np.sum(pose[1:3])
    #         move_to_joint_configuration_blocking(Rarm, pose)
    #         if dist() < current_dist:
    #             changed = True
    #             current_dist = dist()
    #             current_pose = np.copy(pose)
    #         else:
    #             pose = np.copy(current_pose)
    #             move_to_joint_configuration_blocking(Rarm, pose)
    #             break
    #
    #     while True:
    #         pose[2] += dp
    #         pose[3] = -1.5 * np.pi - np.sum(pose[1:3])
    #         move_to_joint_configuration_blocking(Rarm, pose)
    #         if dist() < current_dist:
    #             changed = True
    #             current_dist = dist()
    #             current_pose = np.copy(pose)
    #         else:
    #             pose = np.copy(current_pose)
    #             move_to_joint_configuration_blocking(Rarm, pose)
    #             break
    #
    #     while True:
    #         pose[2] -= dp
    #         pose[3] = -1.5 * np.pi - np.sum(pose[1:3])
    #         move_to_joint_configuration_blocking(Rarm, pose)
    #         if dist() < current_dist:
    #             changed = True
    #             current_dist = dist()
    #             current_pose = np.copy(pose)
    #         else:
    #             pose = np.copy(current_pose)
    #             move_to_joint_configuration_blocking(Rarm, pose)
    #             break
    #     if not changed:
    #         break
    #     pyout(f"{pose/np.pi} -> {current_dist:.3f}")

    pyout()
    # move_to_joint_configuration_blocking(Rarm, poseR_rest)


def take_camera_frame(fname):
    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        ii = len(os.listdir("/home/matt/Pictures"))
        cv2.imwrite(f"/home/matt/Pictures/{str(ii).zfill(4)}_{fname}.jpg",
                    color_image)
    except Exception as e:
        pass


if __name__ == "__main__":
    move_arms()
    # take_camera_frame()

    # pipeline.stop()
