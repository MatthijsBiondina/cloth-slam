import sys
import time
from typing import Optional

import cv2
import numpy as np
import pyrealsense2
from airo_robots.manipulators import URrtde

from corner_grasp.grasp_config import POSE_LEFT_PRESENT, POSE_RIGHT_REST
from utils.tools import pyout


def start_realsense():
    pipeline = pyrealsense2.pipeline()
    config = pyrealsense2.config()
    config.enable_stream(pyrealsense2.stream.color, *(640, 480),
                         pyrealsense2.format.bgr8, 10)
    profile = pipeline.start()

    return pipeline


def show(frame1, frame2, alpha=0.3):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1_gray = np.stack((frame1_gray,) * 3, axis=-1)

    frame = ((1 - alpha) * frame1_gray + alpha * frame2).astype(np.uint8)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(-1)
    return k


def main(ip_right: str = "10.42.0.162",
         ip_left: str = "10.42.0.163"):
    pipeline = start_realsense()

    Rarm = URrtde(ip_right, URrtde.UR3E_CONFIG)
    Larm = URrtde(ip_left, URrtde.UR3E_CONFIG)

    Larm.move_to_joint_configuration(POSE_LEFT_PRESENT).wait()

    pose1 = np.array([+0.350, -1.050, +0.550, -0.000, -1.000, +0.000]) * np.pi
    Rarm.move_to_joint_configuration(pose1).wait()
    frame1 = np.asanyarray(
        pipeline.wait_for_frames().get_color_frame().get_data()).astype(
        np.uint8)[..., ::-1].transpose(1, 0, 2)[::-1]

    pose2 = np.array([+0.350, -0.305, -0.793, -0.402, -1.000, +0.000])
    da = 0.005
    while True:

        pyout(pose2)
        Rarm.move_to_joint_configuration(pose2 * np.pi).wait()
        frame2 = np.asanyarray(
            pipeline.wait_for_frames().get_color_frame().get_data()).astype(
            np.uint8)[..., ::-1].transpose(1, 0, 2)[:, ::-1]

        k = show(frame1, frame2)
        if k == 27:  # esc
            break
        elif k == 119:  # w
            pose2[2] -= da
        elif k == 115:  # s
            pose2[2] += da
        elif k == 97:  # a
            pose2[1] -= da
        elif k == 100:  # d
            pose2[1] += da
        else:
            pyout(k)
        pose2[3] = -1.5 - pose2[1] - pose2[2]

    pyout(f"Final pose: {pose2}")


if __name__ == "__main__":
    main()
    # frame1 = cv2.imread("/home/matt/Pictures/frame1.jpg")
    # frame2 = cv2.imread("/home/matt/Pictures/frame2.jpg")
    # pyout(show(frame1, frame2))
