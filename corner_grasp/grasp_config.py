import numpy as np

POSE_LEFT_SAFE = np.array([0, 0, -0.5, -.75, .5, 0]) * np.pi
POSE_LEFT_REST = np.array([0., -.25, -0.5, -.75, .5, 0]) * np.pi
# POSE_LEFT_PRESENT = np.array([[0.5, -.25, -0.5, -.75, .5, 0],
#                               # [0.7, - 0.67, 0, -0.5, 0, 0],
#                               [0.5, - 0.75, 0, -0.5, 0, 0]]) * np.pi
POSE_LEFT_PRESENT = np.array([0.5, - 0.75, 0, -0.5, 0, 0]) * np.pi
POSE_LEFT_DROP = np.array([0.5, - 0.5, 0, -0.5, 0, 0]) * np.pi

POSE_LEFT_MESS1 = np.array([0, -.5, -0., -.75, .5, .5]) * np.pi
POSE_LEFT_MESS2 = np.array([0, -.33, -0.67, -.5, .5, .5]) * np.pi
POSE_LEFT_MESS3 = np.array([0, -.5, -0.72, -.28, .5, .5]) * np.pi

POSE_RIGHT_REST = np.array([0., -.75, 0.5, -.25, -1., 0.]) * np.pi
POSE_RIGHT_GRAB_INIT = np.array([-0.75, -.5, 0., -.5, 0., 0.]) * np.pi
POSE_RIGHT_GRAB_LEFT = np.array([-1., -.25, -0.75, 0., 0.5, 0.]) * np.pi
# POSE_RIGHT_GRAB_RIGHT = np.array([-.5, -.25, -0.75, -1., 0., 0.]) * np.pi
POSE_RIGHT_GRAB_RIGHT = np.array([-.5, -.75, 0.75, -1., -0.5, 0.]) * np.pi

CAMERA_TCP = np.array(
    [[0.99972987, -0.01355228, -0.01888183, -0.03158717],
     [0.01346261, 0.99989753, -0.00486777, -0.05201502],
     [0.01894587, 0.00461225, 0.99980987, -0.13887213],
     [0., 0., 0., 1.]])

EXPLORATION_TRAJECTORY = \
    np.array([[+0.100, -0.721, +0.670, -0.449, -1.000, +0.000],
              [+0.000, -0.330, +0.700, -0.870, -1.000, +0.000],
              [+0.070, -0.400, +0.850, -0.950, -1.000, +0.000],
              [+0.180, -1.050, +0.850, -0.300, -1.000, +0.000],
              [+0.350, -1.050, +0.550, -0.000, -1.000, +0.000],
              [+0.350, -0.305, -0.793, -0.402, -1.000, +0.000],
              [+0.490, -0.730, -0.600, -0.170, -1.000, +0.000],
              [+0.370, -0.600, -0.850, -0.050, -1.000, +0.000],
              [+0.250, +0.050, -0.820, -0.730, -1.000, +0.000],
              [+0.100, +0.050, -0.550, -1.000, -1.000, +0.000]]) * np.pi
EXPLORATION_RECORD_FLAG = \
    (False, True, True, True, True, False, True, True, True, True)

# PRETRAINED_MODEL_PATH = "/home/matt/Models/neat-disco.ckpt"
PRETRAINED_MODEL_PATH = "/home/matt/Models/genial-water.ckpt"
# PRETRAINED_MODEL_PATH = "/home/matt/Models/honest-yogurt.ckpt"

# Realsense2 Intrinsics
FOCAL_LENGTH_X = 613.616
FOCAL_LENGTH_Y = 611.588
PRINCIPAL_POINT_X = 319.943
PRINCIPAL_POINT_Y = 239.386

CAMERA_OFFSET_NOISE = 20  # in pixels
MAHALANOBIS_THRESHOLD = 1.
MIN_DISTANCE_THRESHOLD = .1
