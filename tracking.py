import json
import os
import shutil
import sys

import cv2
import numpy as np
import pyautogui

from utils.tools import pyout, pbar, makedirs

DATASET = "/home/matt/Datasets/real/020"

if not os.path.exists(f"{DATASET}/images"):
    files = os.listdir(DATASET)
    makedirs(f"{DATASET}/images")
    for file in files:
        shutil.move(f"{DATASET}/{file}", f"{DATASET}/images/{file}")


def load_img(fname):
    img = cv2.imread(f"{DATASET}/images/{fname}")
    img = cv2.resize(img, (480, 640))
    return img


def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)


pyout()


def show(img):
    img = cv2.resize(np.copy(img), img.shape[:2][::-1])
    cv2.imshow("Frame", img)
    return cv2.waitKey(-1)


def save(files, points):
    D = {}
    for file, pts in zip(files, points):
        if pts is not None:
            D[file] = pts

    with open(f"{DATASET}/opticalflow_annotations.json", "w+") as f:
        json.dump(D, f, indent=2)

    pyout("saved!")


def main():
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(
                         cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         10, 0.03))
    img_root = f"{DATASET}/images"
    fnames = [fname for fname in sorted(os.listdir(img_root))]
    frames = [load_img(fname) for fname in sorted(os.listdir(img_root))]
    f_gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    points = [None, ] * len(f_gray)
    ii = 1

    def mouse_event(event, x, y, flags, params):
        nonlocal points, ii
        if event == cv2.EVENT_LBUTTONDOWN:
            if points[ii] is None or len(points[ii]) == 0:
                points[ii] = [(x, y)]
            else:
                dist = lambda x1, x2: (
                        ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** .5)
                D = [dist((x, y), p) for p in points[ii]]
                argmin = min(range(len(D)), key=lambda jj: D[jj])
                pyout(D[argmin])
                if D[argmin] < 5.:
                    points[ii].pop(argmin)
                else:
                    points[ii].append((x, y))
                for jj in range(ii + 1, len(points)):
                    points[jj] = None
            pyautogui.press('enter')

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_event)

    while True:

        frame = frames[ii].copy()
        H, W, _ = frame.shape
        if points[ii] is None:
            if points[ii - 1] is not None and len(points[ii - 1]) > 0:
                P = np.array(points[ii - 1]).astype(np.float32)[:, None, :]

                new_pt, _, _ = cv2.calcOpticalFlowPyrLK(
                    f_gray[ii - 1], f_gray[ii], P, None, **lk_params)
                points[ii] = new_pt.reshape(-1, 2).tolist()
                points[ii] = [tuple(pt) for pt in points[ii]]

        if points[ii] is not None:
            for jj in list(range(len(points[ii])))[::-1]:
                point = points[ii][jj]
                if (0 <= point[0] <= W and 0 <= point[1] <= H):
                    pt = (int(round(point[0])), int(round(point[1])))
                    cv2.circle(frame, pt, 5, (0, 255, 0), -1)
                else:
                    points[ii].pop(jj)

        pyout(ii, points[ii])
        k = show(frame)
        if k == 27:
            break
        elif k == ord('d'):
            ii = min(ii + 1, len(frames) - 1)
        elif k == ord('a'):
            ii = max(1, ii - 1)
        elif k == ord('j'):
            save(fnames, points)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
