import json
import os
import shutil
import sys

import cv2
import numpy as np
import pyautogui

from utils.tools import pyout, pbar, makedirs, listdir


def show(img):
    img = cv2.resize(np.copy(img), img.shape[:2][::-1])
    cv2.imshow("Frame", img)
    return cv2.waitKey(-1)


class KPLabeler:
    LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(
                         cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         10, 0.03))

    def __init__(self, root: str):
        self.root = root

    def __init_placeholders(self, trial):
        fnames = listdir(f"{trial}/images")
        frames = [cv2.imread(fname) for fname in fnames]
        f_gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        points = [None, ] * len(f_gray)

        return fnames, frames, f_gray, points

    def __process_mouse_event(self, event, x, y, points, ii):
        if event == cv2.EVENT_LBUTTONDOWN:
            if points[ii] is None or len(points[ii]) == 0:
                points[ii] = [(x, y)]
            else:
                dist = lambda x1, x2: (
                        ((x1[0] - x2[0]) ** 2 + (
                                x1[1] - x2[1]) ** 2) ** .5)
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
        return points, ii

    def __process_frame(self, frame, points, ii, f_gray):
        H, W, _ = frame.shape
        if points[ii] is None:
            if points[ii - 1] is not None and len(points[ii - 1]) > 0:
                P = np.array(points[ii - 1]).astype(np.float32)[:,
                    None, :]

                new_pt, _, _ = cv2.calcOpticalFlowPyrLK(
                    f_gray[ii - 1], f_gray[ii], P, None, **self.LK_PARAMS)
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

        return frame, points, ii

    def __save(self, trial, img_files, points):
        D = {}
        for file, pts in zip(img_files, points):
            if pts is not None:
                D[f"images/{file.split('/')[-1]}"] = pts

        with open(f"{trial}/annotations.json", "w+") as f:
            json.dump(D, f, indent=2)

        pyout("saved!")

    def run(self):
        for trial in listdir(self.root):
            if os.path.exists(f"{trial}/annotations.json"):
                continue

            self.__modify_trial_to_coco_format(trial)
            fnames, frames, f_gray, points = self.__init_placeholders(trial)
            ii = 1

            def mouse_event(event, x, y, flags, params):
                nonlocal points, ii
                points, ii = self.__process_mouse_event(
                    event, x, y, points, ii)

            cv2.namedWindow("Frame")
            cv2.setMouseCallback("Frame", mouse_event)

            while True:

                frame = frames[ii].copy()
                frame, points, ii = self.__process_frame(
                    frame, points, ii, f_gray)

                k = show(frame)
                if k == 27:
                    break
                elif k == ord('d'):
                    ii = min(ii + 1, len(frames) - 1)
                elif k == ord('a'):
                    ii = max(1, ii - 1)
                elif k == ord('j'):
                    self.__save(trial, fnames, points)

            cv2.destroyAllWindows()

    def __modify_trial_to_coco_format(self, path):
        if not os.path.exists(f"{path}/images"):
            files = listdir(path)
            makedirs(f"{path}/images")
            for in_file in files:
                ou_file = in_file.split("/")
                ou_file.insert(-1, "images")
                ou_file = "/".join(ou_file)
                shutil.move(in_file, ou_file)


# def select_point(event, x, y, flags, params):
#     global point, point_selected, old_points
#     if event == cv2.EVENT_LBUTTONDOWN:
#         point = (x, y)
#         point_selected = True
#         old_points = np.array([[x, y]], dtype=np.float32)


def save(files, points):
    D = {}
    for file, pts in zip(files, points):
        if pts is not None:
            D[file] = pts

    with open(f"{DATASET}/opticalflow_annotations.json", "w+") as f:
        json.dump(D, f, indent=2)

    pyout("saved!")


if __name__ == "__main__":
    main()
