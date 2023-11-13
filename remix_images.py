import os
import shutil
import sys
import time

import pyautogui
from PIL import Image

from utils.tools import pyout, pbar, makedirs

for _ in range(10):
    pyautogui.hotkey('ctrl', 'alt', 'left')
    time.sleep(0.1)

in_dir = "/media/matt/Expansion/Datasets/cloth/simulation_set/images"
ou_dir = "/media/matt/Expansion/Datasets/cloth/processed_set/images"
invoke_dir = "/home/matt/invokeai/outputs/images"

makedirs(ou_dir)


def clicks(locs, sleep=1.):
    for L in locs:
        pyautogui.click(*L)
        time.sleep(sleep)


def count_outs():
    return len(os.listdir("/home/matt/invokeai/outputs/images"))


def cleanup():
    # clear outputs
    for img_file in os.listdir(invoke_dir):
        if img_file.endswith(".png"):
            os.remove(f"{invoke_dir}/{img_file}")
    # clear Pictures
    for file in os.listdir("/home/matt/Pictures"):
        os.remove(f"/home/matt/Pictures/{file}")
    # clear Downloads
    for file in os.listdir("/home/matt/Downloads"):
        os.remove(f"/home/matt/Downloads/{file}")


for fname in pbar(sorted(os.listdir(in_dir))):
    cleanup()

    shutil.copyfile(f"{in_dir}/{fname}", f"/home/matt/Pictures/{fname}")
    clicks([(950, 90), (600, 650), (750, 490), (1335, 375)])
    time.sleep(3)
    clicks([(200, 90)])

    time.sleep(60)
    clicks([(1100, 90), (1100, 180), (1100, 180), (1460, 90)])
    time.sleep(1)

    for file in os.listdir("/home/matt/Downloads"):
        shutil.copyfile(f"/home/matt/Downloads/{file}",
                        f"{ou_dir}/{fname.replace('.png', '.jpg')}")

for fname in pbar(sorted(os.listdir(ou_dir))):
    img = Image.open(f"{ou_dir}/{fname}")
    img = img.resize((1024, 1024))
    img.save(f"{ou_dir}/{fname}")
