import os
import random
import socket
import subprocess
import time
import traceback
from multiprocessing import current_process
import numpy as np
from tqdm import tqdm

bcolors = {'PINK': '\033[95m',
           'BLUE': '\033[94m',
           'CYAN': '\033[96m',
           'GREEN': '\033[92m',
           'YELLOW': '\033[93m',
           'RED': '\033[91m', }


class UGENT:
    BLUE = "#1E64C8"
    YELLOW = "#FFD200"
    WHITE = "#FFFFFF"
    BLACK = "#000000"
    ORANGE = "#F1A42B"
    RED = "#DC4E28"
    AQUA = "#2D8CA8"
    PINK = "#E85E71"
    SKY = "#8BBEE8"
    LIGHTGREEN = "#AEB050"
    PURPLE = "#825491"
    WARMORANGE = "#FB7E3A"
    TURQUOISE = "#27ABAD"
    LIGHTPURPLE = "#BE5190"
    GREEN = "#71A860"


def bash(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def set_seed(seed):
    """
    Set rng seed for all sources of randomness

    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)


def pretty_string(message: str, color=None, bold=False, underline=False):
    """
    add color and effects to string
    :param message:
    :param color:
    :param bold:
    :param underline:
    :return:
    """
    ou = message
    if color:
        ou = bcolors[color] + message + '\033[0m'
    if bold:
        ou = '\033[1m' + ou + '\033[0m'
    if underline:
        ou = '\033[4m' + ou + '\033[0m'
    return ou


def poem(string):
    if len(string) > 20:
        return string[:20] + '...'
    else:
        return string + ' ' * (23 - len(string))


def pyout(*message, color="PINK"):
    """
    Print message preceded by traceback. I use this method to prevent rogue
    "print" statements
    during debugging
    :param message:
    :return:
    """

    message = ' '.join(str(m) for m in message)

    trace = traceback.extract_stack()[-2]

    fname = trace.filename.replace(os.path.abspath(os.curdir), "...")

    trace = f"{fname}: {trace.name}(...) - ln{trace.lineno}"

    tqdm.write(pretty_string(trace, color, bold=True))
    if message is not None:
        tqdm.write(message)


def pyopen(path, mode):
    pyout(f"{mode} >> {os.path.abspath(path)}", color="BLUE")
    return open(path, mode)


pseudo_random_state = 49


def pysend(*message):
    message = ' '.join(str(m) for m in message)
    trace = traceback.extract_stack()[-2]

    fname = trace.filename.replace(os.path.abspath(os.curdir), "...")

    trace = f"{fname}: {trace.name}(...) - ln{trace.lineno}"

    subprocess.Popen(['notify-send', trace, message])


def prng(decimals=4):
    global pseudo_random_state

    ou = 0
    for ii in range(1, decimals + 1):
        pseudo_random_state = (7 * pseudo_random_state) % 101

        ou += (pseudo_random_state % 10) * 10 ** -ii
    ou = str(ou)[:decimals + 2]

    return float(ou)


time_0 = time.time()


def tic():
    global time_0
    time_0 = time.time()


def toc():
    global time_0
    pyout(f"{time.time() - time_0:.2f}")


def makedirs(path):
    pth = ""
    for folder in path.split("/"):
        pth += folder + "/"
        os.makedirs(pth, exist_ok=True)
    pyout(f"mk >> {os.path.abspath(path)}", color="BLUE")


def listdir(path: str):
    filenames = sorted(os.listdir(path))
    filepaths = [f"{path}/{fname}" for fname in filenames]
    filepaths = [os.path.abspath(path) for path in filepaths]
    return filepaths

def fname(path: str):
    return path.split("/")[-1]

def pbar(iterable, desc="", leave=False, total=None, disable=False):
    # return iterable
    host = socket.gethostname()

    if host in ("AM", "kat", "gorilla"):
        return tqdm(iterable, desc=poem(desc), leave=leave, total=total,
                    disable=(current_process().name != "MainProcess"))
    else:
        return iterable
