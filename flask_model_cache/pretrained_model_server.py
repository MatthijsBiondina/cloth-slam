import io
import json
import socket
import warnings
from io import BytesIO
from multiprocessing import Queue, Process
from typing import Tuple

import numpy as np
import torch.cuda
from PIL import Image

from flask_model_cache.pretrained_model import PretrainedModel
from flask_model_cache.server import Server
from utils.socket_utils import init_socket_server
from utils.tools import pyout

HOSTS = {'kat': "172.18.20.240", "gorilla": "172.18.21.117"}
SERVER_PORT = 5001

BOX = {"top": (28, 0, 1052, 1024),
       "mid": (28, 448, 1052, 1472),
       "bot": (28, 896, 1052, 1920)}


def preprocess_image(img: Image):
    if img.width == 1920:
        img = img.rotate(270, expand=True)

    top = img.crop(BOX['top'])
    mid = img.crop(BOX['mid'])
    bot = img.crop(BOX['bot'])

    return top, mid, bot


def postprocess_heatmaps(top: np.ndarray, mid: np.ndarray, bot: np.ndarray):
    heatmap = np.zeros((3, 1920, 1080))
    heatmap[0, BOX['top'][1]:BOX['top'][3], BOX['top'][0]:BOX['top'][2]] = top
    heatmap[1, BOX['mid'][1]:BOX['mid'][3], BOX['mid'][0]:BOX['mid'][2]] = mid
    heatmap[2, BOX['bot'][1]:BOX['bot'][3], BOX['bot'][0]:BOX['bot'][2]] = bot
    heatmap = np.max(heatmap, axis=0)

    heatmap = Image.fromarray(
        np.clip((heatmap * 255), 0., 255.).astype('uint8'))
    with io.BytesIO() as output:
        heatmap.save(output, format="JPEG")
        data = output.getvalue()
    return data


def run(checkpoint, gpu_idx, in_queue, ou_queue, ok_queue):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PretrainedModel(checkpoint, gpu=gpu_idx)
        pyout(f"GPU{gpu_idx} ready.")
    ok_queue.put(f"Worker {gpu_idx}: Ready!")

    while True:
        meta, data = in_queue.get()
        img = Image.open(BytesIO(data))
        top, mid, bot = preprocess_image(img)

        _, heat = model([top, mid, bot])

        heat_top = heat[0].cpu().numpy()
        heat_mid = heat[1].cpu().numpy()
        heat_bot = heat[2].cpu().numpy()

        heatmap = postprocess_heatmaps(heat_top, heat_mid, heat_bot)

        meta = {'filesize': len(heatmap), 'idx': meta['idx'], 'method': 'get'}
        ou_queue.put((meta, heatmap))


class ModelServer(Server):
    MODEL_PATH = "/home/matt/Models/honest-yogurt.ckpt"

    def __init__(self, host, port, checkpoint_name=None):
        super().__init__(host, port)
        if checkpoint_name is None:
            self.checkpoint_name = self.MODEL_PATH
        else:
            self.checkpoint_name = checkpoint_name
        self.workers = self.init_workers()

    def init_workers(self):
        workers = []
        for gpu_ii in range(torch.cuda.device_count()):
            workers.append(Process(
                target=run,
                args=(self.checkpoint_name, gpu_ii, self.in_queue,
                      self.ou_queue, self.ok_queue)))
            workers[-1].start()

        return workers

    @property
    def ready(self):
        return self.ok_queue.qsize() >= 1 + len(self.workers)

    def kill(self):
        self.process.kill()
        for worker in self.workers:
            worker.kill()
            worker.join()


if __name__ == '__main__':
    if socket.gethostname() == 'kat':
        ms = ModelServer("172.18.20.240", 5001)
    elif socket.gethostname() == "gorilla":
        ms = ModelServer("172.18.21.117", 5001)
    else:
        raise ValueError("That's not an animal.")
    ms.processes[-1].join()
