import multiprocessing
import time
from multiprocessing import Queue, Process
from typing import Tuple

import torch.cuda

from flask_model_cache.pretrained_model import PretrainedModel
from keypoint_detection.utils.load_checkpoints import load_from_checkpoint
from utils.tools import pyout, pbar


def model_cache(checkpoint_name, cuda_idx: int,
                in_queue: Queue, ou_queue: Queue,
                shape: Tuple[int, int] = (512, 512)):
    model = PretrainedModel(checkpoint_name, gpu=cuda_idx)
    ou_queue.put({"msg": f"GPU{cuda_idx}: success!"})

    while True:
        resp = in_queue.get()
        if resp['msg'] == "die!":
            break
        else:
            img = resp["image"]

            top_box = (0, 0, 1080, 1080)
            top_img = img.crop(top_box)
            top_img = top_img.resize(shape)

            bot_box = (0, 840, 1080, 1920)
            bot_img = img.crop(bot_box)
            bot_img = bot_img.resize(shape)

            _, top_result = model(top_img)
            _, bot_result = model(bot_img)

            top_heat = torch.zeros((1820 // 2, 1024 // 2))
            top_heat[:1024 // 2] = top_result[0]

            bot_heat = torch.zeros((1820 // 2, 1024 // 2))
            bot_heat[-1024 // 2:] = bot_result[0]

            heatmap = torch.maximum(top_heat, bot_heat)
            heatmap = heatmap.cpu().numpy().tolist()
            resp["heatmap"] = heatmap
            resp["msg"] = "processed"
            ou_queue.put(resp)


class ModelPool:
    def __init__(self, checkpoint_name):
        multiprocessing.set_start_method("spawn")

        self.queue_ou, self.queue_in = Queue(), Queue()
        self.processes = []
        self.initialize_workers(checkpoint_name)

        self.counter = 0

    def initialize_workers(self, checkpoint_name):
        for gpu_ii in range(torch.cuda.device_count()):
            # for gpu_ii in range(1):
            self.processes.append(Process(
                target=model_cache,
                args=(checkpoint_name, gpu_ii, self.queue_ou, self.queue_in)))
            self.processes[-1].start()

        for _ in pbar(range(torch.cuda.device_count()), desc="Loading Cache"):
            resp = self.queue_in.get()
            if not "success" in resp['msg']:
                pyout()

    def add_to_queue(self, img, data):
        msg = data
        msg["msg"] = "process_image"
        msg["image"] = img
        self.queue_ou.put(msg)
        self.counter += 1

    def get_from_queue(self):
        results = []
        for _ in pbar(range(self.counter)):
            results.append(self.queue_in.get())
            self.counter -= 1

        pyout()
