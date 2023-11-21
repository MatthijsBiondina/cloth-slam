import math
import multiprocessing
import pickle
import time
import warnings
from multiprocessing import Queue, Process
from queue import Empty
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from corner_grasp.grasp_config import PRETRAINED_MODEL_PATH, CAMERA_TCP
from flask_model_cache.pretrained_model import PretrainedModel
from utils.exceptions import BreakException
from utils.tools import pyout, pbar
from utils.utils import wait_for_next_cycle, deserialize_ndarray, clear_queue, \
    overlay_heatmap


class InferenceModel:
    def __init__(self,
                 in_queue: Queue):

        self.in_queue = in_queue
        self.out_queue = Queue()
        self.sys_queue = Queue()
        self.ready_queue = Queue()

        self.processes = []
        self.initialize_workers(PRETRAINED_MODEL_PATH)

    def initialize_workers(self, checkpoint_name):
        # for gpu_ii in range(torch.cuda.device_count()):
        for gpu_ii in range(1):
            self.processes.append(Process(
                target=self.run,
                args=(checkpoint_name, gpu_ii, self.in_queue, self.out_queue,
                      self.sys_queue, self.ready_queue)))
            self.processes[-1].start()

        for _ in pbar(range(len(self.processes)), desc="Loading Model Cache"):
            self.ready_queue.get()

    def run(self,
            checkpoint_name: str,
            cuda_idx: int,
            in_queue: Queue,
            ou_queue: Queue,
            sys_queue: Queue,
            ready_queue: Queue,
            kernel: int = 512):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PretrainedModel(checkpoint_name, gpu=cuda_idx)
        ready_queue.put(True)

        while True:
            t_start = time.time()
            try:
                self.__process_sys_command(sys_queue)

                try:
                    tcp_str, img_data = in_queue.get(timeout=1)
                    tcp = pickle.loads(tcp_str)
                    img = deserialize_ndarray(*img_data)
                    pil = self.__preprocess_image(tcp, img)
                    heat = self.__model_inference(pil, model, kernel)

                    self.__show(pil, heat)



                except Empty:
                    pass

                wait_for_next_cycle(t_start)
            except BreakException:
                break

        pyout()

    def __process_sys_command(self, sys_queue: Queue):
        if not sys_queue.empty():
            msg = sys_queue.get()
            if msg == "shutdown":
                raise BreakException()
            else:
                raise ValueError(f"Message {msg} unknown.")

    def __preprocess_image(self,
                           tcp_end_effector: np.ndarray,
                           img: np.ndarray):
        tcp = CAMERA_TCP @ tcp_end_effector
        rotation_3d = tcp[:3, :3]
        theta = np.arctan2(rotation_3d[2, 0], rotation_3d[1, 0])

        if theta > 0:  # rotated 90 degrees
            img = img.transpose(1, 0, 2)[::-1]
        else:
            img = img.transpose(1, 0, 2)[:, ::-1]

        return Image.fromarray(img)

    def __model_inference(self,
                          img: Image.Image,
                          model: PretrainedModel,
                          kernel_shape: int = 512):
        if img.width > img.height:
            raise NotImplementedError
        else:
            W = kernel_shape
            H = int(round(W / img.width * img.height))
            assert H < 2 * kernel_shape

            img_new = img.resize((W, H))
            img_top = img.crop((0, 0, W, W))
            img_bot = img.crop(
                (0, img_new.height - kernel_shape, W, img_new.height))

            _, res_top = model(img_top)
            _, res_bot = model(img_bot)

            heat = torch.cat((res_top[0, 0, :H // 2],
                              res_bot[0, 0, -int(math.ceil(H / 2)):]), dim=0)
            heat = np.clip(heat.detach().cpu().numpy(), 0., 1.)
            heat = cv2.resize(heat, (img.width, img.height))
            return heat

    def __show(self, img, heat):
        new_img = overlay_heatmap(img, heat*255)

        cv2.imshow("img", np.array(new_img))
        cv2.waitKey(50)

    def shutdown(self):
        for _ in self.processes:
            self.sys_queue.put("shutdown")

        clear_queue(self.in_queue)
        clear_queue(self.out_queue)

        for process in self.processes:
            process.join()
