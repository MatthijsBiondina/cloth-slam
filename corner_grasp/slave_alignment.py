import pickle
import time
from multiprocessing import Queue, Process
from typing import Dict, Tuple, List, Any

import numpy as np

from utils.exceptions import BreakException
from utils.tools import pyout
from utils.utils import clear_queue, wait_for_next_cycle, deserialize_ndarray


class TemporalAligner:
    CONTROL_LOOP_FREQUENCY = 15

    def __init__(self,
                 tcp_queue: Queue,
                 img_queue: Queue):
        self.tcp_queue = tcp_queue
        self.img_queue = img_queue
        self.sys_queue = Queue()
        self.pair_queue = Queue()

        self.process = Process(
            target=self.run,
            args=(self.tcp_queue, self.img_queue, self.sys_queue,
                  self.pair_queue))
        self.process.start()

    def run(self,
            tcp_queue: Queue,
            img_queue: Queue,
            sys_queue: Queue,
            out_queue: Queue):
        TCP = {}

        while True:
            t_start = time.time()
            try:
                self.__process_sys_command(sys_queue)
                self.__read_tcp_queue(TCP, tcp_queue)
                self.__read_img_queue(TCP, img_queue, out_queue)

                wait_for_next_cycle(t_start, self.CONTROL_LOOP_FREQUENCY)

            except BreakException:
                break

        pyout()

    def __process_sys_command(self, sys_queue):
        if not sys_queue.empty():
            msg = sys_queue.get()
            if msg == "shutdown":
                raise BreakException()
            else:
                raise ValueError(f"Message {msg} unknown.")

    def __read_tcp_queue(self,
                         TCP: Dict[int, Dict[str, Any]],
                         queue: Queue):
        while not queue.empty():
            timestamp, tcp_data = queue.get()
            pose = pickle.loads(tcp_data)[None, ...]
            seconds = int(timestamp)

            try:
                TCP[seconds]["time"] = np.concatenate(
                    (TCP[seconds]["time"], np.array([timestamp, ])), axis=0)
                TCP[seconds]["pose"] = np.concatenate(
                    (TCP[seconds]["pose"], pose), axis=0)
            except KeyError:
                TCP[seconds] = {"time": np.array([timestamp, ]),
                                "pose": pose}

    def __read_img_queue(self,
                         TCP: Dict[int, List[np.ndarray]],
                         in_queue: Queue,
                         ou_queue: Queue):
        while not in_queue.empty():
            timestamp, img_data, depth_data = in_queue.get()
            seconds = int(timestamp)

            tcp_slice = [TCP[seconds + ds] for ds in (-1, 0, 1)
                         if seconds + ds in TCP]
            tcp_times = np.concatenate([s["time"] for s in tcp_slice], axis=0)
            tcp_poses = np.concatenate([s["pose"] for s in tcp_slice], axis=0)

            argmin = np.argmin(np.abs(tcp_times - timestamp))
            ou_queue.put((
                pickle.dumps(tcp_poses[argmin]), img_data, depth_data))

            for s in list(TCP.keys()):  # Clear old values
                if s < seconds - 1:
                    del TCP[s]
                else:
                    break

    def shutdown(self):
        self.sys_queue.put("shutdown")
        clear_queue(self.tcp_queue)
        clear_queue(self.img_queue)
        clear_queue(self.sys_queue)
        self.process.join()
