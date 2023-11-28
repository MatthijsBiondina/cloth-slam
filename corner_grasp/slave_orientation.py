import time
from multiprocessing import Queue, Process
from queue import Empty

import numpy as np

from utils.exceptions import BreakException
from utils.tools import pyout
from utils.utils import wait_for_next_cycle, deserialize_ndarray


class OrientationSlave:
    def __init__(self, in_queue: Queue):
        self.in_queue = in_queue
        self.ou_queue = Queue()
        self.sys_queue = Queue()

        self.process = Process(
            target=self.run,
            args=(self.in_queue, self.ou_queue, self.sys_queue))
        self.process.start()

    def run(self, in_queue: Queue, ou_queue: Queue, sys_queue: Queue):
        while True:
            t_start = time.time()
            try:
                self.__process_sys_command(sys_queue)
                tcp_str, img_data, depth_data, kp_str = \
                    in_queue.get(timeout=1)

                depth_img = deserialize_ndarray(*depth_data, dtype=np.uint16)

                pyout()


            except BreakException:
                break
            except Empty:
                pass
            finally:
                wait_for_next_cycle(t_start)

    def __process_sys_command(self, sys_queue: Queue):
        if not sys_queue.empty():
            msg = sys_queue.get()
            if msg == "shutdown":
                raise BreakException()
            else:
                raise ValueError(f"Message {msg} unknown.")

    def shutdown(self):
        self.sys_queue.put("shutdown")
        self.process.join()
