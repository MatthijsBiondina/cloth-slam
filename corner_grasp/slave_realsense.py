import base64
import pickle
import time
from multiprocessing import Process, Queue, shared_memory
from typing import Tuple

import cv2
import numpy as np

from utils.tools import pyout
import pyrealsense2

from utils.utils import serialize_ndarray, deserialize_ndarray


class RealsenseSlave():
    # Different from camera FPS. This one's only relevant when the camera is
    # not recording.
    CONTROL_LOOP_FREQUENCY = 60
    MAX_QSIZE = 100

    def __init__(self,
                 resolution: Tuple[int, int] = (1280, 720),
                 frame_rate: int = 10
                 ):
        self.resolution = resolution
        self.frame_rate = frame_rate

        self.system_queue = Queue()
        self.ready_queue = Queue()
        self.image_queue = Queue()

        self.process = Process(
            target=self.run,
            args=(self.system_queue, self.ready_queue, self.image_queue))
        self.process.start()
        while self.ready_queue.empty():
            time.sleep(1 / 60)

    def run(self, system_queue: Queue, ready_queue: Queue, img_queue: Queue):
        pipeline = pyrealsense2.pipeline()
        config = pyrealsense2.config()
        config.enable_stream(pyrealsense2.stream.color, *self.resolution,
                             pyrealsense2.format.bgr8, self.frame_rate)
        profile = pipeline.start()
        ready_queue.put("Ready")

        paused = True
        counter = 0
        while True:
            if not system_queue.empty():
                msg = system_queue.get()
                if msg == "start_recording":
                    paused = False
                elif msg == "stop_recording":
                    paused = True
                elif msg == "shutdown":
                    break
                else:
                    raise ValueError(f"Message \"{msg}\" unknown.")

            if paused:
                time.sleep(1 / self.CONTROL_LOOP_FREQUENCY)
            else:
                frame = pipeline.wait_for_frames().get_color_frame()
                img = np.asanyarray(frame.get_data()).astype(np.uint8)

                if img_queue.qsize() < self.MAX_QSIZE:
                    img_queue.put((frame.timestamp/1000,
                                   serialize_ndarray(img)))
                    counter += 1

        pipeline.stop()

    def start_recording(self):
        self.system_queue.put("start_recording")

    def stop_recording(self):
        self.system_queue.put("stop_recording")

    def shutdown(self):
        pyout(f"Queue size: {self.image_queue.qsize()}")

        self.system_queue.put("shutdown")
        # while not self.image_queue.empty():
        #     timestamp, shape, img_string = self.image_queue.get()
        #     img = deserialize_ndarray(shape, img_string)
        #
        #     cv2.imshow("Window", img)
        #     cv2.waitKey(100)

        self.process.join()
