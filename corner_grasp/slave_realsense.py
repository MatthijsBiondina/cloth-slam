import time
from multiprocessing import Process, Queue, shared_memory
from typing import Tuple

import numpy as np

from utils.tools import pyout
import pyrealsense2


class RealsenseSlave():
    # Different from camera FPS. This one's only relevant when the camera is
    # not recording.
    CONTROL_LOOP_FREQUENCY = 60

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

    def run(self, system_queue, ready_queue, img_queue):
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

                image, shared_mem = self.__create_shared_array(
                    shape=(frame.height, frame.width, 3))
                image[:] = np.asanyarray(frame.get_data())

                img_queue.put((frame.timestamp, shared_mem.name))

                # _, shared_mem_name = img_queue.get()
                # existing_shm = shared_memory.SharedMemory(
                #     name=shared_mem_name)
                # existing_shm.close()
                # existing_shm.unlink()

                counter += 1

        pyout(f"Recorded {counter} frames.")

        pipeline.stop()

    def __create_shared_array(self, shape, dtype=np.uint8):
        size = np.prod(shape)
        shared_mem = shared_memory.SharedMemory(
            create=True, size=size * dtype().itemsize)

        array = np.ndarray(shape, dtype=dtype, buffer=shared_mem.buf)

        return array, shared_mem

    def start_recording(self):
        self.system_queue.put("start_recording")

    def stop_recording(self):
        self.system_queue.put("stop_recording")

    def shutdown(self):
        self.system_queue.put("shutdown")
        self.process.join()

        counter_good, counter_bad = 0, 0
        while not self.image_queue.empty():
            try:
                _, shared_mem_name = self.image_queue.get()
                existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
                existing_shm.close()
                existing_shm.unlink()
                counter_good += 1
            except Exception:
                counter_bad += 1

        pyout(f"Cleaned up {counter_good} frames. Leaked {counter_bad} frames.")


        pyout()
