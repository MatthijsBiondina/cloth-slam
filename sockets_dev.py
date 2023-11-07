import time
from multiprocessing import Process

from flask_model_cache.client import load_images_and_json, send_data_to_server
from flask_model_cache.server import Server
from utils.tools import pbar


def start_server():
    s = Server()


if __name__ == '__main__':
    process = Process(target=start_server)
    process.start()
    time.sleep(1)

    server_addr = ('172.18.20.240', 5000)
    image_dir = "/home/matt/Pictures/towels/trial_exposure_250"
    images_data = load_images_and_json(image_dir)
    send_data_to_server(server_addr, images_data)

    process.kill()
    process.join()
