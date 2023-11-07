import time
from multiprocessing import Process

from flask_model_cache.client import load_images_and_json, \
    send_data_to_server, recv_data_from_server
from flask_model_cache.deployment_manager import DeploymentManager
from flask_model_cache.pretrained_model_server import ModelServer
from flask_model_cache.server import Server
from utils.tools import pbar, pyout

if __name__ == '__main__':

    ds = DeploymentManager('172.18.20.240', 5000)
    while not ds.ready:
        time.sleep(0.1)

    server_addr = ('172.18.20.240', 5000)
    image_dir = "/home/matt/Pictures/towels/trial_exposure_340"
    images_data = load_images_and_json(image_dir)
    send_data_to_server(server_addr, images_data)
    recv_data_from_server(server_addr, images_data)

    ds.kill()
