import base64
import json
import sys
import time
from io import BytesIO
from multiprocessing import Process

import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib.pyplot import colormaps

from flask_model_cache.client import load_images_and_json, \
    send_data_to_server, recv_data_from_server
from flask_model_cache.deployment_manager import DeploymentManager
from flask_model_cache.pretrained_model_server import ModelServer
from flask_model_cache.server import Server
from utils.tools import pbar, pyout, makedirs
from utils.utils import serialize_pillow_image, deserialize_pillow_image


def apply_viridis_colormap(heatmap_bw):
    # Convert grayscale image to numpy array
    heatmap_array = np.array(heatmap_bw)

    # Normalize the heatmap to the range [0, 1]

    normalized_heatmap = np.clip(heatmap_array / 255, 0., 1.)

    # Apply the viridis colormap (it returns RGBA image)
    viridis_colormap = colormaps.get_cmap('viridis')
    viridis_heatmap = viridis_colormap(normalized_heatmap,
                                       bytes=True)  # RGBA values in 0-255 range

    # Convert RGBA heatmap to an Image
    viridis_heatmap_image = Image.fromarray(viridis_heatmap, mode='RGBA')

    return viridis_heatmap_image


def overlay_heatmap(base_image, heatmap_bw, alpha=0.5):
    # Apply the viridis colormap to the heatmap
    viridis_heatmap_image = apply_viridis_colormap(heatmap_bw)
    viridis_heatmap_image.putalpha(int(255 * alpha))
    # Overlay the heatmap on the base image

    # viridis_heatmap_image.show()
    base_image.putalpha(255)
    combined = Image.alpha_composite(base_image, viridis_heatmap_image)

    return combined


def serialize(obj: dict):
    for idx in obj.keys():
        for field in obj[idx].keys():
            if isinstance(obj[idx][field], Image.Image):
                obj[idx][field], _ = serialize_pillow_image(obj[idx][field])
            elif isinstance(obj[idx][field], np.ndarray):
                # Convert np arrays to list
                obj[idx][field] = str(obj[idx][field].tolist())
            elif isinstance(obj[idx][field], bytes):
                obj[idx][field] = (
                    base64.b64encode(obj[idx][field]).decode('ascii'))

    # Return the serializable object
    return obj


if __name__ == '__main__':

    ds = DeploymentManager('172.18.20.240', 5000)
    while not ds.ready:
        time.sleep(0.1)

    server_addr = ('172.18.20.240', 5000)
    image_dir = "/home/matt/Pictures/towels/trial_exposure_340"
    images_data = load_images_and_json(image_dir, debug=False)
    # images_data = images_data[:10]

    send_data_to_server(server_addr, images_data)

    exposure = image_dir.split("_")[-1]
    ou_root = f"/home/matt/Pictures/towels/results_{exposure}"
    makedirs(ou_root)
    results = {}

    for res in recv_data_from_server(server_addr, images_data):
        # for res in pbar(results, desc="Generating output images"):
        ou_img = overlay_heatmap(res['img'], res['heatmap'])
        idx = res['frame_idx']
        ou_img = ou_img.convert("RGB")
        ou_img.save(f"{ou_root}/{str(idx).zfill(3)}.jpg")
        results[res['frame_idx']] = res

    ds.kill()

    serialized_results = serialize(results)
    with open(f"/home/matt/Datasets/cloth/processed_{exposure}", "w+") as f:
        json.dumps(serialized_results, indent=2)
