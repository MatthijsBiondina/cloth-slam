import base64
import io
import json
import os

import numpy as np
from PIL import Image

from flask_model_cache.flask_client import FlaskModel
from utils.tools import pyout, pbar

exposure = "340"
root = f"/home/matt/Pictures/towels/trial_exposure_{exposure}"

model = FlaskModel("http://localhost:5000")

for file in pbar(os.listdir(root)):
    img = Image.open(f"{root}/{file}")
    img = img.rotate(270, expand=True)
    idx = int(file.split("_")[0])
    tcp = list(map(float, file.split("_")[-1].replace(".jpg", "").split(" ")))
    tcp = np.array(tcp).reshape((4, 4))

    model.add_to_queue(img, {"index": idx, "tcp": tcp.tolist()})
model.get_from_queue()
