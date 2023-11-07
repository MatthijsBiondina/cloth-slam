import io
import json
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request, send_file

from torchvision import transforms

from flask_model_cache.model_pool import ModelPool
from flask_model_cache.pretrained_model import PretrainedModel
from utils.tools import pyout

MODEL_NAME = "/home/matt/Models/epoch=12-step=4862.ckpt"
app = Flask(__name__)
pool: ModelPool = None


@app.route('/check_model', methods=['GET'])
def check_model():
    global pool
    return jsonify(model_loaded=bool(pool))


@app.route('/add_to_queue', methods=['POST'])
def add_request_to_queue():
    global pool
    assert pool is not None, "Model not loaded on server"

    image_file = request.files['image']
    img = Image.open(io.BytesIO(image_file.read()))

    json_data = request.files['data'].read()
    data = json.loads(json_data)

    pool.add_to_queue(img, data)

    # Send a confirmation response
    return jsonify({"message": "Image received and added to queue"}), 202

@app.route('/get_from_queue', methods=['GET'])
def empty_results_queue():
    global pool
    assert pool is not None, "Model not loaded on server"

    results = pool.get_from_queue()

    pyout()


def flask_server_startup(model_path):
    global pool

    # model = PretrainedModel(model_path)
    pool = ModelPool(model_path)
    app.run(host="0.0.0.0", port=5000)
