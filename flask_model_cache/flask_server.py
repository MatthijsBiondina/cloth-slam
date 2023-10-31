from io import BytesIO

import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request, send_file

from keypoint_integration.inference.pretrained_model import PretrainedModel
from keypoint_integration.utils.tools import pyout
from torchvision import transforms

MODEL_NAME = "/home/matt/Models/epoch=12-step=4862.ckpt"
app = Flask(__name__)
model = None


@app.route('/check_model', methods=['GET'])
def check_model():
    global model
    return jsonify(model_loaded=bool(model))


@app.route('/process_image', methods=['POST'])
def process_image():
    global model
    assert model is not None, "Model not loaded on server"

    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    # Convert the image file to a PIL Image object
    img = Image.open(file.stream)

    # # Define the transformation
    # transform = transforms.Compose([transforms.ToTensor()])
    #
    # # Apply the transformation and add a batch dimension
    # img_tensor = transform(img).unsqueeze(0)
    # img_tensor = img_tensor.to(model.device)

    with torch.no_grad():
        keypoints, heatmap = model(img)

    heatmap = heatmap.squeeze(0).cpu().numpy()
    assert np.max(heatmap) <= 1

    heatmap_img = Image.fromarray((heatmap * 255).astype('uint8'))

    # Save the heatmap image to a BytesIO object
    heatmap_byte_arr = BytesIO()
    heatmap_img.save(heatmap_byte_arr, format='PNG')

    # Send the heatmap image back to the client
    heatmap_byte_arr.seek(0)
    return send_file(heatmap_byte_arr, mimetype='image/png')


if __name__ == '__main__':
    model = PretrainedModel(MODEL_NAME)
    app.run(host='0.0.0.0', port=5000)
