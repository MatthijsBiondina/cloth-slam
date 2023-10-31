from io import BytesIO

import numpy as np
import requests
from PIL import Image

from keypoint_integration.utils.tools import pyout


class ServerResponseError(Exception):
    def __init__(self, status_code, message=None):
        self.status_code = status_code
        self.message = message or f"Unexpected server response: {status_code}"
        super().__init__(self.message)


class FlaskModel:
    def __init__(self, server_url):
        self.server_url = server_url
        self.check_model_loaded()

    def check_model_loaded(self):
        try:
            response = requests.get(f'{self.server_url}/check_model')
            response_json = response.json()
            assert response_json.get('model_loaded')
        except (requests.exceptions.ConnectionError, AssertionError):
            raise EnvironmentError(
                "Model not loaded. Ensure the server is started and the "
                "model is loaded.")

    def __call__(self, image_array: np.ndarray):
        # Convert numpy array to a bytes buffer
        img = Image.fromarray(image_array.astype('uint8'))
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Send image to server
        files = {'image': ('image.png', BytesIO(img_byte_arr), 'image/png')}
        response = requests.post(f"{self.server_url}/process_image",
                                 files=files)

        if response.status_code == 200:
            image_data = BytesIO(response.content)
            image = Image.open(image_data)

            # Convert to numpy
            processed_image_array = np.array(image)

            return processed_image_array
        else:
            raise ServerResponseError(response.status_code, response.text)
