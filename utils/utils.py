import base64
import io
import time
import warnings
from multiprocessing import Queue
from queue import Empty
from typing import Tuple

import numpy as np
from PIL import Image
from matplotlib.pyplot import colormaps

from utils.tools import pyout


def serialize_ndarray(A: np.ndarray):
    bytes_ = A.tobytes()
    base64_ = base64.b64encode(bytes_)
    return A.shape, base64_.decode()


def deserialize_ndarray(shape: Tuple[int, ...],
                        bytestring: str,
                        dtype=np.uint8):
    bytes_ = base64.b64decode(bytestring)
    array = np.frombuffer(bytes_, dtype=dtype)
    assert array.size == np.prod(shape)
    return array.reshape(shape)


def serialize_pillow_image(img: Image.Image, quality=85):
    img = img.convert('RGB')
    with io.BytesIO() as output:
        img.save(output, format="JPEG", quality=quality, optimize=True)
        data = output.getvalue()

    base64_str = base64.b64encode(data).decode('ascii')

    metadata = {'format': "JPEG", "filesize": len(base64_str)}

    return base64_str, metadata


def deserialize_pillow_image(base64_string):
    # Decode the base64 string to bytes
    image_data = base64.b64decode(base64_string)

    # Create a BytesIO object from the bytes
    image_stream = io.BytesIO(image_data)

    # Use Image.open to read from the bytes stream and get the image
    image = Image.open(image_stream)

    return image


def clear_queue(queue: Queue):
    try:
        while not queue.empty():
            queue.get(timeout=0.1)
    except TypeError:
        pass
    except Empty:
        pass


def wait_for_next_cycle(start_time: float,
                        control_loop_frequency: int = 60):
    loop_duration = 1. / control_loop_frequency
    end_time = time.time()
    elapsed_time = end_time - start_time

    if elapsed_time < loop_duration:
        time.sleep(loop_duration - elapsed_time)
        return True
    else:
        return False


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
