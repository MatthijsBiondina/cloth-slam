import base64
import io

from PIL import Image

from utils.tools import pyout


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
