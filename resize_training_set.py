import json

import numpy as np
from PIL import Image
from pycocotools.mask import frPyObjects, decode, encode

from utils.tools import pyout, pbar, makedirs

in_root = "/home/matt/Datasets/cloth/large_dream"
ou_root = "/home/matt/Datasets/cloth/small_dream"
makedirs(ou_root)
makedirs(f"{ou_root}/images")

with open(f"{in_root}/annotations.json", "r") as f:
    annotations = json.load(f)

for M, A in pbar(
        zip(annotations['images'], annotations['annotations']),
        total=len(annotations['images'])):
    assert M['id'] == A['image_id']

    # Read and resize image
    img = Image.open(f"{in_root}/{M['file_name']}")
    img = img.resize((512, 512))
    img.save(f"{ou_root}/{M['file_name']}")

    # Update image metadata
    M['height'] = 512
    M['width'] = 512

    # Rescale bounding boxes
    A['bbox'] = [x / 2 for x in A['bbox']]

    # Rescale keypoints, excluding visibility flag
    A['keypoints'] = [x / 2 if (ii + 1) % 3 else x
                      for ii, x in enumerate(A['keypoints'])]

    # Rescale segmentation masks
    rle = A['segmentation']

    mask = decode(rle)
    resized_mask = np.array(Image.fromarray(mask).resize((512, 512),
                                                         Image.NEAREST))
    new_rle = encode(np.asfortranarray(resized_mask))

    A['segmentation']['counts'] = new_rle['counts'].decode('utf-8')
    A['segmentation']['size'] = [512, 512]

    # Update the area
    A['area'] = int(
        np.sum(decode(new_rle)).item())  # Converts np.int64 to Python int

with open(f"{ou_root}/annotations.json", "w+") as f:
    json.dump(annotations, f, indent=2)
