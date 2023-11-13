import io
import json
import os
import socket
import time

import imageio
import numpy as np
from PIL import Image

from utils.socket_utils import send_confirmation
from utils.tools import pyout, pbar


def img2bytes(img: Image, quality: int = 85):
    # Convert the image to a JPEG bytestring
    with io.BytesIO() as output:
        img.save(output, format="JPEG", quality=quality, optimize=True)
        data = output.getvalue()

    metadata = {'format': "JPEG", "filesize": len(data)}

    return data, metadata


def correct_tcp_matrix(tcp_matrix):
    """
    Corrects the rotational part of a 4x4 TCP matrix to ensure it is
    orthogonal with a determinant of 1, while keeping the translation part
    unchanged.

    Args:
    tcp_matrix (numpy.ndarray): The 4x4 TCP matrix to be corrected.

    Returns:
    numpy.ndarray: The corrected 4x4 TCP matrix.
    """
    # Extract the rotation part of the matrix
    rotation_matrix = tcp_matrix[:3, :3]

    # Calculate the determinant to check if the rotation matrix is valid
    det = np.linalg.det(rotation_matrix)

    # If the determinant is not close to 1, then correct the rotation matrix
    if not np.isclose(det, 1):
        # Perform Singular Value Decomposition (SVD) on the rotation matrix
        U, _, Vt = np.linalg.svd(rotation_matrix)
        # Reconstruct the rotation matrix from U and Vt to enforce
        # orthogonality
        corrected_rotation = np.dot(U, Vt)

        # If the determinant of the corrected matrix is -1, flip the sign of
        # the last column of U
        if np.isclose(np.linalg.det(corrected_rotation), -1):
            U[:, -1] = -U[:, -1]
            corrected_rotation = np.dot(U, Vt)

        # Replace the rotation part in the original TCP matrix with the
        # corrected one
        tcp_matrix[:3, :3] = corrected_rotation
    tcp_matrix[3, :] = np.array([0, 0, 0, 1])

    # Return the corrected TCP matrix
    return tcp_matrix


def load_images_and_json(root, debug=False):
    """
    Load images and their corresponding metadata from disk.

    This function reads JPEG images and associated metadata from a given directory,
    processes each image's filename to extract indices and TCP matrices,
    and packs them along with the image's byte data into a list.

    :param root: The root directory containing the image files.
    :return: A list of dictionaries, each containing an image, its frame index,
             TCP matrix, byte data, and metadata.
    """

    # List to store image data and metadata
    images_and_data = []

    # Iterate over all files in the directory with progress bar
    for ii, filename in pbar(list(enumerate(os.listdir(root))),
                             desc="Read from disc"):
        if debug and ii >= 10:
            pyout("Early stop for DEBUG mode.")
            break
        # Process only .jpg files
        if filename.endswith('.jpg'):
            # Open the image file
            img = Image.open(f"{root}/{filename}")
            img = img.rotate(270, expand=True)

            # Extract frame index and TCP string from the filename
            idx, tcp_str = filename.replace(".jpg", "").split("_")
            idx = int(idx)  # Convert frame index to integer
            tcp_list = list(map(float, tcp_str.split(
                " ")))  # Convert TCP string to float list
            tcp = correct_tcp_matrix(np.array(tcp_list).reshape(
                (4, 4)))  # Reshape and correct TCP matrix

            # Read the image byte data
            with open(f"{root}/{filename}", "rb") as image_file:
                data = image_file.read()
                metadata = {'filesize': len(data),
                            'idx': ii}  # Prepare metadata

            # Append the image data and metadata to the list
            images_and_data.append(
                {'img': img, 'frame_idx': idx, 'tcp': tcp, 'bytes': data,
                 'meta': metadata})
        else:
            fname, ext = os.path.splitext(filename)
            raise ValueError(f"File type ({ext}) not supported.")

    return images_and_data


def block_until_confirmation(socket):
    msg = socket.recv(1024).decode('utf-8')
    if msg == 'ok':
        return True
    else:
        raise RuntimeError(f"Unexpected server response: {msg}")


def send_data_to_server(server_address, images_and_data):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)

    for data in pbar(images_and_data, desc="Transmitting data"):
        data['meta']['method'] = 'post'
        client_socket.sendall(json.dumps(data['meta']).encode('utf-8'))
        block_until_confirmation(client_socket)
        client_socket.sendall(data['bytes'])
        block_until_confirmation(client_socket)
    pyout("All data sent!")


def recv_data_from_server(server_address, images_and_data):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(server_address)

    bar = pbar(range(len(images_and_data)), desc="Receiving Heatmaps")
    results = []
    while True:
        if len(results) == len(images_and_data):
            break

        client.sendall(json.dumps({'method': 'get'}).encode('utf-8'))
        resp = client.recv(1024).decode('utf-8')
        if resp == "False":
            time.sleep(0.1)
        else:
            meta = json.loads(resp)
            send_confirmation(client, "ok")
            buffer = b''
            while len(buffer) < meta['filesize']:
                data = client.recv(1024)
                if not data:
                    break
                buffer += data

            item = [i for i in images_and_data
                    if i['meta']['idx'] == meta['idx']][0]
            item['heatmap'] = Image.open(io.BytesIO(buffer))

            yield item

            results.append(item)

            #
            # # for i in images_and_data:
            # #     pyout(f"{idx} =? {i['frame_idx']}")
            # #     pyout(i)
            #
            # results.append((meta, buffer))

            bar.update(1)
            send_confirmation(client, "ok")

    # return results


if __name__ == "__main__":
    server_addr = ('172.18.20.240', 5000)
    image_dir = "/home/matt/Pictures/towels/trial_exposure_250"
    images_data = load_images_and_json(image_dir)
    send_data_to_server(server_addr, images_data)
    pyout()
