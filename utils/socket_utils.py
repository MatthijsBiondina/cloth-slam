import json
import os
import signal
import socket
import subprocess
import time
from multiprocessing import Queue
from typing import Tuple

from utils.tools import pyout


def init_socket_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    close_socket_if_occupied(port)
    server_socket.bind((host, port))
    pyout(f"Server started on {host}:{port}")
    return server_socket


def init_socket_client(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    return client_socket


def close_socket_if_occupied(port):
    pids = get_pids(port)
    for pid in pids:
        try:
            pid = int(pid)
            os.kill(pid, signal.SIGKILL)
            pyout(f"Successfully killed process with PID: {pid}")
        except OSError as e:
            pyout(f"Error killing process {pid}: {e}")
        except ValueError:
            pyout(f"PID must be an integer: {pid}")
    while len(get_pids(port)) > 0:
        time.sleep(0.1)


def get_pids(port):
    result = subprocess.run(['lsof', '-i', f'tcp:{port}'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    output = result.stdout
    output = output.split("\n")[1:]
    pids = []
    for line in output:
        if "(LISTEN)" in line:
            while "  " in line:
                line = line.replace("  ", " ")
            pids.append(line.split(' ')[1])
    pids = list(set(pids))
    return pids


def send_confirmation(sock, msg):
    sock.send(str(msg).encode('utf-8'))


def recv_confirmation(sock):
    msg = sock.recv(1024).decode('utf-8')
    if msg == 'ok':
        return True
    else:
        raise RuntimeError(f"Unexpected server response: {msg}")


def client_handler(client_socket: socket.socket,
                   client_address: Tuple[str, int],
                   task_queue: Queue,
                   res_queue: Queue):
    pyout(f"Connection started "
          f"({client_address[0]}:{client_address[1]})")
    while True:
        # Receive metadata from the client
        try:
            resp = client_socket.recv(1024)
        except ConnectionResetError:
            pyout(f"Connection dropped "
                  f"({client_address[0]}:{client_address[1]})")
            break
        if len(resp) == 0:
            pyout(f"Connection dropped "
                  f"({client_address[0]}:{client_address[1]})")
            break
        else:
            try:
                meta_data = json.loads(resp.decode('utf-8'))
            except:
                pyout()
            if 'method' not in meta_data:
                send_confirmation(
                    client_socket,
                    "Please specify 'task' in metadata. I don't know "
                    "what to do with this package.")
            elif meta_data['method'] == 'post':
                send_confirmation(client_socket, 'ok')
                # Receive image data from client
                buffer = b''
                while len(buffer) < meta_data['filesize']:
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    buffer += data
                send_confirmation(client_socket, 'ok')
                task_queue.put((meta_data, buffer))
            elif meta_data['method'] == "idle":
                if task_queue.empty():
                    send_confirmation(client_socket, True)
                else:
                    send_confirmation(client_socket, False)
            elif meta_data['method'] == "get":
                if res_queue.empty():
                    send_confirmation(client_socket, False)
                else:
                    meta, data = res_queue.get()
                    client_socket.sendall(json.dumps(meta).encode('utf-8'))
                    recv_confirmation(client_socket)
                    client_socket.sendall(data)
                    recv_confirmation(client_socket)
            else:
                pyout()
