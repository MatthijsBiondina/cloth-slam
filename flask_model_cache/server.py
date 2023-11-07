import json
import multiprocessing
import os
import re
import signal
import socket
import subprocess
import time
from multiprocessing import Queue
from typing import Tuple

from utils.tools import pyout


def send_confirmation(socket, msg):
    socket.send(str(msg).encode('utf-8'))


def client_handler(client_socket: socket.socket,
                   client_address: Tuple[str, int],
                   task_queue: Queue, res_queueu: Queue):
    while True:
        # Receive metadata from the client
        meta_data = client_socket.recv(1024)
        if len(meta_data) == 0:
            pyout(f"Connection dropped "
                  f"({client_address[0]}:{client_address[1]})")
            break
        else:
            meta_data = json.loads(meta_data.decode('utf-8'))
            if 'task' not in meta_data:
                send_confirmation(
                    client_socket,
                    "Please specify 'task' in metadata. I don't know what to "
                    "do with this package.")
            elif meta_data['task'] == 'process':
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


class DeploymentManager:
    SERVER_ADDRESSES = {'kat': ('172.18.20.240', 5001)}

    def __init__(self, task_queue, results_queue):
        self.task_queue: Queue = task_queue
        self.results_queue: Queue = results_queue


class Server():
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 5000

    def __init__(self):
        self.task_queue, self.results_queue = Queue(), Queue()
        self.run()

    def run(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_socket.bind((self.SERVER_HOST, self.SERVER_PORT))
        except OSError:
            self.kill_socket()
            server_socket.bind((self.SERVER_HOST, self.SERVER_PORT))

        server_socket.listen(5)
        pyout("Server is running and waiting for connections...")

        try:
            while True:
                client_sock, addr = server_socket.accept()
                process = multiprocessing.Process(
                    target=client_handler,
                    args=(
                        client_sock, addr, self.task_queue,
                        self.results_queue))
                process.start()
        finally:
            server_socket.close()

    def kill_socket(self):
        pids = self.get_pids()
        for pid in pids:
            try:
                pid = int(pid)
                os.kill(pid, signal.SIGKILL)
                pyout(f"Successfully killed process with PID: {pid}")
            except OSError as e:
                pyout(f"Error killing process {pid}: {e}")
            except ValueError:
                pyout(f"PID must be an integer: {pid}")
        while len(self.get_pids()) > 0:
            time.sleep(0.1)

    def get_pids(self):
        result = subprocess.run(['lsof', '-i', f'tcp:{self.SERVER_PORT}'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        output = result.stdout
        pyout(output)
        output = output.split("\n")[1:]
        pids = []
        for line in output:
            if "(LISTEN)" in line:
                while "  " in line:
                    line = line.replace("  ", " ")
                pids.append(line.split(' ')[1])
        pids = list(set(pids))
        return pids


if __name__ == '__main__':
    s = Server()
