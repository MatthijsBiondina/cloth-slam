import json
import multiprocessing
import socket
from multiprocessing import Queue, Process
from typing import Tuple

from utils.socket_utils import init_socket_server, client_handler
from utils.tools import pyout


class Server():
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.in_queue, self.ou_queue = Queue(), Queue()
        self.ok_queue = multiprocessing.Queue()
        self.processes = [Process(
            target=self.run,
            args=(self.in_queue, self.ou_queue, self.ok_queue))]
        self.processes[-1].start()

    def run(self, in_queue, ou_queue, ok_queue):
        server_socket = init_socket_server(self.host, self.port)
        server_socket.listen(5)
        ok_queue.put("Ready!")
        try:
            while True:
                client_sock, addr = server_socket.accept()
                process = multiprocessing.Process(
                    target=client_handler,
                    args=(client_sock, addr, in_queue, ou_queue))
                process.start()
        finally:
            server_socket.close()

    @property
    def ready(self):
        return not self.ok_queue.empty()

    def kill(self):
        for process in self.processes:
            process.kill()
        for process in self.processes:
            process.join()


if __name__ == '__main__':
    s = Server("172.18.20.240", 5000)
