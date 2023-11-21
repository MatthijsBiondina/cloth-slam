import json
from multiprocessing import Process

from flask_model_cache.server import Server
from utils.socket_utils import init_socket_client, recv_confirmation, \
    send_confirmation
from utils.tools import pyout


def run(hosts, port, in_queue, ou_queue, ok_queue):
    sockets = {}
    for hostname, ip in hosts.items():
        sockets[hostname] = init_socket_client(ip, port)
        ok_queue.put(f"{hostname} ready")

    while True:
        # Send to workers
        if not in_queue.empty():
            for _, client in sockets.items():
                client.sendall(json.dumps({"method": "idle"}).encode('utf-8'))
                idle = client.recv(1024).decode('utf-8') == "True"
                if idle:
                    meta, data = in_queue.get()
                    client.sendall(json.dumps(meta).encode('utf-8'))
                    recv_confirmation(client)
                    client.sendall(data)
                    recv_confirmation(client)
                    break

        # Receive from workers
        for _, client in sockets.items():
            while True:
                client.sendall(json.dumps({'method': 'get'}).encode('utf-8'))
                result_pending = client.recv(1024).decode('utf-8')
                if result_pending == "False":
                    break
                else:
                    meta_data = json.loads(result_pending)
                    send_confirmation(client, "ok")
                    buffer = b''
                    while len(buffer) < meta_data['filesize']:
                        data = client.recv(1024)
                        if not data:
                            break
                        buffer += data
                    send_confirmation(client, 'ok')
                    ou_queue.put((meta_data, buffer))


class DeploymentManager(Server):
    WORKER_HOSTS = {'kat': "172.18.20.240", 'gorilla': "172.18.21.117"}
    # WORKER_HOSTS = {'gorilla': "172.18.212.117"}
    WORKERS_PORT = 5001

    def __init__(self, host, port):
        super().__init__(host, port)
        self.init_workers()

    def init_workers(self):
        self.processes.append(Process(
            target=run,
            args=(self.WORKER_HOSTS, self.WORKERS_PORT,
                  self.in_queue, self.ou_queue, self.ok_queue)))
        self.processes[-1].start()

    @property
    def ready(self):
        return self.ok_queue.qsize() >= 1 + len(self.WORKER_HOSTS)
