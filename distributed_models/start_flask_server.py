import socket

from flask_model_cache.pretrained_model_server import ModelServer

# MODEL_PATH = "/home/matt/Models/honest-yogurt.ckpt"

if __name__ == '__main__':
    if socket.gethostname() == 'kat':
        ms = ModelServer("172.18.20.240", 5001)
    elif socket.gethostname() == "gorilla":
        ms = ModelServer("172.18.21.117", 5001)
    else:
        raise ValueError("That's not an animal.")
    ms.processes[-1].join()
