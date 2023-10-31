from flask_model_cache.flask_server import flask_server_startup

MODEL_PATH = "/home/matt/Models/epoch=14-step=5610.ckpt"

if __name__ == '__main__':
    flask_server_startup(model_path=MODEL_PATH)
