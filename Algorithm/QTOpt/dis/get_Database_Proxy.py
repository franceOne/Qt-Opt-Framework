from ReplayBuffer import ReplayBuffer
from flask import Flask
stateSize = 3
actionSize = 1
camerashape=  (500,500,3)
bufferPath = 'saved_model/buffer/TEST/FullState'

database_adresses = ["http://localhost:5000"]


def create_app():
    app = Flask(__name__)
    app.config['database_adresses'] = database_adresses
    return app