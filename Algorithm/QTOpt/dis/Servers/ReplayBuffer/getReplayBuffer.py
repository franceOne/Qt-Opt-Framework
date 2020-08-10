from ReplayBuffer import ReplayBuffer
from flask import Flask


stateSize = 31
actionSize = 4
camerashape=  (500,500,3)


replayBuffer = ReplayBuffer(state_size=stateSize, action_size=actionSize, camerashape=camerashape)


def create_app():
    app = Flask(__name__)
    app.config['ReplayBuffer'] = replayBuffer
    return app