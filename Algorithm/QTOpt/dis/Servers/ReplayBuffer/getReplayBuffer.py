from ReplayBuffer import ReplayBuffer
from flask import Flask
import gym


def createEnvironemnt(environment = "FetchReach-v1"):
    return gym.make(environment).env

enviroment = createEnvironemnt()
config = {
    "stateSize" : enviroment.observation_space["observation"].shape[0]+ enviroment.observation_space["achieved_goal"].shape[0] + enviroment.observation_space["desired_goal"].shape[0],    
    "actionSize":  enviroment.action_space.shape[0]
}

stateSize = config["stateSize"]
actionSize = config["actionSize"]
camerashape=  (500,500,3)


replayBuffer = ReplayBuffer(state_size=stateSize, action_size=actionSize, camerashape=camerashape)


def create_app():
    app = Flask(__name__)
    app.config['ReplayBuffer'] = replayBuffer
    return app