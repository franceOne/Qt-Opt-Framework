from ModelBackend import Model
from flask import Flask
import tensorflow as tf
import gym
#from ../../ import PendulumFullState1 as Config


#stateSize, actionSize, camerashape, optimizer, loss = Config.getConfiguration()
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
loss =  "mse"
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.7, clipvalue=10)

agent = Model( optimizer, loss,  state_size=stateSize, action_size=actionSize, camerashape=camerashape)


def create_app():
    app = Flask(__name__)
    app.config['Model'] = agent
    return app