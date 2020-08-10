from ModelBackend import Model
from flask import Flask
import tensorflow as tf
#from ../../ import PendulumFullState1 as Config


#stateSize, actionSize, camerashape, optimizer, loss = Config.getConfiguration()

stateSize = 31
actionSize = 4
camerashape=  (500,500,3)
loss =  "mse"
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.7, clipvalue=10)

agent = Model( optimizer, loss,  state_size=stateSize, action_size=actionSize, camerashape=camerashape)


def create_app():
    app = Flask(__name__)
    app.config['Model'] = agent
    return app