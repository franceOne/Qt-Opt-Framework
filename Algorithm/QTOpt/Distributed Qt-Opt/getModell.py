from ModelBackend import Model
from flask import Flask
import tensorflow as tf
stateSize = 3
actionSize = 1
camerashape=  (500,500,3)
bufferPath = 'saved_model/buffer/TEST/FullState'
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.7, clipvalue=10)
loss =  "mse"

agent = Model( optimizer, loss,  state_size=stateSize, action_size=actionSize, camerashape=camerashape)


def create_app():
    app = Flask(__name__)
    app.config['Model'] = agent
    return app