import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar
from QLearningDerive import Agent
import Training
import gym
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam


enviroment = gym.make("HalfCheetah-v2").env
#enviroment.render()

print('Number of states: {} Height: {} Low {}'.format(enviroment.observation_space, enviroment.observation_space.high , enviroment.observation_space.low))
print('Number of actions: {} Height: {} Low {}'.format(enviroment.action_space, enviroment.action_space.high, enviroment.action_space.low  ))

print( "Action:", enviroment.action_space.sample())

print("State",  enviroment.reset())

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)

loss =  "mse"
agent = Agent(enviroment, optimizer, loss, state_size=6, action_size=6)

batch_size = 128
num_of_episodes = 500
agent.q_network.summary()


def policyFunction(action):
    return action

print("Train")
Training.train(enviroment, agent, policyFunction,  observationsize=17, batch_size=batch_size, num_of_episodes=num_of_episodes )
print("RUN")
Training.runModel(enviroment, policyFunction, agent, 100 )
