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


enviroment = gym.make("Pendulum-v0").env
#enviroment.render()

print('Number of states: {} High: {} Low {}'.format(enviroment.observation_space, enviroment.observation_space.high , enviroment.observation_space.low))
print('Number of actions: {} High: {} Low {}'.format(enviroment.action_space, enviroment.action_space.high, enviroment.action_space.low  ))

print( "Action:", enviroment.action_space.sample())

print("State",  enviroment.reset())


def policyFunction(action):
    return action

optimizer = tf.keras.optimizers.SGD(learning_rate=0.00005, momentum=0.7)

loss =  "mse"
stateSize = 3
actionSize = 1
agent = Agent(enviroment, optimizer, loss, policyFunction,  state_size=stateSize, action_size=actionSize, camerashape=enviroment.render(mode="rgb_array").shape)

batch_size = 32
num_of_episodes = 500
agent.q_network.summary()



print("Train")
Training.train(enviroment, agent, policyFunction,  observationsize=stateSize, num_of_episodes=num_of_episodes, train=True , maxStepSize=100, loadModell=True)
print("RUN")
Training.train(enviroment, agent, policyFunction, observationsize=stateSize,  num_of_episodes=100, train=False, maxStepSize=300, loadModell=True)

