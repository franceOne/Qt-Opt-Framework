import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar
from SimpleQLearning import Agent
import Training
import gym
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam


enviroment = gym.make("Taxi-v2").env
enviroment.render()

print('Number of states: {}'.format(enviroment.observation_space.n))
print('Number of actions: {}'.format(enviroment.action_space.n))


optimizer = Adam(learning_rate=0.01)
agent = Agent(enviroment, optimizer)

batch_size = 32
num_of_episodes = 100
timesteps_per_episode = 1000
agent.q_network.summary()


Training.train(enviroment,optimizer, agent, batch_size, num_of_episodes,timesteps_per_episode)