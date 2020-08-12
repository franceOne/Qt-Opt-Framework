import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar
from QLearningDerive import Agent
import gym
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from DataCollector  import DataCollector
from  Model import Model as Md
from ReplayBuffer import ReplayBuffer
from  Trainingsworkers import Trainingworkers
from _thread import start_new_thread
from BellmannUpdater import BellmanUpdater


enviroment = gym.make("Pendulum-v0").env

print('Number of states: {} High: {} Low {}'.format(enviroment.observation_space, enviroment.observation_space.high , enviroment.observation_space.low))
print('Number of actions: {} High: {} Low {}'.format(enviroment.action_space, enviroment.action_space.high, enviroment.action_space.low  ))

print( "Action:", enviroment.action_space.sample())

print("State",  enviroment.reset())


def policyFunction(action):
    return action

def getState(state):
    return state[0]


optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.7, clipvalue=10)


#modelSrc  = "simulation/src/RLAlgorithm/Algorithm/QTOpt/saved_model/TEST"
#modelSrc = 'simulation/src/RLAlgorithm/Algorithm/QTOpt/saved_model/DQN'
#modelSrc = 'simulation/src/RLAlgorithm/Algorithm/QTOpt/saved_model/TEST2'


modelSrcWeights=  'simulation/src/RLAlgorithm/Algorithm/QTOpt/Distributed Qt-Opt/saved_model/Weights/TEST/WitoutState'

loss =  "mse"
stateSize = 1
actionSize = 1
agent = Md(enviroment, optimizer, loss, policyFunction, modelSrcWeights,  state_size=stateSize, action_size=actionSize, camerashape=enviroment.render(mode="rgb_array").shape)
agent.loadWeights()
replayBuffer = ReplayBuffer(state_size=stateSize, action_size=actionSize, camerashape=enviroment.render(mode="rgb_array").shape)
bellmannUpdater = BellmanUpdater(replayBuffer, agent)
dataCollector =  DataCollector(replayBuffer, agent, enviroment, policyFunction, getState)
trainingsworker = Trainingworkers(replayBuffer, agent)
start_new_thread(dataCollector.start, ("Thread-1", 1 ))
start_new_thread(bellmannUpdater.start, ("Thread-2",1  ))
start_new_thread(trainingsworker.start, () )



batch_size = 32
num_of_episodes = 500
agent.q_network.summary()







print("Train")
#Training.train(enviroment, agent, policyFunction,  observationsize=stateSize, num_of_episodes=num_of_episodes, train=True , maxStepSize=100, loadModell=True, saveModell=True)

input("Testen des modells")
#Training.train(enviroment, agent, policyFunction, observationsize=stateSize,  num_of_episodes=100, train=False, maxStepSize=100, loadModell=True)

