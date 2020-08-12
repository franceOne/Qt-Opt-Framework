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
from threading import Thread, Lock
from flask import Flask
from clientWrapper import Client
from ReplayLog import ReplayLog
from ModelClientWrapper import ModelClient

enviroment = gym.make("Pendulum-v0").env
#enviroment.render()

print('Number of states: {} High: {} Low {}'.format(enviroment.observation_space, enviroment.observation_space.high , enviroment.observation_space.low))
print('Number of actions: {} High: {} Low {}'.format(enviroment.action_space, enviroment.action_space.high, enviroment.action_space.low  ))

print( "Action:", enviroment.action_space.sample())

print("State",  enviroment.reset())


def policyFunction(action):
    return action


def getState(state):
    return state

def createEnvironemnt(environment = "Pendulum-v0"):
    return gym.make(environment).env



optimizer = tf.keras.optimizers.SGD(learning_rate=0.00000005, momentum=0.7, clipvalue=2)


#modelSrc  = "simulation/src/RLAlgorithm/Algorithm/QTOpt/saved_model/TEST"
#modelSrc = 'simulation/src/RLAlgorithm/Algorithm/QTOpt/saved_model/DQN'
#modelSrc = 'simulation/src/RLAlgorithm/Algorithm/QTOpt/saved_model/TEST2'


modelSrcWeights=  'saved_model/Weights/TEST/FullState'
dataCollectionPath = 'saved_model/buffer/TEST/FullState/NumpyData'

stateSize = 3
actionSize = 1
camerashape=  (500,500,3)
loss =  "mse"
client = Client("localhost:5000")
modelClient = ModelClient("localhost:5001")

main_lock = Lock()
model_lock = Lock()

agent = Md(modelClient, model_lock, enviroment, optimizer, loss, policyFunction, modelSrcWeights,  state_size=stateSize, action_size=actionSize, camerashape=camerashape)
agent.loadWeights()
replayBuffer = ReplayBuffer(state_size=stateSize, action_size=actionSize, camerashape=camerashape)

bellmannUpdater = BellmanUpdater(client, agent)
trainingsworker = Trainingworkers(client,  agent)

dataCollerctorNumber = 0
bellmannNumber = 0
trainingsWorkerNumber = 1



#ReplayLog(dataCollectionPath+"_0/", client)


for i in range(dataCollerctorNumber):
    print("start datacollector", i)
    start_new_thread(DataCollector(i, client,  agent, createEnvironemnt(), policyFunction, getState, dataCollectionPath).start, (main_lock, True))
   
for i in range(bellmannNumber):
    print("start belmann updater", i)
    start_new_thread(bellmannUpdater.start, ())

for i in range(trainingsWorkerNumber):
    print("start tainingworkers", i)
    start_new_thread(trainingsworker.start, ())





#start_new_thread( DataCollector(replayBuffer, agent, createEnvironemnt(), policyFunction, getState).start, (False,))
input("Testen des modells")
