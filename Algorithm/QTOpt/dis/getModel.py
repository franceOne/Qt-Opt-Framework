import gym
import tensorflow as tf
from RunClient import runClient
import numpy as np
import sys
import time

import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar
import gym
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from DataCollector  import DataCollector
from  Model import Model as Md
from  Trainingsworkers import Trainingworkers
from _thread import start_new_thread
from BellmannUpdater import BellmanUpdater
from threading import Thread, Lock
from flask import Flask
from clientWrapper import Client
from ReplayLog import ReplayLog
from ModelClientWrapper import ModelClient

def createEnvironemnt(environment = "FetchReach-v1"):
    return gym.make(environment).env


enviroment = createEnvironemnt()


modelSrcWeights=  'saved_model/Weights/Reward/Fetch_reach_new/FullState'


print('Number of states: {} '.format(enviroment.observation_space))
print('Number of actions: {} '.format(enviroment.action_space, enviroment.action_space, enviroment.action_space  ))

print('States Shape:', enviroment.observation_space.shape)
print('Action Shape:', enviroment.action_space.shape)

print(enviroment.observation_space["observation"].shape[0]+ enviroment.observation_space["achieved_goal"].shape[0] + enviroment.observation_space["desired_goal"].shape[0] )

config = {
    "stateSize" : enviroment.observation_space["observation"].shape[0]+ enviroment.observation_space["achieved_goal"].shape[0] + enviroment.observation_space["desired_goal"].shape[0],    
    "actionSize":  enviroment.action_space.shape[0]
}

print(config["stateSize"], config["actionSize"])


def policyFunction(action):
    return action


def getState(state):
    #print("Observation", state["observation"], "archieved", state["achieved_goal"], "des", state["desired_goal"])
    array =  np.concatenate([state["observation"],state["achieved_goal"], state["desired_goal"]],  axis=None)
    return array

def getReward(state, reward):
    return reward

def getObservation(envrionment, state ):
    return state["observation"]

def getData(environment, action):
    next_state, reward, terminated, info = enviroment.step(action)
    return getObservation(envrionment, next_state), getState(next_state), reward, terminated



def returnFunctions():
    return getData, getState, getObservation, getReward, policyFunction


camerashape=  (500,500,3)
loss =  "mse"
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.7, clipvalue=10)
model_lock = Lock()



def getAgent(path, modelPath="localhost:5001"):
    modelClient = ModelClient(modelPath)
    agent = Md(modelClient, model_lock, createEnvironemnt(), optimizer, loss, policyFunction, None, storedUrl=path, state_size=config["stateSize"], action_size= config["actionSize"], camerashape=camerashape)
    return agent