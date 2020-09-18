import gym
import tensorflow as tf
from RunClient import runClient
import numpy as np
import sys
import time

def createEnvironemnt(environment = "Pendulum-v0"):
    return gym.make(environment).env

config = {
    "stateSize" : 3,    
    "actionSize":  1
}

dataCollerctorNumber = 1
bellmannNumber = 1
trainingsWorkerNumber = 1
replayLog = False

def get_arg(i):
    return int(sys.argv[i])

for i in range(1, len(sys.argv)):
    if i == 1:
        dataCollerctorNumber = int(sys.argv[i])
    if i == 2:
        bellmannNumber = get_arg(i)
    if i == 3:
        trainingsWorkerNumber = get_arg(i)
    if i == 4:
        replayLog = True

enviroment = createEnvironemnt()


print('Number of states: {} '.format(enviroment.observation_space))
print('Number of actions: {} '.format(enviroment.action_space, enviroment.action_space, enviroment.action_space  ))

print('States Shape:', enviroment.observation_space.shape)
print('Action Shape:', enviroment.action_space.shape)



def policyFunction(action):
    return action


def getState(state):
    return state
    

def getReward(state, reward):
    return reward

def getObservation(envrionment, state ):
    return None

def getData(environment, action):
    return None

def get_cem_action_size():
    return 3


def returnFunctions():
    return getData, getState, getObservation, getReward, policyFunction, get_cem_action_size 

name = 'pendulum/1000epochs'

modelSrcWeights=  'saved_model/Weights/'+name
dataCollectionPath = 'saved_model/data/'+name
camerashape=  (500,500,3)
loss =  "mse"
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.7, clipvalue=10)


def run():
    runClient(config["stateSize"], config["actionSize"], camerashape, 
    returnFunctions,  createEnvironemnt, optimizer, loss, 
    modelSrcWeights, dataCollectionPath, 
    dataCollerctorNumber, bellmannNumber, trainingsWorkerNumber, replayLog, loadWeights=True)
    input("... \n")

run()
