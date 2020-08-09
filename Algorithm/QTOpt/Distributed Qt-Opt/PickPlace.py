import gym
import tensorflow as tf
from RunClient import runClient
import numpy as np
import sys

def createEnvironemnt(environment = "FetchPickAndPlace-v1"):
    return gym.make(environment).env



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
    archieved_goal = state["achieved_goal"]
    desired_goal = state["desired_goal"]
    dist = -np.linalg.norm(archieved_goal-desired_goal) 
    return dist

def getObservation(envrionment, state ):
    return state["observation"]

def getData(environment, action):
    next_state, reward, terminated, info = enviroment.step(action)
    return getObservation(envrionment, next_state), getState(next_state), reward, terminated



def returnFunctions():
    return getData, getState, getObservation, getReward, policyFunction



state = enviroment.reset()
print(state["achieved_goal"], state["desired_goal"] )
getReward(state, 0)
exit(1)

modelSrcWeights=  'saved_model/Weights/TEST/Pick_Place/FullState'
dataCollectionPath = 'saved_model/buffer/TEST3/Pick_Place/FullState/NumpyData'


camerashape=  (500,500,3)
loss =  "mse"
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.7, clipvalue=10)





def run():
    runClient(config["stateSize"], config["actionSize"], camerashape, 
    returnFunctions,  createEnvironemnt, optimizer, loss, 
    modelSrcWeights, dataCollectionPath, 
    dataCollerctorNumber, bellmannNumber, trainingsWorkerNumber, replayLog)
    input("... \n")

run()
