import gym
import tensorflow as tf
from RunClient import runClient
import numpy as np
import sys
import time

def createEnvironemnt(environment = "FetchReach-v1"):
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
    #print(state)
    #return reward
    archieved_goal = state["achieved_goal"]
    desired_goal = state["desired_goal"]
    observation = state["observation"]
    grip  = observation[0:3]
    abs_object = observation[3:6]
    rel_object = observation[6:9]
    dis_grip_obj = np.linalg.norm(abs_object- grip)
    dis_obj_goal = np.linalg.norm(archieved_goal-desired_goal)


    #print((- np.exp(dis_obj_goal*10)) + 1)

    return (- np.exp(dis_obj_goal*10)) + 1
    
    
    dist = - ( (dis_grip_obj*30) )
    print("dis grip_obj", dis_grip_obj)
    print("dis_obj_goal", dis_obj_goal)
    print("robot", grip)
    print(exp(dist))
    #time.sleep(3)
    if dis_grip_obj < 0.02:
        return 0
    else:
        return exp(dist) - 1
    #return reward + dist

def getObservation(envrionment, state ):
    return state["observation"]

def getData(environment, action):
    next_state, reward, terminated, info = enviroment.step(action)
    return getObservation(envrionment, next_state), getState(next_state), reward, terminated



def returnFunctions():
    return getData, getState, getObservation, getReward, policyFunction


name = 'fetch_reach/1000epochs'

modelSrcWeights=  'saved_model/Weights/'+name
dataCollectionPath = 'saved_model/data/'+name
camerashape=  (500,500,3)
loss =  "mse"
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.7, clipvalue=10)



state = enviroment.reset()

print(state["observation"], "\n", state["achieved_goal"], "\n", state["desired_goal"], "\n")
observation = state["observation"]
grip  = observation[0:3]
abs_object = observation[3:6]
rel_object = observation[6:9]

print(grip, abs_object, "\n", rel_object, "\n" , abs_object-grip)

print("Rel_object", rel_object)
print("Gripper state", grip)

def run():
    runClient(config["stateSize"], config["actionSize"], camerashape, 
    returnFunctions,  createEnvironemnt, optimizer, loss, 
    modelSrcWeights, dataCollectionPath, 
    dataCollerctorNumber, bellmannNumber, trainingsWorkerNumber, replayLog, loadWeights=True)
    input("... \n")

run()
