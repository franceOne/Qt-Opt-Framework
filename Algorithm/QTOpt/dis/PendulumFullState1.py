import gym
import tensorflow as tf
from RunClient import runClient

def createEnvironemnt(environment = "Pendulum-v0"):
    return gym.make(environment).env


enviroment = createEnvironemnt()
#enviroment.render()

print('Number of states: {} High: {} Low {}'.format(enviroment.observation_space, enviroment.observation_space.high , enviroment.observation_space.low))
print('Number of actions: {} High: {} Low {}'.format(enviroment.action_space, enviroment.action_space.high, enviroment.action_space.low  ))

print('States Shape:', enviroment.observation_space.shape)
print('Action Shape:', enviroment.action_space.shape)

#print( "Action:", enviroment.action_space.sample())

#print("State",  enviroment.reset())


def policyFunction(action):
    return action


def getState(state):
    return state



modelSrcWeights=  'saved_model/Weights/putty/FullState'
dataCollectionPath = 'saved_model/buffer/putty/FullState/NumpyData'

stateSize = 3
actionSize = 1
camerashape=  (500,500,3)
loss =  "mse"
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.7, clipvalue=10)


dataCollerctorNumber = 1
bellmannNumber = 2
trainingsWorkerNumber = 2


def getConfiguration():
    return stateSize, actionSize, camerashape, optimizer, loss


def run():
    runClient(stateSize, actionSize, camerashape, 
    policyFunction, getState, createEnvironemnt, optimizer, loss, 
    modelSrcWeights, dataCollectionPath, dataCollerctorNumber, bellmannNumber, trainingsWorkerNumber)
    input("... \n")

#run()



