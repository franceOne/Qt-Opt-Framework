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

def runClient(stateSize, actionSize, camerashape, functions, getEnvironment, optimizer, loss, modelSrcWeights, dataCollectionPath, 
dataCollerctorNumber = 1,bellmannNumber = 1, trainingsWorkerNumber = 1, replayLog =  True, loadWeights = False,  
replayBufferPath = "localhost:5000", modelPath = "localhost:5001"  ):
    #Inits
    main_lock = Lock()
    model_lock = Lock()
    getData, getState, getObservation, getReward, policyFunction = functions()

    print("\n DataCollectors:", dataCollerctorNumber, "\n Bellmans:", bellmannNumber, "\n Trainingsworkers:", trainingsWorkerNumber, "\n replayLog", replayLog)
    
    # Client- Helpers
    client = Client(replayBufferPath)
    modelClient = ModelClient(modelPath)

    agent = Md(modelClient, model_lock, getEnvironment(), optimizer, loss, policyFunction, modelSrcWeights,  state_size=stateSize, action_size= actionSize, camerashape=camerashape)
        
    bellmannUpdater = BellmanUpdater(client, agent)
    trainingsworker = Trainingworkers(client,  agent)

   
    if replayLog:
        print("Run ReplayLog")
        start_new_thread( ReplayLog(dataCollectionPath+"_0/", client))
        print("Finish ReplayLog")


    for i in range(dataCollerctorNumber):
        print("start datacollector", i)
        start_new_thread(DataCollector(i, client,  agent, getEnvironment(), policyFunction, getState, getReward, dataCollectionPath).start, (main_lock, True))
    
    for i in range(bellmannNumber):
        print("start belmann updater", i)
        start_new_thread(bellmannUpdater.start, ())

    for i in range(trainingsWorkerNumber):
        print("start tainingworkers", i)
        start_new_thread(trainingsworker.start, ())



