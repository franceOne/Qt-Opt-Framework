from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow import keras
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import tensorflow as tf
import numpy as np
import random
from IPython.display import clear_output
import math
from CEM import CEM
import tensorflow.keras.backend as kb
import time


class BellmanUpdater:
    def __init__(self,  buffer, agent):
        self.agent = agent
        self.buffer = buffer
      
        # Init Model Props
        self.batch_size = 32
        self.gamma = 0.9


    def start(self , thread_name= "", delay= 0):
       while True:
          self.trains()


    def getQNetwork(self):
      return self.agent.getQNetwork()


    def trains(self):
      bufferSize = self.buffer.getOnlineDataSize()
      if bufferSize > self.batch_size:
        print("Run Bellmanupdater")
        states, actions, cameras, next_states, next_cameras, rewards, terminates = self.buffer.getOnlineBuffer(self.batch_size)
        self.train(states, actions, cameras, next_states, next_cameras, rewards, terminates, self.batch_size)
      
        
  

    def getTarget1Network(self):
        return self.agent.getTarget1Network()

        
    def train(self ,states, actions, cameras, next_states, next_cameras, rewards, terminates, batch_size):
    
      npTerminates = np.asarray(terminates)
      npRewards = np.asarray(rewards)
      npActions = np.asarray(actions)
      npStates = np.asarray(states)
      npNextStates = np.asarray(next_states)
      npCameras = np.asarray(cameras)
      npNext_Cameras = np.asarray(next_cameras)
      intTerminates = np.array(list(map(lambda y: [1- int(y)], npTerminates)))
    
       
      state_action_array  =  self.agent.getStateActionArray(npStates, npActions)
     
      #(BATCH, Actions)
      next_actions_samples = []
      for i  in range(batch_size):
        #(,Action)
        next_actions_samples.append(self.agent._get_cem_optimal_Action(next_states[i], npNext_Cameras[i], True))

      #print("next")
      next_actions = np.asarray(next_actions_samples)
      #print(next_actions)
      #print(next_states)
      next_state_action_array  =  self.agent.getStateActionArray(next_states, next_actions)
    
      q_next = self.agent.get_valueFunction(next_state_action_array, npNext_Cameras)



      q_target = np.empty([batch_size, 1])    
      #print(q_target)
      for i in range(batch_size):
        myTarget = npRewards[i] + self.gamma * np.amax(q_next[i])*intTerminates[i]
        if i == 0:
          pass
          #print("my Target", myTarget) 
        q_target[i] =  npRewards[i] + self.gamma * np.amax(q_next[i])*intTerminates[i]
        

      #print("Choosed Action", npActions[0])

      #print("Values", q_values[0])
      if math.isnan(q_target[0][0]):
        for i in range(batch_size):
           print(npRewards[i] + self.gamma * np.amax(q_next[i])*intTerminates[i])
           input("stop")
           print("npRewards", npRewards)
           input("stop1")
           print("q_next", q_next)
           print("intTerminates", intTerminates)
           input("MH")
           
     
        print(q_target)
        input("stop")

      self.buffer.storeTrainBuffer(npStates, npActions, npCameras, npRewards, npNextStates, npNext_Cameras, npTerminates, q_target, batch_size)

     

