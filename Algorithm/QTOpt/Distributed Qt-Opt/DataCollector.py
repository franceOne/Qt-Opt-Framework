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
import random
from IPython.display import clear_output
from collections import deque
import progressbar

class DataCollector:
    def __init__(self, buffer, agent, environment, action_space_policy, state_policy):
        self.agent = agent
        self.buffer = buffer
        
        #Init variables
               
        self.policyFunction = action_space_policy
        self.environment = environment
        self.max_step_size = 100
        self.get_state = state_policy
    

        self.num_of_episodes = 100
        self.target1Network = None
        self.updateTarget1Network()


        self.step = 0
        self.episode = 0

      

    def start(self, threadName  ="", delay = 0):
        while True:
         self.collectData()




    def getTarget1Network(self):
        if self.target1Network:
            return self.target1Network
        else:
            self.updateTarget1Network()
            return self.target1Network


    def storeData(state, action, concatenatedImage, reward, next_state, next_concatenatedImage, terminated):
        self.buffer.storeOnlineData(state, action, concatenatedImage, reward, next_state, next_concatenatedImage, terminated)


    def updateTarget1Network(self):
        self.target1Network = self.agent.getTarget1Network()

    

          

    def collectData(self):
         enviroment = self.environment
         # Begin new Episode
         for i in range(self.num_of_episodes):
            
            # Reset the enviroment
            state = self.environment.reset()

            # Initialize variables
            rewardSum = 0
            terminated = False
            step = 0
            lastImage = enviroment.render(mode="rgb_array")
            bar = progressbar.ProgressBar(maxval=self.max_step_size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            bar.start()
            camera = enviroment.render(mode="rgb_array")
            
            if i % 2 == 0:
                self.updateTarget1Network()
                print("fetch new TargetNetwork")
            #print(type(camera))


            # Run Episode
            while not terminated:
                #print("step")

                #Render
                enviroment.render()
                concatenatedImage = np.concatenate((lastImage, camera), axis=0)
                
                # Run Action
                action = self.agent.get_Action(enviroment, self.get_state(state), self.agent.getReshapedImg(concatenatedImage), True, self.getTarget1Network())
                action = self.policyFunction(action)

                # Take action    
                next_state, reward, terminated, info = enviroment.step(action)
                next_camera  =  enviroment.render(mode="rgb_array")

                next_concatenatedImage = np.concatenate((camera, next_camera), axis=0)
                #print("is camera eq", np.array_equal(camera, next_camera))
                #print("action", action, "terminated", terminated, "reward", reward)
                #next_state = np.reshape(next_state, [1,observationsize]) 
                self.buffer.storeOnlineData(self.get_state(state), action, self.agent.getReshapedImg(concatenatedImage), reward, self.get_state(next_state), self.agent.getReshapedImg(next_concatenatedImage), terminated)
                
                #Update Counter
                self.step  += 1
                step += 1
                rewardSum += reward
                state = next_state
                lastImage = camera
                camera = next_camera
                bar.update(step)

                if terminated or step>=self.max_step_size:
                    bar.finish()
                    print("**********************************")
                    print("Episode {} Reward {}".format(self.episode, rewardSum))
                    print("**********************************")
                    break

            self.episode += 1



