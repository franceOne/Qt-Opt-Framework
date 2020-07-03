import numpy as np

class Trainingworkers:
    def __init__(self,  buffer, agent):
        self.buffer = buffer
        self.agent = agent

        self.batch_size = 64
        self.step = 0
   

    def start(self):
        while True:
            self.updateQFunction()

    def getNetwork(self):
        return self.agent.getQNetwork()


    def updateQFunction(self):
        dataSize = self.buffer.getTrainDataSize()
        if dataSize > self.batch_size:
            print("Train Network")
            states, actions, cameras, next_states, next_cameras, rewards, terminates, q_target = self.buffer.getTrainBuffer(self.batch_size)
            self.train(cameras, states, actions, q_target)


    def getWeights(self):
        return self.getNetwork().get_weights()

    def getStateActionArray(self, state, action):
       return np.concatenate((action, state), axis=1) 

    def train(self , npCameras, npStates, npActions , q_target):
    
      self.step +=1

      #Train
      state_action_array  =  self.getStateActionArray(npStates, npActions)
      self.getNetwork().train_on_batch([npCameras,state_action_array], q_target)
      
      #Save weights and update lagged Networks
      self.agent.saveWeights()
      #self.getWeights()
      if (self.step%10 == 0):
        self.agent.alighn_target_model()
        print(self.getWeights())
     
     




