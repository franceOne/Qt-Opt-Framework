import numpy as np
import tensorflow as tf

class Trainingworkers:
    def __init__(self, clientWrapper,  agent):
        self.clientWrapper = clientWrapper
        self.agent = agent

        self.batch_size = 64
        self.step = 0
   

    def start(self):
        while True:
            self.updateQFunction()

    def getNetwork(self):
        return self.agent.getQNetwork()


    def updateQFunction(self):
        dataSize = self.clientWrapper.getTrainDataSize()
      
        if dataSize > self.batch_size:
            print("Train Network", "Size", dataSize)
            states, actions, cameras, next_states, next_cameras, rewards, terminates, q_target = self.clientWrapper.getTrainBuffer(self.batch_size)
            self.train(cameras, states, actions, q_target)
        else:
            print("Trainworker: TrainBufferSize is to small:", dataSize)


    def train(self , npCameras, npStates, npActions , q_target):
    
      self.step +=1

      #Train
      state_action_array  =  self.agent.getStateActionArray(npStates, npActions)
      qnetwork = self.getNetwork()
      qnetwork.train_on_batch([npCameras,state_action_array], q_target)

      print("TRRRAAAAINNN")

      """"TEEEST"""
      qnetwork_clone = self.agent.getQNetwork_without_compile()
      qnetwork_clone_two = self.getNetwork()
      qnetwork_clone_three = self.getNetwork()
      optimizer = self.agent.getOptimizer()
      loss = self.agent.getLoss()

      weights_before = np.asarray(qnetwork_clone_two.get_weights())
      weights_before_two = np.asarray(qnetwork_clone_two.get_weights())

     
      while True:
        
        with tf.GradientTape() as tape:
                predict = qnetwork_clone([npCameras,state_action_array])
                print(predict)
                loss_value = loss(q_target, predict)
                

        print( "LOSS", loss_value.numpy().mean() )
        print(predict.shape, q_target.shape)
        #print(npCameras, state_action_array)
        #print(predict)
        grads = tape.gradient(loss_value, qnetwork_clone.trainable_variables, unconnected_gradients='zero')

        optimizer.apply_gradients(zip(grads, qnetwork_clone.trainable_variables))

        

        weights_aftter = np.asarray(qnetwork_clone.get_weights())
        print(type(weights_aftter), type(weights_before), weights_aftter.shape, weights_before.shape)
        print(np.array_equal(weights_aftter, weights_before), np.array_equal(weights_before_two, weights_before))

        for i in range(weights_before_two.shape[0]):
            break
           
            if not np.array_equal(weights_before_two[i], weights_before[i]):
                print("NOT TRUEEE")
                print("START", weights_before[i],  "COMPARE WiTH",weights_before_two[i])

            if not np.array_equal(weights_aftter[i], weights_before[i]):
                print("NOT TRUEEE OTHER")
                print("START", weights_before[i],  "COMPARE WiTH",weights_aftter[i])

        """TEESST"""
      
      #Save weights and update lagged Networks
      print("Traininsworker save weights")
      self.agent.saveWeights(qnetwork.get_weights())
     
      
     
     




