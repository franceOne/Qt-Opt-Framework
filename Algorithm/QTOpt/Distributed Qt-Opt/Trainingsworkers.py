import numpy as np
import tensorflow as tf
import time

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
            print("Trainworker: TrainBufferSize is to small:", dataSize, "Go Sleep")
            time.sleep(10)


    def parseGrad(self, grads):
        for i in range(len(grads)):
            grads[i] = grads[i].numpy()

        grads = np.asarray(grads)
        return grads

    def train(self , npCameras, npStates, npActions , q_target):
    
        self.step +=1

        #Train
        state_action_array  =  self.agent.getStateActionArray(npStates, npActions)
        qnetwork = self.agent.getQNetwork()
        #qnetwork.train_on_batch([npCameras,state_action_array], q_target)

        print("TRRRAAAAINNN")

     
        loss = self.agent.getLoss()
        
        
        with tf.GradientTape() as tape:
                predict = qnetwork([npCameras, state_action_array])
                loss_value = loss(q_target, predict)
                

        print( "LOSS", loss_value.numpy().mean() )
        grads = tape.gradient(loss_value, qnetwork.trainable_variables, unconnected_gradients='zero')
        old_gras = self.parseGrad(grads)

        for i in range(len(old_gras)):
            const = "test"
            #print("Before Clipping", old_gras[i].mean())
        
        
        grads = [(tf.clip_by_value(grad, -1.0, 1.0))
                                  for grad in grads]
        #print("gradients", len(grads), type(np.asarray(grads)))

        for i in range(len(grads)):
            grads[i] = grads[i].numpy()

        grads = np.asarray(grads)

        for i in range(len(grads)):
            const = "test"
            #print("After clipping", grads[i].mean())

        if np.isnan(loss_value.numpy().mean()):
            print("TRAININGSWORKER", "STOOOP", "loss is nan")
            while True:
                pass
            #print(q_target)
        else:
            #Save weights and update lagged Networks
            print("Traininsworker upate gradients")
            self.agent.updateNetworkByGradient(grads)
     
      
     
     




