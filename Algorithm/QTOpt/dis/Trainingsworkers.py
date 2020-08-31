import numpy as np
import tensorflow as tf
import time
import os

class Trainingworkers:
    def __init__(self, clientWrapper,  agent, lock, path):
        self.clientWrapper = clientWrapper
        self.agent = agent

        self.batch_size = 64
        self.step = 0
        self.path = path
        self.lock = lock
   

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

    def loadNumpy(self, path):
        if not(os.path.exists(path)):
            print("Path does not exist", path)
            return None 
        loaded_file = np.load(path)
        return loaded_file


    def saveLoss(self, data, output_filename="loss.npy"):
        path = self.path

        if path is None:
            return
        homedir = os.path.expanduser("~")
        # construct the directory string
        dir_path = os.path.dirname(os.path.realpath(__file__))
        pathset = os.path.join(dir_path, path)
        # check the directory does not exist
       
        if not(os.path.exists(pathset)):
            # create the directory you want to save to
            os.makedirs(pathset)
            ds = {"ORE_MAX_GIORNATA": 5}
            # write the file in the new directory
        path_to_store = os.path.join(pathset, output_filename)
        oldData = self.loadNumpy(path_to_store)
        #print("data", data.shape, oldData, data)

        newData = [data]
        if oldData is not None:
            newData = np.concatenate((oldData, [data]), axis= 0)
            #print(output_filename, "olddata", oldData.shape, "data", data.shape, oldData, data)
            #print("newData", newData)
        np.save(path_to_store, newData)

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
        loss_mean = loss_value.numpy().mean()

        with self.lock:
                self.saveLoss(loss_mean)

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
     
      
     
     




