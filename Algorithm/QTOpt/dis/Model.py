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


class Model:
    def __init__(self, modelClientWrapper, lock, enviroment, optimizer, loss,  action_space_policy, storedUrl = None ,  state_size= 4, action_size= 1, camerashape = (500,500,3),  cem_update_itr = 2, select_num = 6, num_samples = 64):
        #Model Properties
        self._state_size = state_size
        self._action_size = action_size
        self._camerashape = camerashape
        _imgReshapeSize = 16
        self._imgReshape = (_imgReshapeSize,_imgReshapeSize)
        cameraShapeList = list(self._camerashape)
        cameraShapeList[0] = _imgReshapeSize
        cameraShapeList[1] = _imgReshapeSize
        self._imgReshapeWithDepth =  cameraShape = tuple(cameraShapeList)
        self._networkOutputSize = 1
        self._actionStateReshape = (None,32)
        self.lock = lock
      
        self._optimizer = optimizer
        self._loss = tf.keras.losses.MSE

        self.url = storedUrl
        self.modelClientWrapper = modelClientWrapper


        #CEM
        self.cem_update_itr = cem_update_itr
        self.cem_select_num = select_num
        self.cem_num_samples = num_samples
        self.cem  = CEM(self._action_size)
        self.getActionPolicy = action_space_policy


        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.epsilon = 0.1
        self.train_step = 0

        #Iint ema
        self._emaFactor = 0.9999

         # Build networks
        self.q_network = self._build_compile_model()

        if self.url:
          try:
            self.q_network.load_weights(self.url)
            self.modelClientWrapper.storeQNetwork(self.q_network.get_weights())
            print("Load Modell from file")
          except Exception as e:
            print("Network coult not been loaded", e)
       


    def saveWeightsToFile(self, model):
      if self.url:
        model.save_weights(self.url)
        print("Save Modell to file")

    def getOptimizer(self):
      return self._optimizer

    def getLoss(self):
      return self._loss
   
    def getQNetwork(self):
      with self.lock:
        weights = self.modelClientWrapper.getQNetwork()
        newModel = self._build_compile_model()
        newModel.set_weights(weights)
        self.saveWeightsToFile(newModel)
        return newModel

  
    def saveWeights(self, weights):
      self.modelClientWrapper.storeQNetwork(weights)


    def updateNetworkByGradient(self, gradients):
      self.modelClientWrapper.updateNetworkByGradient(gradients)
      
    def loadWeights(self):
      try:
        self.q_network.load_weights(self.url)
        self.modelClientWrapper.storeQNetwork(self.q_network.get_weights())
      except:
        print("Weights could not been loaded")

      


    def checkWeights(self):
      print("Q_network", self.q_network.get_weights()[0:3])

      for i in range(self.numLaggedNNetwork):
          print("laggetNetwork", self._laggedTargetNetworkList[i].get_weights()[0:3])

      for i in range(self.numTagetNetworks):
        print("targetNetwork", self.targetNetworkList[i].get_weights()[0:3])



    """
    def updateLaggedNetworks(self):
      newLaggedNetworkIndex = (self.actualLaggedNetwork+1)%len(self._laggedTargetNetworkList)
      self._laggedTargetNetworkList[newLaggedNetworkIndex].set_weights(self.q_network.get_weights())
      
      self.actualLaggedNetwork = newLaggedNetworkIndex
    """


    def getReshapedImg(self, img):
      return tf.image.resize(img, list(self._imgReshape))

    def getStateActionArray(self, state, action):
       return np.concatenate((action, state), axis=1)


    def cross_entropy_loss(self,y_actual,y_pred):
      """
       -Sum(target log(pred)) = 
      """
      cross_EntropySum =  y_actual * tf.keras.backend.log(y_pred)
      custom_loss= - cross_EntropySum
      print(y_actual, y_pred, -custom_loss)
      input("W")
      return -custom_loss


    def _buildCameraModel(self):
      imgInput =   keras.Input(shape=(self._imgReshapeWithDepth), name='img_input')
      x = layers.BatchNormalization()(imgInput)
      x = layers.Conv2D(16, (3,3),activation="relu", name="input_Conv")(x)
      x = layers.BatchNormalization()(x)
      x = layers.GlobalMaxPool2D(name="camera_maxPool_output")(x)
      x = layers.BatchNormalization()(x)
      return (imgInput,x)

    def _buildActionStateModel(self):
      inputShape = self._state_size+self._action_size
      actionStateInput = keras.Input(shape=(inputShape,), name='q_input')
      x = layers.BatchNormalization()(actionStateInput)
      x = layers.Dense(75,  activation='relu', name="action_State_dense1")(x)
      x = layers.BatchNormalization()(x)
      x = layers.Dense(50,  activation='relu', name="action_State_dense2")(x)
      x = layers.BatchNormalization()(x)
      x = layers.Dense(16,  activation='relu', name="action_State_dense_output")(x)
      x = layers.BatchNormalization()(x)
      #reshape = layers.Reshape(self._actionStateReshape, name="action_State_reshape")(x)
      return (actionStateInput,x)

  
      
    def _build_compile_model(self):
      inputImg, outputImg = self._buildCameraModel()
      inputActionState, outputActionState = self._buildActionStateModel()
      x = layers.add([outputImg, outputActionState], name="add_actionstate_camera")
      x = layers.BatchNormalization()(x)
      #x = layers.Conv2D(16, (3,1),activation="relu", name="combined_conv2d")(x)
      x = layers.Dense(20,  activation='relu', name="combined_dense1")(x)
      x = layers.BatchNormalization()(x)
      output = layers.Dense(self._networkOutputSize, activation='linear', name="output")(x)    
      model = keras.Model(inputs=[inputImg, inputActionState], outputs = output)
      model.compile(loss=self._loss, optimizer=self._optimizer)
      return model


    def _build_compile_model_without_compile(self):
      inputImg, outputImg = self._buildCameraModel()
      inputActionState, outputActionState = self._buildActionStateModel()
      x = layers.add([outputImg, outputActionState], name="add_actionstate_camera")
      x = layers.BatchNormalization()(x)
      #x = layers.Conv2D(16, (3,1),activation="relu", name="combined_conv2d")(x)
      x = layers.Dense(20,  activation='relu', name="combined_dense1")(x)
      x = layers.BatchNormalization()(x)
      output = layers.Dense(self._networkOutputSize, activation='linear', name="output")(x)    
      model = keras.Model(inputs=[inputImg, inputActionState], outputs = output)
      model.compile(loss=self._loss, optimizer=self._optimizer)
      return model




    def getTarget1Network(self):
      with self.lock:
        target1, target2 = self.modelClientWrapper.getTargetNetworks()
        if(target1) is not None:
          model = self._build_compile_model()
          model.set_weights(target1)
          return  model
        else:
          print("MODEL: ERROR FETCHING QNETWORK 1")


    def getTarget2Network(self):
      with self.lock:
        target1, target2 = self.modelClientWrapper.getTargetNetworks()
        if(target2 is not None):
          model = self._build_compile_model()
          model.set_weights(target2)
          return  model
        else:
          print("MODEL: ERROR FETCHING QNETWORK 2")


    def _get_cem_optimal_Action(self,state, camera, training, networkToUse = None):
      #print("CEM state", state)

      #(32, BATCH ,4)
      states = np.tile(state, (self.cem_num_samples,1))
      #(32, BATCH,64,64,64,3)
      cameras = np.tile(camera, (self.cem_num_samples,1,1,1))

      #print(states.shape)
  
      #print("CEM STATES", states, states.shape)
      self.cem.reset()
      for i in range(self.cem_update_itr):
        #(32, BATCH,action)
        actions = self.cem.sample_multi(self.cem_num_samples)

        actions = self.getActionPolicy(actions)
        #print("CEM ActiONS", actions, actions.shape)
        stateActionArray = self.getStateActionArray(states, actions)
        #print("CEM STATe Action Array", stateActionArray, stateActionArray.shape)

        if networkToUse is not None:
          q_values = networkToUse.predict_on_batch([cameras, stateActionArray])
        elif training:
          q_values = self.getTarget1Network().predict_on_batch([cameras, stateActionArray])
        else:
          print("get q_values by q_network")
          q_values = self.getQNetwork().predict_on_batch([cameras, stateActionArray])
        reshaped_q_values = np.reshape(q_values, -1)
        #Max Index
        max_indx = np.argmax(reshaped_q_values)
     
        # Max n Index
        idx = np.argsort(reshaped_q_values)[-self.cem_select_num:]
        selected_actions = actions[idx]
        self.cem.update(selected_actions)
            
      optimal_action = actions[max_indx]
      return optimal_action



    def getActionFromEpsilonGreedyPolicy(self, enviroment):
       action = enviroment.action_space.sample()
       #print("EpsilonGreedyAction:", action)
       return action

    def getReshapedImg(self, img):
      return tf.image.resize(img, list(self._imgReshape))

          
    def get_Action(self, enviroment, state, camera, training, networkToUse = None):
        if training and (np.random.rand() <= self.epsilon):
            action = self.getActionFromEpsilonGreedyPolicy(enviroment)
            #print("Epsilon", action)
            return action

        optimal_action = self._get_cem_optimal_Action(state, camera, training, networkToUse)
        #print("CEM", optimal_action)
        return optimal_action
          
