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
    def __init__(self,  optimizer, loss,  state_size= 4, action_size= 1, camerashape = (500,500,3),  cem_update_itr = 2, select_num = 6, num_samples = 64):
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
      
        self._optimizer = optimizer
        self._loss = loss

        self. is_initWeights = False

        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.epsilon = 0.3
        self.train_step = 0

        #Iint ema
        self._emaFactor = 0.9999

         # Build networks
        self.q_network = self._build_compile_model()
        self.numLaggedNNetwork = 3

        #Lagges Network
        self.actualLaggedNetwork = 0
        self._laggedTargetNetworkList = [tf.keras.models.clone_model(self.q_network) for _ in range(self.numLaggedNNetwork)]

        #TargetNetworks
        self.numTagetNetworks = 3
        self.actualTargetNetwork = 0
        self.target2Index = 2
        self.targetNetworkList = [tf.keras.models.clone_model(self.q_network) for _ in range(self.numTagetNetworks)]


        self.alighn_target_model()

   
    def updateByGradient(self, gradients):
      print("update, by gradients", len(gradients))
      self._optimizer.apply_gradients(zip(gradients, self.getQNetwork().trainable_variables))
      self.alighn_target_model()

    def getQNetwork(self):
      return self.q_network

    def initWeights(self):
      for i in range(self.numTagetNetworks):
         self.targetNetworkList[i].set_weights(self.q_network.get_weights())

      for i in range(self.numLaggedNNetwork):
          self._laggedTargetNetworkList[i].set_weights(self.q_network.get_weights())
      self.is_initWeights = True


     
    def reset(self):
      self.is_initWeights()
      self.alighn_target_model()

      
    def loadWeights(self):
      #print(self.q_network.get_weights())
      self.q_network.load_weights(self.url)
      #print(self.q_network.get_weights())
      self.reset()
      #self.checkWeights()
      #input("wait")

    def storeqNetworkWeights(self, weights):
      self.q_network.set_weights(weights)
      if not self.is_initWeights:
          self.initWeights()
      self.alighn_target_model()
    
     


    def checkWeights(self):
      print("Q_network", self.q_network.get_weights()[0:3])

      for i in range(self.numLaggedNNetwork):
          print("laggetNetwork", self._laggedTargetNetworkList[i].get_weights()[0:3])

      for i in range(self.numTagetNetworks):
        print("targetNetwork", self.targetNetworkList[i].get_weights()[0:3])


    def alighn_target_model(self):
      self.updateLaggedNetworks()
      targetNetwrok1EMAWeights = self.calculateEma(self.q_network.get_weights(), self.getLaggedNetwork().get_weights())
      
      newTargetIndex = (self.actualTargetNetwork+1)%self.numTagetNetworks     
      self.targetNetworkList[newTargetIndex].set_weights(targetNetwrok1EMAWeights)
      self.actualTargetNetwork = newTargetIndex

    

    def updateLaggedNetworks(self):
      newLaggedNetworkIndex = (self.actualLaggedNetwork+1)%len(self._laggedTargetNetworkList)
      self._laggedTargetNetworkList[newLaggedNetworkIndex].set_weights(self.q_network.get_weights())
      
      self.actualLaggedNetwork = newLaggedNetworkIndex


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


    def getLaggedNetwork(self):
      return self._laggedTargetNetworkList[self.actualLaggedNetwork]
        


    def calculateEma(self,weightsN ,weightsN_1):
      weightsN  = np.asarray(weightsN)
      weightsN_1 = np.asarray(weightsN_1)
      alpha =  self._emaFactor
      akpha_1 = 1- self._emaFactor
      return  (alpha * weightsN_1) + (akpha_1 * weightsN)



    def getTarget1Network(self):
      return self.targetNetworkList[self.actualTargetNetwork]


    def getTarget2Network(self):
      return self.targetNetworkList[(self.actualTargetNetwork-self.target2Index)%len(self.targetNetworkList)]
