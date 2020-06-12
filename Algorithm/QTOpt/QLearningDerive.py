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


class Agent:
    def __init__(self, enviroment, optimizer, loss,  action_space_policy, state_size= 4, action_size= 1, camerashape = (500,500,3),  cem_update_itr = 2, select_num = 6, num_samples = 64):
        self._state_size = state_size
        self._action_size = action_size
        self._camerashape = camerashape
        _imgReshapeSize = 64
        self._imgReshape = (_imgReshapeSize,_imgReshapeSize)
        cameraShapeList = list(self._camerashape)
        cameraShapeList[0] = _imgReshapeSize
        cameraShapeList[1] = _imgReshapeSize
        self._imgReshapeWithDepth =  cameraShape = tuple(cameraShapeList)
        
        self._networkOutputSize = 1
        self._actionStateReshape = (1,1,32)
      
        self._optimizer = optimizer
        self._loss = loss


        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.epsilon = 0.3
        self.train_step = 0

        #Iint ema
        self._emaFactor = 0.9999


         # Build networks
        self.q_network = self._build_compile_model()
        self.numLaggedNNetwork = 3
        self.actualLaggedNetwork = 0
        self._laggedTargetNetworkList = [tf.keras.models.clone_model(self.q_network) for _ in range(self.numLaggedNNetwork)]

        #TargetNetworks
        self.numTagetNetworks = 3
        self.targetNetworkList = [tf.keras.models.clone_model(self.q_network) for _ in range(self.numTagetNetworks)]
      
        self.actualTargetNetwork = 0
        self.target2Index = 2
    
        self.replayBufferMaxLength = 500
        self.replyBufferBatchSize = 16
        # (s,a, S', r)
        data_spec = (tf.TensorSpec(self._state_size, tf.float64, 'state'),
        tf.TensorSpec(self._action_size, tf.float64, 'action'),
        tf.TensorSpec(self._imgReshapeWithDepth, tf.float32, 'camera'),
        tf.TensorSpec(self._state_size, tf.float64, 'next_state'),
        tf.TensorSpec(self._imgReshapeWithDepth, tf.float32, 'next_camera'),
        tf.TensorSpec(1, tf.float64, 'reward'),
        tf.TensorSpec(1, tf.bool, 'terminated'))
        
        self.replayBuffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, batch_size = 1, max_length=self.replayBufferMaxLength )
            
        #Init CEM
        self.cem_update_itr = cem_update_itr
        self.cem_select_num = select_num
        self.cem_num_samples = num_samples
        self.cem  = CEM(self._action_size)
        self.getActionPolicy = action_space_policy


       

        self.alighn_target_model()

     
     
     
      def custom_loss(y_actual,y_pred):
        """
         if y == 1:
          return -log(yHat)
        else:
          return -log(1 - yHat)
        """
      custom_loss=kb.square(y_actual-y_pred)
      return custom_loss


    def updateLaggedNetworks(self):
      newLaggedNetworkIndex = (self.actualLaggedNetwork+1)%len(self._laggedTargetNetworkList)
      newLaggedNetwork = self._laggedTargetNetworkList[newLaggedNetworkIndex]
      newLaggedNetwork.set_weights(self.q_network.get_weights())
      self._laggedTargetNetworkList[newLaggedNetworkIndex] = newLaggedNetwork
      self.actualLaggedNetwork = newLaggedNetworkIndex



    def store(self, state, action, camera, reward, next_state, next_camera, terminated, training):
        #(s,a,c, s', c', r t)
        camera = self.getReshapedImg(camera)
        next_camera = self.getReshapedImg(next_camera)
        values = (state, tf.dtypes.cast(action, tf.float64), camera, next_state, next_camera, reward, terminated)
        nestedStructure = tf.nest.map_structure(lambda t: tf.stack([t]* 1),values)
        self.replayBuffer.add_batch(nestedStructure)

        if(training and self.replayBuffer.num_frames() > self.replyBufferBatchSize):
          self.retrain(self.replyBufferBatchSize)
          

    def getStateActionArray(self, state, action):
       return np.concatenate((action, state), axis=1) 


    def _buildCameraMoel(self):
      imgInput =   keras.Input(shape=(self._imgReshapeWithDepth), name='img_input')
      x = layers.Conv2D(32, (6,2),activation="relu", name="input_Conv")(imgInput)
      output = layers.MaxPool2D((20,20), name="output_camra")(x)
      return (imgInput,output)

    def _buildActionStateModel(self):
      inputShape = self._state_size+self._action_size
      actionStateInput = keras.Input(shape=(inputShape,), name='q_input')
      x = layers.Dense(32,  activation='relu', name="dense_output")(actionStateInput)
      reshape = layers.Reshape(self._actionStateReshape, name="reshape")(x)
      return (actionStateInput,reshape)



      
    def _build_compile_model(self):
      inputImg, outputImg = self._buildCameraMoel()
      inputActionState, outputActionState = self._buildActionStateModel()
      x = layers.add([outputImg, outputActionState])
      x = layers.Dense(1,  activation='relu')(x)
      x = layers.Flatten()(x)
      output = layers.Dense(self._networkOutputSize, activation='sigmoid')(x)    
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

    def alighn_target_model(self):
        newTargetIndex = (self.actualTargetNetwork+1)%self.numTagetNetworks
        targetnetwork = self.targetNetworkList[newTargetIndex]

        targetNetwrok1EMAWeights = self.calculateEma(self.q_network.get_weights(), self.getLaggedNetwork().get_weights())
        targetnetwork.set_weights(targetNetwrok1EMAWeights)

        self.targetNetworkList[newTargetIndex] = targetnetwork
        self.actualTargetNetwork = newTargetIndex


    def get_valueFunction(self, next_state_action_array, next_camera):
      target1 = self.getTarget1Network().predict_on_batch([next_camera, next_state_action_array])
      target2 = self.getTarget2Network().predict_on_batch([next_camera,next_state_action_array])
      return np.minimum(target1, target2)

    

    def _get_cem_optimal_Action(self,state, camera):
      #print("CEM state", state)
 
      states = np.tile(state, (self.cem_num_samples,1))
      cameras = np.tile(camera, (self.cem_num_samples,1,1,1))
  
      #print("CEM STATES", states, states.shape)
      self.cem.reset()
      for i in range(self.cem_update_itr):
        actions = self.cem.sample_multi(self.cem_num_samples)
        actions = self.getActionPolicy(actions)
        #print("CEM ActiONS", actions, actions.shape)
        stateActionArray = self.getStateActionArray(states, actions)
        #print("CEM STATe Action Array", stateActionArray, stateActionArray.shape)
        q_values = self.getTarget1Network().predict_on_batch([cameras, stateActionArray])
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

    def getTarget1Network(self):
      return self.targetNetworkList[self.actualTargetNetwork]

    def getTarget2Network(self):
      return self.targetNetworkList[(self.actualTargetNetwork-self.target2Index)%len(self.targetNetworkList)]
          
    def get_Action(self, enviroment, state, camera, training):
        if training and (np.random.rand() <= self.epsilon and self.train_step<100):
            action = self.getActionFromEpsilonGreedyPolicy(enviroment)
            #print("Epsilon", action)
            return action
      
        optimal_action = self._get_cem_optimal_Action(state, self.getReshapedImg(camera))
        #print("CEM", optimal_action)
        return optimal_action
        
    def train(self ,states, actions, cameras, next_states, next_cameras, rewards, terminates, batch_size):
      loss = 0
      self.train_step += 1

      npTerminates = np.asarray(terminates)
      npRewards = np.asarray(rewards)
      npActions = np.asarray(actions)
      npStates = np.asarray(states)
      npCameras = np.asarray(cameras)
      npNext_Cameras = np.asarray(next_cameras)
      intTerminates = np.array(list(map(lambda y: [1- int(y)], npTerminates)))
    
       
      state_action_array  =  self.getStateActionArray(npStates, npActions)
      q_values = self.q_network.predict_on_batch([npCameras,state_action_array])

      #Sample Next_Actions FROM CEM
      next_actions_samples = []
      for i  in range(batch_size):
        next_actions_samples.append(self._get_cem_optimal_Action(next_states[i], npNext_Cameras[i]))

      next_actions = np.asarray(next_actions_samples)
      #print(next_actions)
      #print(next_states)
      next_state_action_array  =  self.getStateActionArray(next_states, next_actions)
    
      q_next = self.get_valueFunction(next_state_action_array, npNext_Cameras)

      q_target = np.copy(q_values)

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

      #print("Next", q_next[0])

      #print("Target Q", q_target[0])

      #print("Rward", npRewards[0])

      #print("Should Target", npRewards[0] + self.gamma * np.amax(q_next[0])*intTerminates[0] )

      #print("output Calc", npRewards + self.gamma* np.amax(q_next, axis=1)*intTerminates )
      
      #print("Full Target Q", q_target)
      #input("wait")

      self.updateLaggedNetworks()
      training_history = self.q_network.train_on_batch([npCameras,state_action_array], q_target)

      if self.train_step % 100 == 0:
        #print("update parameter")
        self.alighn_target_model()
     

    def retrain(self, batch_size):
        dataset  = self.replayBuffer.as_dataset(sample_batch_size = batch_size, num_steps=1)
        iterator = iter(dataset)
        (minibatch, prop) = next(iterator)
       
        states = minibatch[0]
        actions = minibatch[1]
        cameras = minibatch[2]
        next_states = minibatch[3]
        next_cameras = minibatch[4]
        rewards = minibatch[5]
        terminates = minibatch[6]

        cameraShapeList = list(self._imgReshapeWithDepth)
        cameraShapeList.insert(0, batch_size)
        cameraShape = tuple(cameraShapeList)
           
        states = tf.reshape(states, (batch_size, self._state_size))
        actions = tf.reshape(actions, (batch_size,self._action_size))
        cameras = tf.reshape(cameras, cameraShape)
        next_states = tf.reshape(next_states, (batch_size, self._state_size))
        next_cameras = tf.reshape(next_cameras, cameraShape)
        rewards = tf.reshape(rewards, (batch_size, 1))
        terminates = tf.reshape(terminates, (batch_size,1))


        self.train(states, actions, cameras, next_states, next_cameras, rewards, terminates, batch_size)
