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

class Agent:
    def __init__(self, enviroment, optimizer, loss,  action_space_policy, state_size= 4, action_size= 1,  cem_update_itr = 2, select_num = 6, num_samples = 64):
        
               
        self._state_size = state_size
        self._action_size = action_size
        self._networkOutputSize = 1
        
        self._optimizer = optimizer
        self._loss = loss


        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.epsilon = 0.3
        self.train_step = 0
        

        #TargetNetworks
        self.numTagetNetworks = 3
        self.targetNetworkList = [self._build_compile_model() for _ in range(self.numTagetNetworks)]
      
        self.actualTargetNetwork = 0
        self.target2Index = 2
        
        self.replayBufferMaxLength = 5000
        self.replyBufferBatchSize = 32
        # (s,a, S', r)
        data_spec = (tf.TensorSpec(self._state_size, tf.float64, 'state'),
        tf.TensorSpec(self._action_size, tf.float64, 'action'),
        tf.TensorSpec(self._state_size, tf.float64, 'next_state'),
        tf.TensorSpec(1, tf.float64, 'reward'),
        tf.TensorSpec(1, tf.bool, 'terminated'))
        
        self.replayBuffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, batch_size = 1, max_length=self.replayBufferMaxLength )
            
        #Init CEM
        self.cem_update_itr = cem_update_itr
        self.cem_select_num = select_num
        self.cem_num_samples = num_samples
        self.cem  = CEM(self._action_size)
        self.getActionPolicy = action_space_policy


        # Build networks
        self.q_network = self._build_compile_model()
        self.policy = self._build_compile_model()
        self.alighn_target_model()

  

    def store(self, state, action, reward, next_state, terminated, training):
        values = (state, tf.dtypes.cast(action, tf.float64), next_state, reward, terminated)
        nestedStructure = tf.nest.map_structure(lambda t: tf.stack([t]* 1),values)
        self.replayBuffer.add_batch(nestedStructure)

        if(training and self.replayBuffer.num_frames() > self.replyBufferBatchSize):
          self.retrain(self.replyBufferBatchSize)

    def getStateActionArray(self, state, action):
       return np.concatenate((action, state), axis=1) 
      
    def _build_compile_model(self):
        inputShape = self._state_size+self._action_size
        layerInput = keras.Input(shape=(inputShape,), name='q_input')
        x = layers.BatchNormalization(name="bacth_0")(layerInput)
        x = layers.Dense(20, activation='relu', name="dense_1")(x)
        x = layers.BatchNormalization(name="bacth_1")(x)
        x = layers.Dense(20, activation='relu', name="dense_2")(x)
        x = layers.BatchNormalization(name="bacth_2")(x)
        x = layers.Dense(20, activation='relu', name="dense_3")(x)
        x = layers.BatchNormalization(name="bacth_3")(x)
        output = layers.Dense(self._networkOutputSize,  activation='linear', name="dense_output")(x)
        model = keras.Model(inputs=layerInput, outputs = output)
        model.compile(loss=self._loss, optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        targetnetwork = self.targetNetworkList[self.actualTargetNetwork]
        targetnetwork.set_weights(self.q_network.get_weights())

        self.targetNetworkList[self.actualTargetNetwork] = targetnetwork
        self.actualTargetNetwork = (self.actualTargetNetwork+1)%self.numTagetNetworks


    def get_valueFunction(self, next_state_action_array):
      target1 = self.getTarget1Network().predict(next_state_action_array)
      target2 = self.getTarget2Network().predict(next_state_action_array)
      return np.minimum(target1, target2)

    
    def getActionPolicy(self,actions):
      def myfunc(a):
          return int(a>0.5)
      vfunc = np.vectorize(myfunc)
      return vfunc(actions)
    

    def _get_cem_optimal_Action(self,state):
      #print("CEM state", state)
      states = np.tile(state, (self.cem_num_samples,1))
      #print("CEM STATES", states, states.shape)
      self.cem.reset()
      for i in range(self.cem_update_itr):
        actions = self.cem.sample_multi(self.cem_num_samples)
        actions = self.getActionPolicy(actions)
        #print("CEM ActiONS", actions, actions.shape)
        stateActionArray = self.getStateActionArray(states, actions)
        #print("CEM STATe Action Array", stateActionArray, stateActionArray.shape)
        q_values = self.getTarget1Network().predict_on_batch(stateActionArray)
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


    def getTarget1Network(self):
      return self.targetNetworkList[self.actualTargetNetwork]

    def getTarget2Network(self):
      return self.targetNetworkList[(self.actualTargetNetwork-self.target2Index)%len(self.targetNetworkList)]
          
    def get_Action(self, enviroment, state, training):
        if training and (np.random.rand() <= self.epsilon and self.train_step<100):
            action = self.getActionFromEpsilonGreedyPolicy(enviroment)
            #print("Epsilon", action)
            return action
      
        optimal_action = self._get_cem_optimal_Action(state)
        #print("CEM", optimal_action)
        return optimal_action
        
    def train(self ,states, actions, next_states, rewards, terminates, batch_size):
      loss = 0
      self.train_step += 1

      npTerminates = np.asarray(terminates)
      npRewards = np.asarray(rewards)
      npActions = np.asarray(actions)
      npStates = np.asarray(states)
      intTerminates = np.array(list(map(lambda y: [1- int(y)], npTerminates)))
    
       
      state_action_array  =  self.getStateActionArray(npStates, npActions)
      q_values = self.q_network.predict(state_action_array)

      #Sample Next_Actions FROM CEM
      next_actions_samples = []
      for i  in range(batch_size):
        next_actions_samples.append(self._get_cem_optimal_Action(next_states[i]))

      next_actions = np.asarray(next_actions_samples)
      #print(next_actions)
      #print(next_states)
      next_state_action_array  =  self.getStateActionArray(next_states, next_actions)
    
      q_next = self.get_valueFunction(next_state_action_array)

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

      training_history = self.q_network.train_on_batch(state_action_array, q_target)

      if self.train_step % 100 == 0:
        #print("update parameter")
        self.alighn_target_model()
     

    def retrain(self, batch_size):
        dataset  = self.replayBuffer.as_dataset(sample_batch_size = batch_size, num_steps=1)
        iterator = iter(dataset)
        (minibatch, prop) = next(iterator)
       
        states = minibatch[0]
        actions = minibatch[1]
        next_states = minibatch[2]
        rewards = minibatch[3]
        terminates = minibatch[4]
           
        states = tf.reshape(states, (batch_size, self._state_size))
        actions = tf.reshape(actions, (batch_size,self._action_size))
        next_states = tf.reshape(next_states, (batch_size, self._state_size))
        rewards = tf.reshape(rewards, (batch_size, 1))
        terminates = tf.reshape(terminates, (batch_size,1))

        self.train(states, actions, next_states, rewards, terminates, batch_size)
