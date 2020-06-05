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



class Agent:
    def __init__(self, enviroment, optimizer, loss):
        
        # Initialize atributes
        self.state = {
          "high": enviroment.observation_space.high,
          "low": enviroment.observation_space.low,
          "size": enviroment.observation_space.high.size
        }
        self._state_size = enviroment.observation_space.high.size
        self._action_size = enviroment.action_space.n

        self._optimizer = optimizer
        self._loss = loss
        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.epsilon = 0.3
        self.train_step = 0
        self.movingAverage = 0.9999
        self.laggedq2 = 2

        self.numTagetNetworks = 5
        self.q1Function = pass
        self.q2function =  pass
        self.actualNetworkList = [self._build_compile_model() for _ in range(self.numTagetNetworks)]
        self.actualTargetNetwork = 0

        self.movingAverageTensorList = [self._build_compile_model().get_weights() for _ in range(self.numTagetNetworks)]
        self.actualMovingIndex = 0
        
        
        self.onPolicyReplayBufferMaxLength = 5000
        self.onPolicyReplayBufferBatchSize = 32

        self.offPolicyReplayBufferMaxLength = 5000
        self.offPolicyReplayBufferBatchSize = 32
        
        # (s,a, S', r)
        data_spec = (tf.TensorSpec(self._state_size, tf.float64, 'state'),
        tf.TensorSpec(1, tf.int32, 'action'),
        tf.TensorSpec(self._state_size, tf.float64, 'next_state'),
        tf.TensorSpec(1, tf.float32, 'reward'),
        tf.TensorSpec(1, tf.bool, 'terminated'))
        
        self.onPolicyReplay = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, batch_size = 1, max_length=self.replayBufferMaxLength )
        self.offPolicyReplay = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, batch_size = 1, max_length=self.replayBufferMaxLength )
            
        # Build networks
        self.q_network = self._build_compile_model()
        self.alighn_target_model()
        self.loadOffPolicyData()

    def store(self, state, action, reward, next_state, terminated, training):
        values = (state, action, next_state, reward, terminated)
        nestedStructure = tf.nest.map_structure(lambda t: tf.stack([t]* 1),values)
        self.replayBuffer.add_batch(nestedStructure)

        if(training and self.replayBuffer.num_frames() > self.replyBufferBatchSize):
          self.retrain(self.replyBufferBatchSize)
      
    def loadOffPolicyData():
      pass  

    def _build_compile_model(self):
        layerInput = keras.Input(shape=(self._state_size+ s self._action_size), name='q_input')
        x = layers.BatchNormalization()(layerInput)
        x = layers.Dense(20, activation='relu', name="dense_1")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(20, activation='relu', name="dense_2")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(20, activation='relu', name="dense_3")(x)
        x = layers.BatchNormalization()(x)
        output = layers.Dense(self._action_size,  activation='linear', name="dense_output")(x)
        model = keras.Model(inputs=layerInput, outputs = output)
        model.compile(loss=self._loss, optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        targetnetwork = self.targetNetworkList[self.actualTargetNetwork]
        targetnetwork.set_weights(self.q_network.get_weights())

        self.targetNetworkList[self.actualTargetNetwork] = targetnetwork
        self.actualTargetNetwork = (self.actualTargetNetwork+1)%self.numTagetNetworks

    
    def saveExpMovAver(self, actualNetwork, previousNetwork):
      actuaNetworkWeights = actualNetwork.get_weights()
      previousnetworkWeights = previousNetwork.get_weights()

      movAverageWeights = tf.math.scalar_mul(self.movingAverage, previousnetworkWeights)+ tf.math.scalar_mul(1-self.movingAverage, actuaNetworkWeights)+
      self.actualMovingIndex = (self.actualMovingIndex+1) % len(self.movingAverageTensorList)
      self.movingAverageTensorList[self.actualMovingIndex]  =  movAverageWeights



    
    def getValueFunction(state):
      q1NetworkWeights = self.movingAverageTensorList[self.actualMovingIndex]
      q2NetworkWeights = self.movingAverageTensorList[(self.actualMovingIndex- self.laggedq2)% len(self.movingAverageTensorList) ] 
      
      q1Network = self._build_compile_model()
      q1Network.set_weights(q1NetworkWeights)

      q2Network =  self._build_compile_model()
      q2Network.set_weights(q2NetworkWeights)
      
      min(self.getTargetOneNetwork().predict(state, ))

      v1 = q1Network.preq2dict(state)
      v1Action = np.argmax(v1, axis=1)

      q1 = q1Network.predict(state)
      q2 = q2Network.predict(state)

      q1Values = np

      

      

    
    def getTargetOneNetwork():
      return self.q_network

    def getTargetTwoNetwork():
      return self.q_network
    
    def getActualNetwork(self):
      return self.q_network

    def getActionFromEpsilonGreedyPolicy(self, enviroment):
       action = enviroment.action_space.sample()
       print("EpsilonGreedyAction:", action)
       return action


    def getTargetNetwork(self):
      return self.targetNetworkList[self.actualTargetNetwork]
          
    def get_Action(self, enviroment, state, training):
        if training and np.random.rand() <= self.epsilon:
            return self.getActionFromEpsilonGreedyPolicy(enviroment)
      
        q_values = self.getActualNetwork().predict(state)
        argmax = q_values[0].argmax()
        print("action", argmax)
        #print("input", state)
        #print("Action", argmax, "from", q_values )
        return argmax

    def train(self ,states, actions, next_states, rewards, terminates, batch_size):
      #print("train")
      loss = 0
      self.train_step += 1

      npTerminates = np.asarray(terminates)
      npRewards = np.asarray(rewards)
      npActions = np.asarray(actions)
      intTerminates = np.array(list(map(lambda y: [1- int(y)], npTerminates)))
      
       
      q_values = self.q_network.predict(states)
      q_next = self.q_network.predict(next_states)     

      q_target = np.copy(q_values)

      for i in range(batch_size):
        myTarget = npRewards[i] + self.gamma * np.amax(q_next[i])*intTerminates[i]
        if i == 0:
          pass
          print("my Target", myTarget) 
        
        q_target[i][npActions[i]] =  npRewards[i] + self.gamma * np.amax(q_next[i])*intTerminates[i]

      #print("Choosed Action", npActions[0])

      print("Values", q_values[0])
      if math.isnan(q_target[0][0]):
        input("stop")

      #print("Next", q_next[0])

      print("Target Q", q_target[0])

      #print("Rward", npRewards[0])

      #print("Should Target", npRewards[0] + self.gamma * np.amax(q_next[0])*intTerminates[0] )

      #print("output Calc", npRewards + self.gamma* np.amax(q_next, axis=1)*intTerminates )
      
      #print("Full Target Q", q_target)
      #input("wait")

      training_history = self.q_network.train_on_batch(states, q_target)

      if self.train_step % 100 == 0:
        print("update parameter")
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
        actions = tf.reshape(actions, (batch_size,1))
        next_states = tf.reshape(next_states, (batch_size, self._state_size))
        rewards = tf.reshape(rewards, (batch_size, 1))
        terminates = tf.reshape(terminates, (batch_size,1))

        self.train(states, actions, next_states, rewards, terminates, batch_size)
