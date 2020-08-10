from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import tensorflow as tf
import numpy as np
import random
import math

class ReplayBuffer:
    def __init__(self,  state_size= 4, action_size= 1, camerashape = (500,500,3)):
        self._state_size = state_size
        self._action_size = action_size
        self._camerashape = camerashape
        _imgReshapeSize = 16
        self._imgReshape = (_imgReshapeSize,_imgReshapeSize)
        cameraShapeList = list(self._camerashape)
        cameraShapeList[0] = _imgReshapeSize
        cameraShapeList[1] = _imgReshapeSize
        self._imgReshapeWithDepth =  cameraShape = tuple(cameraShapeList)

        self.onlineBufferMaxLengh = 5000
        self.trainBufferMaxLength = 5000
        self.offlineBuffferMaxLength = 10000


        self.onlineBuffer = self.initOnlineAndOfflineBuffers("online")
        self.offlineBuffer =  self.initOnlineAndOfflineBuffers("offline")
        self.trainBuffer = self.initTrainBuffer()


    def storeTrainBuffer(self, state, action, camera, reward, next_state, next_camera, terminated, q_target_value, batch_size):
        for i in range(batch_size):
        
            stateData = state[i]
            actionData = action[i]
            cameraData = camera[i]
            rewardData = reward[i]
            next_stateData = next_state[i]
            next_cameraData = next_camera[i]
            terminatedData = terminated[i]
            q_target_valueData = q_target_value[i]

            values = (stateData, tf.dtypes.cast(actionData, tf.float64), tf.dtypes.cast(cameraData, tf.float32), next_stateData, tf.dtypes.cast(next_cameraData, tf.float32), rewardData, terminatedData, tf.dtypes.cast(q_target_valueData,tf.float32))
            nestedStructure = tf.nest.map_structure(lambda t: tf.stack([t]* 1),values)
            self.trainBuffer.add_batch(nestedStructure)

    def storeOnlineData(self,  state, action, camera, reward, next_state, next_camera, terminated):
        
        #(s,a,c, s', c', r t)
        values = (state, tf.dtypes.cast(action, tf.float64), tf.dtypes.cast(camera, tf.float32), next_state, tf.dtypes.cast(next_camera, tf.float32), tf.dtypes.cast(reward, tf.float64), terminated)
        
        #print(values, "VALUES") 
        nestedStructure = tf.nest.map_structure(lambda t: tf.stack([t]* 1),values)
        self.onlineBuffer.add_batch(nestedStructure)
      

    def storeOfflineData(self,  state, action, camera, reward, next_state, next_camera, terminated):
        
        #(s,a,c, s', c', r t)
        values = (state, tf.dtypes.cast(action, tf.float64), tf.dtypes.cast(camera, tf.float32), next_state, tf.dtypes.cast(next_camera, tf.float32), tf.dtypes.cast(reward, tf.float64), terminated)
        nestedStructure = tf.nest.map_structure(lambda t: tf.stack([t]* 1),values)
        self.offlineBuffer.add_batch(nestedStructure)
       


    def getOnlineBufferSize(self):
        return  self.onlineBuffer.num_frames()

    def getOfflineBufferSize(self):
        return  self.offlineBuffer.num_frames()

    def getTrainBufferSize(self):
        return  self.trainBuffer.num_frames()
    
    def initOnlineAndOfflineBuffers(self, bufferType="online"):
         # (s,a, S', r)
        data_spec = (tf.TensorSpec(self._state_size, tf.float64, 'state'),
        tf.TensorSpec(self._action_size, tf.float64, 'action'),
        tf.TensorSpec(self._imgReshapeWithDepth, tf.float32, 'camera'),
        tf.TensorSpec(self._state_size, tf.float64, 'next_state'),
        tf.TensorSpec(self._imgReshapeWithDepth, tf.float32, 'next_camera'),
        tf.TensorSpec(1, tf.float64, 'reward'),
        tf.TensorSpec(1, tf.bool, 'terminated'))
        if(bufferType == "online"):
            return tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, batch_size = 1, max_length=self.onlineBufferMaxLengh )
        elif(bufferType == "offline"):
            return tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, batch_size = 1, max_length=self.offlineBuffferMaxLength )
        else:
            print("Error init buffer")


    def getTrainBuffer(self, batch_size):
        dataset  = self.trainBuffer.as_dataset(sample_batch_size = batch_size, num_steps=1)
        #print("before train")
        iterator = iter(dataset)
        (minibatch, prop) = next(iterator)
       
        states = minibatch[0]
        actions = minibatch[1]
        cameras = minibatch[2]
        next_states = minibatch[3]
        next_cameras = minibatch[4]
        rewards = minibatch[5]
        terminates = minibatch[6]
        q_values = minibatch[7]

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
        q_values = tf.reshape(q_values, (batch_size,1))

        return (states, actions, cameras, next_states, next_cameras, rewards, terminates, q_values)



    
    def getOnlineBuffer(self, batch_size):
        dataset  = self.onlineBuffer.as_dataset(sample_batch_size = batch_size, num_steps=1)
        #print("before train")
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

        return (states, actions, cameras, next_states, next_cameras, rewards, terminates)

    def getOfflineBuffer(self, batch_size):
        dataset  = self.offlineBuffer.as_dataset(sample_batch_size = batch_size, num_steps=1)
        #print("before train")
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

        return (states, actions, cameras, next_states, next_cameras, rewards, terminates)





    
    def initTrainBuffer(self):
         # (s,a, S', r, q_value)
        data_spec = (tf.TensorSpec(self._state_size, tf.float64, 'state'),
        tf.TensorSpec(self._action_size, tf.float64, 'action'),
        tf.TensorSpec(self._imgReshapeWithDepth, tf.float32, 'camera'),
        tf.TensorSpec(self._state_size, tf.float64, 'next_state'),
        tf.TensorSpec(self._imgReshapeWithDepth, tf.float32, 'next_camera'),
        tf.TensorSpec(1, tf.float64, 'reward'),
        tf.TensorSpec(1, tf.bool, 'terminated'),
        tf.TensorSpec(1, tf.float32, 'q_value'))
        return tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, batch_size = 1, max_length=self.trainBufferMaxLength )


