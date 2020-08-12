

################################################################################
### client.py
################################################################################


#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Dummy client, interacts with the server sending and receiving
compressed numpy arrays.
Run:
python client.py
"""


from __future__ import print_function
import io
import numpy as np
import zlib
import requests

class Client:
    def __init__(self, server_host):
        self.server_host = server_host
        #self.getOnlineBuffer(32)



    def storeOnlineData(self, state, action, camera, reward, next_state, next_camera, terminated):
        url = "http://"+self.server_host+"/storeonlinedata"

        #print("State",type(state) , " \n Action", type(action),  "Camerat",type(camera), "\n Reward", type(reward), "Next_states,..", type(next_state), type(next_camera), "\n Term", type(terminated))
      
        data = {'state': np.asarray(state).tolist(), 'action': np.asarray(action).tolist(), 
        'camera': np.asarray(camera).tolist(), 'next_camera': np.asarray(next_camera).tolist(), 'reward': reward.item(), 'next_state': np.asarray(next_state).tolist(), 'terminated': terminated  }
        #
        #print("\nstoreOnlineData to", url)
        #
        try:
            requests.post(url, json=data)
            #
        except Exception as e:
            print("error store onlinedata", e)

    def storeOfflineData(self, state, action, camera, reward, next_state, next_camera, terminated):
        url = "http://"+self.server_host+"/storeofflinedata"

        #print(type(reward[0]), terminated)
      
        data = {'state': np.asarray(state).tolist(), 'action': np.asarray(action).tolist(), 
        'camera': np.asarray(camera).tolist(), 'next_camera': np.asarray(next_camera).tolist(), 'reward': reward.tolist(), 'next_state': np.asarray(next_state).tolist(), 'terminated': terminated.tolist()  }
        #
        #print("\nstoreOnlineData to", url)
        #
        try:
            requests.post(url, json=data)
            #
        except Exception as e:
            print("error store storeofflinedata", e)
      

    def getOnlineDataSize(self):
        url = "http://"+self.server_host+"/getOnlineBufferSize"
        #print("\nrecieving array to", url)
        try:
            resp = requests.get(url).text
            print("getOnlineBufferSize", resp)
            return int(resp)
        except:
            print("Error fetching file")

    def getTrainDataSize(self):
        url = "http://"+self.server_host+"/getTrainingBufferSize"
        #print("\nrecieving array to", url)
        try:
            resp = requests.get(url).text
            print("getTrainingBufferSize", resp)
            return int(resp)
        except:
            print("Error fetching file")


    def getOnlineBuffer(self, batch_size):
        url = "http://"+self.server_host+"/getOnlineBuffer/"+str(batch_size)
            #print("\nrecieving array to", url)
        try:
            resp = requests.get(url).json()
            #print(resp.keys())
            state = np.array(resp['state'])
            action = np.array(resp['action'])
            camera = np.array(resp['camera'])
            reward = np.array(resp['reward'])
            next_state = np.array(resp['next_state'])
            next_camera = np.array(resp['next_camera'])
            terminated = resp['terminated']
            print("get Onlinebuffer", state.shape, "...")
            return state, action, camera, next_state, next_camera, reward, terminated
        except:
            print("Error fetching file")
            return None, None, None, None, None, None, None

    def storeTrainBuffer(self, state, action, camera, reward, next_state, next_camera, terminated, q_target, batch_size):
        url = "http://"+self.server_host+"/storetraindata/"+str(batch_size)

        data = {'state': np.asarray(state).tolist(), 'action': np.asarray(action).tolist(), 
        'camera': np.asarray(camera).tolist(), 'next_camera': np.asarray(next_camera).tolist(), 'reward': np.asarray(reward).tolist(), 'next_state': np.asarray(next_state).tolist(), 'terminated': np.asarray(terminated).tolist(),
        'q_target':  np.asarray(q_target).tolist() }
        #
        print("\store Traindata", url)
        #print(state.shape, camera.shape, reward, next_state.shape, next_camera.shape, terminated)
        #
        try:
            requests.post(url, json=data)
            print("TrainData Stored")
        except Exception as e:
            print("error store traindata", e)

    def getTrainBuffer(self, batch_size):
        url = "http://"+self.server_host+"/getTrainBuffer/"+str(batch_size)
            #print("\nrecieving array to", url)
        try:
            resp = requests.get(url).json()
            #print(resp.keys())
            states = np.array(resp['state'])
            actions = np.array(resp['action'])
            cameras = np.array(resp['camera'])
            rewards = np.array(resp['reward'])
            next_states = np.array(resp['next_state'])
            next_cameras = np.array(resp['next_camera'])
            terminates = np.array(resp['terminated'])
            q_target = np.array(resp['q_value'])
            print("get trainBuffer", states.shape, "...")
            return states, actions, cameras, next_states, next_cameras, rewards, terminates, q_target
        except:
            print("Error fetching file")
            return None, None, None, None, None, None, None, None
                



# ## CONFIG

SERVER_HOST= "localhost"
SERVER_PORT = 5000







