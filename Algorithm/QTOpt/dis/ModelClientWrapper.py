

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
from time import sleep

class ModelClient:
    def __init__(self, server_host):
        self.server_host = server_host


    def storeQNetwork(self, weights):
        url = "http://"+self.server_host+"/storeqnetwork"

        weights = np.asarray(weights).tolist() 

        for i in range(len(weights)):
            weights[i] = weights[i].tolist() 
      
        data = {'qnetwork': np.asarray(weights).tolist()  }
        #
        print("store qnetwork", len(weights), weights[0])
        print("STOOOREE QNETWORK")
        sleep(0.05)
        #
        try:
            pass
            requests.post(url, json=data)
            #
        except Exception as e:
            print("error store QNetwork", e)

    def updateNetworkByGradient(self, gradients):
        url = "http://"+self.server_host+"/updatenetwork"

        gradients = np.asarray(gradients).tolist() 

        for i in range(len(gradients)):
            gradients[i] = gradients[i].tolist() 
      
        data = {'qradients': np.asarray(gradients).tolist()  }
        #
        print("store gradients", len(gradients))
        #
        try:
            requests.post(url, json=data)
            #
        except Exception as e:
            print("error store QNetwork", e)


    def getQNetwork(self):
        url = "http://"+self.server_host+"/getqnetwork"
            #print("\nrecieving array to", url)
        try:
            resp = requests.get(url).json()
            #print(resp.keys())
            qnetwork = resp['qnetwork']

            for i in range(len(qnetwork)):
                qnetwork[i] = np.asarray(qnetwork[i])

            qnetwork = np.asarray(qnetwork)
            return qnetwork
        except:
            print("Error fetching file")

    def getTargetNetworks(self):
        url = "http://"+self.server_host+"/gettargetnetworks"
            #print("\nrecieving array to", url)
        try:
            resp = requests.get(url).json()
            #print(resp.keys())
            target1 = resp['target1']
            target2 = resp['target2']

            for i in range(len(target1)):
                target1[i] = np.asarray(target1[i])

            target1 = np.asarray(target1)

            for i in range(len(target2)):
                target2[i] = np.asarray(target2[i])

            target2 = np.asarray(target2)
            
            return target1, target2
        except:
            print("Error fetching file")
            return None,None

