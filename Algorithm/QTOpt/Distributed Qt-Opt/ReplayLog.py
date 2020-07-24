import numpy as np
import os

class ReplayLog:
    def __init__(self, dataPath, clientWrapper):
        self.clientWrapper = clientWrapper
        self.path = dataPath
        self.storeOfflinedata()

    def existPath(self):
        if not(os.path.exists(os.path.join(self.path, "state.npy")) or 
         os.path.exists(os.path.join(self.path, "action.npy")) or
         os.path.exists(os.path.join(self.path, "action.npy"))):
            return False
        else:
             return True



    def getData(self):
        
        if not self.existPath():
            print("Path does not exist")
            return None
       
        state = np.load(self.path+ "state.npy")
        action = np.load(self.path+ "action.npy")
        image = np.load(self.path+"image.npy")
        reward = np.load(self.path+"reward.npy")
        next_state = np.load(self.path+"next_state.npy")
        next_image = np.load(self.path+"next_image.npy")
        terminated = np.load(self.path+"terminated.npy")

        return state, action, image, reward, next_state, next_state, terminated

    def getMinSize(self, arrays):
            sizes = []
            for i in range(len(arrays)):
                    sizes.append(len(arrays[i]))
             
            return min(sizes)

    def storeOfflinedata(self):
        print("OfflineData will be loaded")
        state, action, image, reward, next_state, next_image, terminated = self.getData()
        print(action.shape, image.shape, reward.shape, next_state.shape, next_image.shape, terminated.shape)
        minSize = self.getMinSize([state, action, image, reward, next_state, next_image, terminated])
        print(minSize)
        if state is not None:

            for i in range(minSize):
                state_i = state[i]
                action_i = action[i]
                image_i = image[i]
                reward_i = reward[i][0]
                next_state_i = next_state[i]
                next_image_i = next_image[i]
                terminated_i = terminated[i][0]
            
                self.clientWrapper.storeOfflineData(state_i, action_i, image_i, 
                reward_i, next_state_i, next_image_i, terminated_i)
