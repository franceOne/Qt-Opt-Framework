import tensorflow as tf
import numpy as np
import random
from IPython.display import clear_output
import progressbar
import os
from mujoco_py import GlfwContext


class DataCollector:
    def __init__(self, id, clientWrapper, agent, environment, action_space_policy, state_policy, reward_policy, path = "/data", cluster = True):
        self.agent = agent
        self.environment = environment
        self.clientWrapper = clientWrapper
        if path is not None:
            self.path = path + "_"+str(id)
        else:
            self.path = None
        self.id = id
        self.cluster = cluster
        
        
        #Init variables
               
        self.policyFunction = action_space_policy
        self.max_step_size = 100
        self.get_state = state_policy
        self.reward_policy = reward_policy
    

        self.target1Network = None
        self.updateTarget1Network()


        self.episode = 0
        self.minSize = None
      


    def start(self, lock, train = True, ):
        if not self.cluster:
            GlfwContext(offscreen=True)  # Create a window to init GLFW.
        self.collectData(train,lock)


    def getTarget1Network(self):
        if self.target1Network:
            return self.target1Network
        else:
            self.updateTarget1Network()
            return self.target1Network

    def updateTarget1Network(self):
        self.target1Network = self.agent.getTarget1Network()


    def loadNumpy(self, path):
        if not(os.path.exists(path)):
            print("Path does not exist", path)
            return None 
        loaded_file = np.load(path)
        return loaded_file

    def getPath(self, path, output_filename):
        homedir = os.path.expanduser("~")
        # construct the directory string
        dir_path = os.path.dirname(os.path.realpath(__file__))
        pathset = os.path.join(dir_path, path)
        path_to_store = os.path.join(pathset, output_filename)
        return path_to_store



    def safeRewards(self,path, data, output_filename="rewardsPerEpoch.npy"):

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

    


    



    def numpySave(self,path, output_filename, data):
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

        if self.minSize is not 0:
            oldData = oldData[0: self.minSize]
       
        newData = [data]
        if oldData is not None:
            newData = np.concatenate((oldData, [data]), axis= 0)
            #print(output_filename, "olddata", oldData.shape, "data", data.shape, oldData, data)
            #print("newData", newData)
        np.save(path_to_store, newData)

    def getMinSize(self, paths):
        if self.minSize is not None:
            return self.minSize
        else:
            sizes = []
            for i in range(len(paths)):
                data = self.loadNumpy(self.getPath(self.path, paths[i]))
                if data is not None:
                    sizes.append(len(data))
                else:
                    sizes.append(0)
            self.minSize = min(sizes)
            return self.minSize

    def storeData(self, state, action, image, reward, next_state, next_image, terminated):
        self.clientWrapper.storeOnlineData(state, action, image, reward, next_state, next_image, terminated)
        
        if self.path is not None:
            paths = ["state.npy", "action.npy", "image.npy", "reward.npy", "next_state.npy", "next_image.npy", "terminated.npy"]
            self.getMinSize(paths)
            self.numpySave(self.path, "state.npy", state)
            self.numpySave(self.path, "action.npy", action)
            self.numpySave(self.path, "image.npy", image)
            self.numpySave(self.path,"reward.npy", np.array([reward]))
            self.numpySave(self.path,"next_state.npy", next_state)
            self.numpySave(self.path,"next_image.npy", next_image)
            self.numpySave(self.path,"terminated.npy", np.array([terminated]))
                

 

    def collectData(self, train, lock):
        enviroment = self.environment
        i = 0
        print("Collect Data")
       
        # Begin new Episode
        while True:
            i +=1
            # Reset the enviroment
            state = self.environment.reset()
            state = self.get_state(state)

            # Initialize variables
            rewardSum = 0
            terminated = False
            step = 0
           
            bar = progressbar.ProgressBar(maxval=self.max_step_size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            
            if i % 10 == 0 or not train:
                self.updateTarget1Network()
                print("fetch new TargetNetwork")

            with lock:
                lastImage = enviroment.render(mode="rgb_array")
                if not self.cluster:
                    enviroment.render()
                camera = lastImage


            # Run Episode
            while not terminated:
              
                concatenatedImage = np.concatenate((lastImage, camera), axis=0)
                
                # Run Action
                action = self.agent.get_Action(enviroment, state, self.agent.getReshapedImg(concatenatedImage), train, self.getTarget1Network())
                action = self.policyFunction(action)

                # Take action    
                next_state, reward, terminated, info = enviroment.step(action)
                reward = self.reward_policy(next_state,reward)
                #print("Reward", reward)
                next_state = self.get_state(next_state)

                with lock:
                    next_camera  =  enviroment.render(mode="rgb_array")
                    if not self.cluster:
                        enviroment.render()

                next_concatenatedImage = np.concatenate((camera, next_camera), axis=0)
               
                self.storeData(state, action, self.agent.getReshapedImg(concatenatedImage), reward, next_state, self.agent.getReshapedImg(next_concatenatedImage), terminated)
                
                #Update Counter
                step += 1
                rewardSum += reward
                state = next_state
                lastImage = camera
                camera = next_camera
                bar.update(step)

                if terminated or step >= self.max_step_size:
                    bar.finish()
                    print("**********************************")
                    print("Episode {} Reward {}".format(self.episode, rewardSum))
                    self.safeRewards(self.path,rewardSum)
                    print("**********************************")
                    break

            self.episode += 1





