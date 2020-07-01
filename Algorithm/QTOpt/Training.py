import numpy as np
import random
from IPython.display import clear_output
import progressbar

#modelSrc  = "simulation/src/RLAlgorithm/Algorithm/QTOpt/saved_model/TEST"
#modelSrc = 'simulation/src/RLAlgorithm/Algorithm/QTOpt/saved_model/DQN'
modelSrc = 'simulation/src/RLAlgorithm/Algorithm/QTOpt/saved_model/TEST2'


modelSrcWeights=  'simulation/src/RLAlgorithm/Algorithm/QTOpt/saved_model/Weights/TEST2'



def train(enviroment, agent, policyFunction, observationsize = 4, num_of_episodes=100, train=True, maxStepSize = 50, loadModell = True, saveModell = False ):

    if loadModell:
        print("load model", modelSrcWeights)
        agent.loadWeights(modelSrcWeights)

   
    for e in range(0, num_of_episodes):
        # Reset the enviroment
       
        state = enviroment.reset()
        #state = np.reshape(state, [1,observationsize])

        # Initialize variables
        rewardSum = 0
        terminated = False
        step = 0
       
        bar = progressbar.ProgressBar(maxval=maxStepSize, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        camera = enviroment.render(mode="rgb_array")
        while not terminated:
            #print("step")
            
            
            if not train:
                enviroment.render()
            step += 1
            # Run Action
            action = agent.get_Action(enviroment,state, camera,  train)
            action = policyFunction(action)

            
            #print("POLICY ACTION", action)
            
            # Take action    
            next_state, reward, terminated, info = enviroment.step(action)
            next_camera  =  enviroment.render(mode="rgb_array")
            #print("is camera eq", np.array_equal(camera, next_camera))
            #print("action", action, "terminated", terminated, "reward", reward)
            #next_state = np.reshape(next_state, [1,observationsize]) 
            agent.store(state, action, camera, reward, next_state, next_camera, terminated, train)
            rewardSum += reward
            state = next_state
            camera = next_camera
            
            if terminated or step>=maxStepSize:
                print("Episode {} Reward {}".format(e, rewardSum))
                if train and saveModell:
                    agent.saveWeights(modelSrcWeights)
                    print("Save Modell", modelSrcWeights)
                break
                
            
            bar.update(step)
        
        bar.finish()
        print("**********************************")
        print("Episode: {}".format(e + 1))
        print("**********************************")
    print("save model")
    agent.saveModel()

