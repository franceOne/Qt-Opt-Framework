import numpy as np
import random
from IPython.display import clear_output
import progressbar




def train(enviroment, agent, policyFunction, observationsize = 4, batch_size=32, num_of_episodes=100, train=True, maxStepSize = 50 ):

    if  not train:
        input("Run des modells")
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
        
        while not terminated:
            camera = enviroment.render(mode="rgb_array")
            print("Camera", camera.shape[0])
            if not train:
                enviroment.render()
            step += 1
            # Run Action
            action = agent.get_Action(enviroment,state, camera,  train)
            action = policyFunction(action)

            
            #print("POLICY ACTION", action)
            
            # Take action    
            next_state, reward, terminated, info = enviroment.step(action)
            #print("action", action, "terminated", terminated, "reward", reward)
            #next_state = np.reshape(next_state, [1,observationsize]) 
            agent.store(state, action, reward, next_state, terminated, train)
            rewardSum += reward
            state = next_state
            
            if terminated or step>=maxStepSize:
                print("Episode {} Reward {}".format(e, rewardSum))
                if not train:
                    break
                else:
                    agent.alighn_target_model()
                    break
                
            
            bar.update(step)
        
        bar.finish()
        print("**********************************")
        print("Episode: {}".format(e + 1))
        print("**********************************")

