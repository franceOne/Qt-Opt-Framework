import numpy as np
import random
from IPython.display import clear_output
import progressbar




def train(enviroment, agent, policyFunction, batch_size=32, num_of_episodes=100, ):
    for e in range(0, num_of_episodes):
        # Reset the enviroment
        state = enviroment.reset()
        state = np.reshape(state, [1,4])
              
        
        # Initialize variables
        rewardSum = 0
        terminated = False
       
        bar = progressbar.ProgressBar(maxval=200, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        while not terminated:
            # Run Action
            action = agent.get_Action(enviroment,state, True)
            action = policyFunction(action)
            #enviroment.render()
            
            # Take action    
            next_state, reward, terminated, info = enviroment.step(action)
            print("action", action, "terminated", terminated, "reward", reward)
            next_state = np.reshape(next_state, [1,4]) 
            agent.store(state, action, reward, next_state, terminated, True)
            rewardSum += reward
           
            
            state = next_state
            
            if terminated:
                agent.alighn_target_model()
                print("Episode {} Reward {}".format(e, rewardSum))
                break
                
            
            bar.update(rewardSum if rewardSum < 200 else 200)
        
        bar.finish()
        print("**********************************")
        print("Episode: {}".format(e + 1))
        print("**********************************")


def runModel(enviroment, agent, policyFunction, num_of_episodes=100, ):
    variable = input('Bestätige Run des Modells!: ')
    for e in range(0, num_of_episodes):
        # Reset the enviroment
        state = enviroment.reset()
        state = np.reshape(state, [1,4])
              
        # Initialize variables
        rewardSum = 0
        terminated = False
        
        bar = progressbar.ProgressBar(maxval=200, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        while not terminated:
            # Run Action
            action = agent.get_Action(enviroment,state, False)
            enviroment.render()
                       
            # Take action    
            next_state, reward, terminated, info = enviroment.step(action)
            next_state = np.reshape(next_state, [1,4]) 
            agent.store(state, action, reward, next_state, terminated, False)
            
            state = next_state
            rewardSum += reward
            
            if terminated:
                break
            
            bar.update(rewardSum if rewardSum < 200 else 200)
        
        bar.finish()
        print("**********************************")
        print("Episode: {} Reward: {}".format((e + 1), rewardSum))
        enviroment.render()
        print("**********************************")
