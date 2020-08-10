import gym
from mujoco_py import GlfwContext

env = gym.make('FetchPickAndPlace-v1')
env.reset()
GlfwContext(offscreen=True)  # Create a window to init GLFW.

for _ in range(1000):
    #env.render()
    #env.render(mode="rgb_array")
    print(env.render(mode="rgb_array"))
    env.render()
    state, reward, terminated, info  = env.step(env.action_space.sample()) # take a random action
    #print(state)
    print( reward)
env.close()