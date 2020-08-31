from getModel import getAgent

path_to_load = "test/reach"

agent = getAgent(path_to_load)
agent.loadWeights()