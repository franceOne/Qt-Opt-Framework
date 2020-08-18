from getModel import getAgent

path_to_load = "test/save_model"

agent = getAgent(path_to_load)
agent.loadWeights()