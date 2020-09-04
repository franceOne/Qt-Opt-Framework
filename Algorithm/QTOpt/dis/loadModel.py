from getModel import getAgent

path_to_load = "saved_model/Weights/fetch_reach/1000epochs"
path_to_load = "saved_model/Weights/fetch_reach/1000epochs"

agent = getAgent(path_to_load)
agent.loadWeights()