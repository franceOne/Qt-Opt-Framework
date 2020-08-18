from getModel import getAgent

path_to_save = "test/save_model"

agent = getAgent(path_to_save)
agent.getQNetwork()