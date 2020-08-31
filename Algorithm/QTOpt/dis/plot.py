import numpy as np
import os
import matplotlib.pyplot as plt

name = 'Reward/Fetch_reach__test/FullState_0'
dataCollectionPath = 'saved_model/data/'+name
file_name = dataCollectionPath+ "/rewardsPerEpoch.npy"


def loadNumpy(path):
    if not(os.path.exists(path)):
        print("Path does not exist", path)
        return None 
    loaded_file = np.load(path)
    return loaded_file

data_to_plot = loadNumpy(file_name)


if data_to_plot is not None:
    plt.plot(data_to_plot)
    plt.show()
