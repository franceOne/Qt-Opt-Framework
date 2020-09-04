import numpy as np
import os
import matplotlib.pyplot as plt

name = 'fetch_reach/1000epochs/'
dataCollectionPath = 'saved_model/data/'+name
file_name = dataCollectionPath+ "/loss.npy"

file_name = "saved_model/data/fetch_reach/1000epochs_0/rewardsPerEpoch.npy"


def loadNumpy(path):
    if not(os.path.exists(path)):
        print("Path does not exist", path)
        return None 
    loaded_file = np.load(path)
    return loaded_file

data_to_plot = loadNumpy(file_name)
print(data_to_plot)

x = [i for i in range(data_to_plot.shape[0])]


if data_to_plot is not None:
    plt.plot(data_to_plot)
    plt.show()
