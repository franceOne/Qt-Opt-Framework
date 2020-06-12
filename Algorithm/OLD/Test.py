import gym

from IPython.display import clear_output
from time import sleep
import tensorflow as tf
import numpy as np



array = np.array([[0,1], [4,2]])

array2 = np.array([[2,1], [2,2]])


def myfunc(a):
    return int(a>0.5)
vfunc = np.vectorize(myfunc)

print(0.99*array + 0.01*array2)