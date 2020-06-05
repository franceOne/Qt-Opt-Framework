import gym

from IPython.display import clear_output
from time import sleep
import tensorflow as tf
import numpy as np



array = np.array([[0], [4]])

array2 = np.array([[2], [2]])


def myfunc(a):
    return int(a>0.5)
vfunc = np.vectorize(myfunc)

print(vfunc(array))