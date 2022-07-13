import numpy as np
from numpy import asarray, int8, load
from numpy import savetxt
from numpy import loadtxt

name='sequential_dense_MatMul'
# load array
data = load(name + '.npy')
# print the array

print(data)
np.savetxt(name + '.csv', data, delimiter=',', fmt='%d')