import csv
import numpy as np
import sys

inputs = np.genfromtxt('bezdekIris.data', delimiter=',', usecols=(0, 1, 2, 3))

print inputs

layer1 = np.random.random((4, 3))

ans = np.dot(inputs, layer1)
print ans