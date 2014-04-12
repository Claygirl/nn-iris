# -*- coding: utf-8 -*-

import csv
import numpy as np
import sys

NEURONS_FIRST_LAYER = 5
NEURONS_SECOND_LAYER = 3
ALPHA = 1
ETA = 0.1

def code_answer():
    names = np.genfromtxt('bezdekIris.data', delimiter=',', dtype="string", usecols=(4))
    ans = np.zeros((names.shape[0], 3), dtype="int")
    
    for i in range(names.shape[0]):
        if names[i] == "Iris-setosa":
            ans[i][0] = 1
        elif names[i] == "Iris-versicolor":
            ans[i][1] = 1
        elif names[i] == "Iris-virginica":
            ans[i][2] = 1
    
    return ans

inputs = np.genfromtxt('bezdekIris.data', delimiter=',', usecols=(0, 1, 2, 3))
inputs = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)
layer1 = np.random.random((inputs.shape[1], NEURONS_FIRST_LAYER))
layer2 = np.random.random((NEURONS_FIRST_LAYER + 1, NEURONS_SECOND_LAYER))
answer = code_answer()

for i in range(inputs.shape[0]):
    net1 = np.dot(inputs[i], layer1)
    s = -1*ALPHA*net1
    out1 = 1 / (1 + np.exp(s))
    out1 = np.append(out1, [1], axis=1)
#    print "obrót: " + str(i)
#    print out1
    
    net2 = np.dot(out1, layer2)
    s = -1*ALPHA*net2
    out2 = 1 / (1 + np.exp(s))
#    print out2
  
    error = 0.5 * sum((out2 - answer[i]) ** 2)

    print "calosc: "
    print error

