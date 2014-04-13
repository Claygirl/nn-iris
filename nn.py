# -*- coding: utf-8 -*-

import csv
import numpy as np
import sys

NEURONS_FIRST_LAYER = 4
NEURONS_SECOND_LAYER = 3
ALPHA = 0.5
BETA = 0.9
ETA = 0.1

def code_answer():
    names = np.genfromtxt('bezdekIris.data', delimiter=',', dtype="string", usecols=(4))
    ans = np.zeros((names.shape[0], 3), dtype="int")
    
    for i in range(names.shape[0]):
        if names[i] == "Iris-setosa":
            ans[i][2] = 1
        elif names[i] == "Iris-versicolor":
            ans[i][1] = 1
        elif names[i] == "Iris-virginica":
            ans[i][0] = 1
    
    return ans

def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

def train(momentum=False):    
    global weights1, weights2, inputs, answer
    prev1 = 0
    prev2 = 0
    for m in range(100):
        for i in range(inputs.shape[0]):
            net1 = np.dot(inputs[i], weights1)
            s1 = ALPHA*net1
            out1 = sigmoid(s1)
            
            net2 = np.dot(out1, weights2)
            s2 = ALPHA*net2
            out2 = sigmoid(s2)
            
            error2 = (answer[i] - out2) * sigmoid_gradient(s2)
            error1 = np.dot(error2, np.transpose(weights2)) * sigmoid_gradient(s1)
            
            if(momentum):
                delta1 = ETA * np.dot(error1[:,None], inputs[i][:,None].T) + BETA * prev1
                prev1 = delta1
            else:
                delta1 = ETA * np.dot(error1[:,None], inputs[i][:,None].T)
                
            weights1 = weights1 + delta1.T
            
            if(momentum):
                delta2 = ETA * np.dot(error2[:,None], out1[:,None].T) + BETA * prev2
                perv2 = delta2
            else:
                delta2 = ETA * np.dot(error2[:,None], out1[:,None].T)
            
            weights2 = weights2 + delta2.T  
            
def test():
    global weights1, weights2, inputs, answer
    for i in range(inputs.shape[0]):
        net1 = np.dot(inputs[i], weights1)
        s1 = ALPHA*net1
        out1 = sigmoid(s1)
        
        net2 = np.dot(out1, weights2)
        s2 = ALPHA*net2
        out2 = sigmoid(s2)
        
        error_all = np.sum(np.power(out2 - answer[i], 2))
        f.write("answer: " + str(answer[i]) + "\n")
        f.write("out: " + str(out2) + "\n")
        f.write("error: " + str(error_all) + "\n")
            
inputs = np.genfromtxt('bezdekIris.data', delimiter=',', usecols=(0, 1, 2, 3))
inputs = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)
weights1 = np.random.random((inputs.shape[1], NEURONS_FIRST_LAYER))
weights2 = np.random.random((NEURONS_FIRST_LAYER, NEURONS_SECOND_LAYER))
answer = code_answer()

f = open('output.txt', 'w')

test()

train()

f.write("Po uczeniu: " + "\n")

test()
    
f.close

    
    
