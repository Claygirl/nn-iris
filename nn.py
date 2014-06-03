# -*- coding: utf-8 -*-

import csv
import numpy as np
import sys

NEURONS_INPUT_LAYER = 4
NEURONS_HIDDEN_LAYER = 0
ALPHA = 0.5
BETA = 0.9
ETA = 0.05

def code_answer(names):
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

def train(inputs, answer):
    if(NEURONS_HIDDEN_LAYER > 0):
        weights1 = np.random.random((inputs.shape[1], NEURONS_INPUT_LAYER))
        weights2 = np.random.random((NEURONS_INPUT_LAYER, NEURONS_HIDDEN_LAYER))
        weights3 = np.random.random((NEURONS_HIDDEN_LAYER, answer.shape[1]))
    else:
        weights1 = np.random.random((inputs.shape[1], NEURONS_INPUT_LAYER))
        weights2 = np.random.random((NEURONS_INPUT_LAYER, answer.shape[1]))
    
    prev1 = 0
    prev2 = 0
    prev3 = 0
    
    for m in range(1000):
        for i in range(inputs.shape[0]):
            net1 = np.dot(inputs[i], weights1)
            s1 = ALPHA*net1
            out1 = sigmoid(s1)
            
            net2 = np.dot(out1, weights2)
            s2 = ALPHA*net2
            out2 = sigmoid(s2)
            
            if(NEURONS_HIDDEN_LAYER > 0):
                net3 = np.dot(out2, weights3)
                s3 = ALPHA*net3
                out3 = sigmoid(s3)
            
            if(NEURONS_HIDDEN_LAYER > 0):
                error3 = (answer[i] - out3) * sigmoid_gradient(s3)
                error2 = np.dot(error3, np.transpose(weights3)) * sigmoid_gradient(s2)
                error1 = np.dot(error2, np.transpose(weights2)) * sigmoid_gradient(s1)
            else:
                error2 = (answer[i] - out2) * sigmoid_gradient(s2)
                error1 = np.dot(error2, np.transpose(weights2)) * sigmoid_gradient(s1)
            
            
            delta1 = ETA * np.dot(error1[:,None], inputs[i][:,None].T) + BETA * prev1
            
            prev1 = delta1
                
            weights1 = weights1 + delta1.T
                
            delta2 = ETA * np.dot(error2[:,None], out1[:,None].T) + BETA * prev2
            
            prev2 = delta2
            
            weights2 = weights2 + delta2.T  
            
            if(NEURONS_HIDDEN_LAYER > 0):
                delta3 = ETA * np.dot(error3[:,None], out2[:,None].T) + BETA * prev3
                
                prev3 = delta3
                
                weights3 = weights3 + delta3.T  
                
    if(NEURONS_HIDDEN_LAYER > 0):
        return [weights1, weights2, weights3]
    else:
        return [weights1, weights2]
            
def test(inputs, answer, weights):
    error = 0
    for i in range(inputs.shape[0]):
        inn = inputs[i]
        for j in range(len(weights)):
            net = np.dot(inn, weights[j])
            s = ALPHA*net
            out = sigmoid(s)
            inn = out
            
        outcome = np.argmax(out)    
        ans = np.argmax(answer[i])

        if(outcome != ans):
            error += 1

    return error
    
            
inputs = np.genfromtxt('bezdekIris.data', delimiter=',', usecols=(0, 1, 2, 3))
names = np.genfromtxt('bezdekIris.data', delimiter=',', dtype="string", usecols=(4))
inputs = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)
folds = np.hstack((inputs, np.atleast_2d(names).T))
np.random.shuffle(folds)
folds = np.split(folds, 5)

def crossvalidate():
    sum = 0
    for i in range(5):
        inputs = folds[i][:, :5]
        inputs = inputs.astype(np.float32)
        answer = folds[i][:, 5]
        answer = code_answer(answer)
        weights = train(inputs, answer)
        error = test(inputs, answer, weights)
        f.write("Number of misclassifications in fold " + str(i) + ": " + str(error) + "\n")
        sum += error
    
    f.write("Average: " + str(sum/5.0) + "\n")

f = open('output.txt', 'w')

f.write("Without hidden layer (4 - 3): \n")
crossvalidate()

f.write("\nWith hidden layer(4 - 5 - 3): \n")
NEURONS_HIDDEN_LAYER = 5
crossvalidate()

f.write("\nWith hidden layer(6 - 5 - 3): \n")
NEURONS_INPUT_LAYER = 6
NEURONS_HIDDEN_LAYER = 5
crossvalidate()
    
f.close