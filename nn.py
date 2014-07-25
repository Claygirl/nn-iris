# -*- coding: utf-8 -*-

import numpy as np

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


def train(inputs, answer, structure, weights):
    layers = len(structure) + 1

    if weights is None:
        weights = {}

        for w in range(layers):
            if w == 0:
                weights[w] = np.random.random((inputs.shape[1], structure[w]))
            elif w == len(structure + 1):
                weights[w] = np.random.random((structure[w - 1], answer.shape[1]))
            else:
                weights[w] = np.random.random((structure[w - 1], structure[w]))
    
    for m in range(100):
        prev = {}
        
        for i in range(inputs.shape[0]):
            s = {}
            delta = {}            
            error = {}  
            out = {}
        
            inn = inputs[i]
            for w in range(layers):
                net = np.dot(inn, weights[w])
                s[w] = ALPHA*net
                out[w] = sigmoid(s[w])
                inn = out[w]
            
            for w in reversed(range(layers)):
                if w == len(structure + 1):
                    error[w] = (answer[i] - out[w]) * sigmoid_gradient(s[w])
                else:
                    error[w] = np.dot(error[w + 1], np.transpose(weights[w + 1])) * sigmoid_gradient(s[w])
            
                prev[w] = 0
            
            
            for w in range(layers):
                if w == 0:
                    delta[w] = ETA * np.dot(error[w][:,None], inputs[i][:,None].T) + BETA * prev[w]
                else:
                    delta[w] = ETA * np.dot(error[w][:,None], out[w - 1][:,None].T) + BETA * prev[w]
                    
                prev[w] = delta[w]
                weights[w] = weights[w] + delta[w].T
    
    return weights
            
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

        if outcome != ans:
            error += 1

    return error
    
            
inputs = np.genfromtxt('bezdekIris.data', delimiter=',', usecols=(0, 1, 2, 3))
names = np.genfromtxt('bezdekIris.data', delimiter=',', dtype="string", usecols=(4))
inputs = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)
testset = np.hstack((inputs, np.atleast_2d(names).T))
np.random.shuffle(testset)
testset = np.split(testset, 5)

def crossvalidate(structure):
    sum = 0
    for i in range(5):
        weights = None
        test_inputs = testset[i][:, :5]
        test_inputs = test_inputs.astype(np.float32)
        test_answer = testset[i][:, 5]
        test_answer = code_answer(test_answer)

        for j in range(5):
            if i == j:
                continue
            train_inputs = testset[j][:, :5]
            train_inputs = train_inputs.astype(np.float32)
            train_answer = testset[j][:, 5]
            train_answer = code_answer(train_answer)
            weights = train(train_inputs, train_answer, structure, weights)
        
        error = test(test_inputs, test_answer, weights)
        f.write("Number of misclassifications in fold " + str(i) + ": " + str(error) + "\n")
        sum += error
        print weights

    f.write("Average: " + str(sum/5.0) + "\n")
    print "Average: " + str(sum/5.0) + "\n"

f = open('output.txt', 'w')

f.write("Without hidden layer (4 - 3): \n")
structure = np.array([4])
crossvalidate(structure)

f.write("\nWith hidden layer(4 - 5 - 3): \n")
structure = np.array([4, 5])
crossvalidate(structure)

f.write("\nWith hidden layer(6 - 5 - 3): \n")
structure = np.array([6, 5])
crossvalidate(structure)

f.close
