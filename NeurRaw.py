import numpy as np
inputs = [1,2,3,2.5] 
weights = [[0.2,0.8,-0.5,1], 
           [0.5,-0.91,0.26,-0.5], 
           [-0.26,-0.27,0.17,0.87]
           ] #turn into matrices
biases = [2,3,0.5] #turn into matrices

#4 input into 3 neuron; each neuron has its own setting almost like a dj set, so every input to every neuron is unique.

#do dot product instead cz its faster
outputs = np.dot(weights,inputs) + biases

print(outputs)