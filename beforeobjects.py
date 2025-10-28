# THIS FILE ISNT USED, MOVED TO NeurRaw.py FOR BETTER ORGANIZATION

import numpy as np
inputs = [[1,2,3,2.5],
          [2,5,-1,2],
          [-1.5,2.7,3.3,-0.8] 
          ] #this is a 4x3, weights is a 4x3. Row and column must match in dot product. 
            # 4 != 3. so we need to transpose weight (swap column with row) by np.array(weights).T in the outputs
        
weights = [[0.2,0.8,-0.5,1], 
           [0.5,-0.91,0.26,-0.5], 
           [-0.26,-0.27,0.17,0.87]
           ] #3 neurons, 3 weights and biases

biases = [2,3,0.5]

#4 input into 3 neuron; each neuron has its own setting almost like a dj set, so every input to every neuron is unique.

#2nd hidden layer of neurons
weights2 = [[0.1,-0.14,0.5], 
           [-0.5,0.12,-0.33], 
           [-0.44,0.73,-0.13]
           ] #because the layer 1 has 3 neuron output, the input of layer 2 neurons is 3 inputs. therefore 3x3 is logical.

biases2 = [-1,2,-0.5]

layer1_outputs = np.dot(inputs,np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs,np.array(weights2).T) + biases2

print(layer1_outputs)
print(layer2_outputs)