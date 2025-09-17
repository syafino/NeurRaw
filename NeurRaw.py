inputs = [1,2,3,2.5] 
#input into neurons
#weight = w


#Input int each neuron, each neuron has its own weights. and its own biases 
# w1 = [0.2,0.8,-0.5,1]
# w2 = [0.5,-0.91,0.26,-0.5]
# w3 = [-0.26,-0.27,0.17,0.87]

weights = [[0.2,0.8,-0.5,1], 
           [0.5,-0.91,0.26,-0.5], 
           [-0.26,-0.27,0.17,0.87]
           ] #turn into matrices

#Bias = b
# b1 = 2
# b2 = 3
# b3 = 0.5

biases = [2,3,0.5] #turn into matrices

#4 input into 3 neuron; each neuron has its own setting almost like a dj set, so every input to every neuron is unique.

# outputs = [ inputs[0]*w1[0] + inputs[1]*w1[1]+inputs[2]*w1[2]+inputs[3]*w1[3] + b1,
#            inputs[0]*w2[0] + inputs[1]*w2[1]+inputs[2]*w2[2]+inputs[3]*w2[3] + b2,
#            inputs[0]*w3[0] + inputs[1]*w3[1]+inputs[2]*w3[2]+inputs[3]*w3[3]
#            ]

# print(outputs)

#do dot product instead cz its faster
outputs = []
for neuron_weights, neuron_biases in zip(weights,biases):
    neuron_output = 0
    for n_input, n_weights in zip(inputs, neuron_weights):
        neuron_output += n_input * n_weights
    neuron_output += neuron_biases
    outputs.append(neuron_output)

print(outputs)