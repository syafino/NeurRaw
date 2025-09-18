import numpy as np

np.random.seed(0)

X = [[1,2,3,2.5],
    [2,5,-1,2],
    [-1.5,2.7,3.3,-0.8] 
    ] 

#initializing a neural network = initializing weights and biases, usually random numbers between (-1,1). 
#we want small values so the values tend to not explode into very large numbers 
#weights are usually (-0.1,0.1) and biases are usually 0, but when like the weights and inputs are very small, 
#it wont change a thing even after hundreds of layers, so maybe not 0. 
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons) 
        #were making the column match the row of inputs, so we dont need to transpose the matrix
        self.biases = np.zeros((1,n_neurons)) #creates an 1d array of 0s
        #1,n_inputs is the shape. 1 dimensional, n_inputs index.
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

layer1 = Layer_Dense(4,5) #4 inputs, 5 neurons, 5 outputs
layer2 = Layer_Dense(5,2) #5 inputs (to match layer1), 2 neurons, 2 outputs

layer1.forward(X)
print(layer1.output)
print("l")
layer2.forward(layer1.output)
print (layer2.output)

