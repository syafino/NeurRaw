import numpy as np
import nnfs

nnfs.init() #initializes the random values.
# X = [[1,2,3,2.5],
#     [2,5,-1,2],
#     [-1.5,2.7,3.3,-0.8] 
#     ] 

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

class Activation_ReLU: #Rectified linear function (if its below 0 , it outputs 0, if its above 0 it outputs its usual value)
    def forward(self, inputs):
        self.output = np.maximum(0,inputs) 
        #numpy maximum compares first param with second param and maximum is outputed.
        #therefore if 0 is compared with negative or 0 value, it will just output 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True)) #exponentiate the outputs, max = 0, to prevent overflow.
        probabilities = exp_values/np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilities

# https://cs231n.github.io/neural-networks-case-study/
#This creates a dataset that resembles a spiral
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Loss:
    def calculate(self,output,y): #output is the values from neural network
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class lcce(Loss): #Loss categorical cross entropy 
    def forward(self, y_pred, y_true): #y_pred is value from neural network
        samples = len(y_pred) #y_true is the target values from one hot encoding
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: #target values is in 1d array
            correct_confidences = y_pred_clipped[range(samples), y_true] #range(samples) = the columns, row that gets inputted is from y_true
        elif len(y_true.shape) == 2: #target values is in 2d array
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
        neg_log_likelihoods = -np.log(correct_confidences)

        return neg_log_likelihoods


 

X,y = create_data(100,3)
layer1 = Layer_Dense(2,3) #2 inputs because x and y
activation1 = Activation_ReLU()

layer2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

layer1.forward(X) #the dot product + biases
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])

loss_function = lcce()
loss = loss_function.calculate(layer2.output, y)

print("loss: ", loss)


