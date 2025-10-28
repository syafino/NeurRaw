import numpy as np
from NeurRaw import Layer_Dense, Activation_ReLU, Activation_Softmax, lcce

class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate  
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
            
        # binary mask of random values
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
            
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
            
        layer.weights += weight_updates
        layer.biases += bias_updates
        
    def post_update_params(self):
        self.iterations += 1

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        
    def add(self, layer):
        self.layers.append(layer)
        
    def set_loss(self, loss):
        self.loss = loss
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def forward(self, X, training=True):
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                if 'training' in layer.forward.__code__.co_varnames:
                    layer.forward(X, training)
                else:
                    layer.forward(X)
                X = layer.output
        return X
        
    def train(self, X, y, epochs=1000, print_every=100):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X, training=True)
            
            # Calculate loss
            data_loss = self.loss.calculate(output, y)
            
            # Print progress
            if epoch % print_every == 0:
                print(f'Epoch {epoch}, Loss: {data_loss:.4f}')
                
        return data_loss
        
    def predict(self, X):
        return self.forward(X, training=False)

# Example: Enhanced network with dropout
def create_enhanced_network():
    network = NeuralNetwork()
    
    # Add layers
    network.add(Layer_Dense(2, 64))
    network.add(Activation_ReLU())
    network.add(Layer_Dropout(0.1))
    
    network.add(Layer_Dense(64, 64))
    network.add(Activation_ReLU())
    network.add(Layer_Dropout(0.1))
    
    network.add(Layer_Dense(64, 3))
    network.add(Activation_Softmax())
    
    # Set loss and optimizer
    network.set_loss(lcce())
    network.set_optimizer(Optimizer_SGD(learning_rate=0.05, decay=5e-7, momentum=0.9))
    
    return network

if __name__ == "__main__":
    from NeurRaw import create_data
    
    # Create data
    X, y = create_data(100, 3)
    
    # Create and train network
    network = create_enhanced_network()
    network.train(X, y, epochs=1000, print_every=200)
    
    # Make predictions
    predictions = network.predict(X[:5])
    print(f"\nPredictions for first 5 samples:")
    print(predictions)