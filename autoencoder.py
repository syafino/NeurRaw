import numpy as np
from NeurRaw import Layer_Dense, Activation_ReLU

class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500)))

class Loss_MeanSquaredError:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2, axis=-1)
    
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Autoencoder:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder1 = Layer_Dense(input_dim, hidden_dim * 2)
        self.encoder_activation1 = Activation_ReLU()
        
        self.encoder2 = Layer_Dense(hidden_dim * 2, hidden_dim)
        self.encoder_activation2 = Activation_ReLU()
        
        # Decoder
        self.decoder1 = Layer_Dense(hidden_dim, hidden_dim * 2)
        self.decoder_activation1 = Activation_ReLU()
        
        self.decoder2 = Layer_Dense(hidden_dim * 2, input_dim)
        self.decoder_activation2 = Activation_Sigmoid()
        
        self.loss_function = Loss_MeanSquaredError()
        
    def encode(self, X):
        #encode input
        self.encoder1.forward(X)
        self.encoder_activation1.forward(self.encoder1.output)
        
        self.encoder2.forward(self.encoder_activation1.output)
        self.encoder_activation2.forward(self.encoder2.output)
        
        return self.encoder_activation2.output
    
    def decode(self, encoded):
        #decode input
        self.decoder1.forward(encoded)
        self.decoder_activation1.forward(self.decoder1.output)
        
        self.decoder2.forward(self.decoder_activation1.output)
        self.decoder_activation2.forward(self.decoder2.output)
        
        return self.decoder_activation2.output
    
    def forward(self, X):
        #encode, then decode
        encoded = self.encode(X)
        decoded = self.decode(encoded)
        return decoded
    
    def train_step(self, X, learning_rate=0.01):
        #using gradient descent to update weights
        # Forward pass
        reconstructed = self.forward(X)
        
        # Calculate loss
        loss = self.loss_function.calculate(reconstructed, X)
        
        return loss, reconstructed

def create_noisy_data(n_samples=1000, noise_factor=0.1):
    # Create clean sinusoidal data
    x = np.linspace(0, 4*np.pi, n_samples)
    clean_data = np.column_stack([
        np.sin(x),
        np.cos(x),
        np.sin(2*x),
        np.cos(2*x)
    ])
    
    # Add noise
    noise = np.random.normal(0, noise_factor, clean_data.shape)
    noisy_data = clean_data + noise
    
    # Normalize to [0, 1]
    noisy_data = (noisy_data + 1) / 2
    clean_data = (clean_data + 1) / 2
    
    return noisy_data, clean_data

if __name__ == "__main__":
    # Create sample data
    noisy_data, clean_data = create_noisy_data(500, noise_factor=0.2)
    
    print(f"Data shape: {noisy_data.shape}")
    print(f"Data range: [{noisy_data.min():.3f}, {noisy_data.max():.3f}]")
    
    # Create autoencoder
    autoencoder = Autoencoder(input_dim=4, hidden_dim=2)
    
    # Training loop
    losses = []
    epochs = 1000
    
    for epoch in range(epochs):
        loss, reconstructed = autoencoder.train_step(noisy_data, learning_rate=0.1)
        losses.append(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    # Test the autoencoder
    print("\nTesting autoencoder:")
    
    # Encode some samples
    encoded = autoencoder.encode(noisy_data[:5])
    decoded = autoencoder.decode(encoded)
    
    print("Original vs Reconstructed (first 5 samples):")
    for i in range(5):
        print(f"Original:      {noisy_data[i]}")
        print(f"Reconstructed: {decoded[i]}")
        print(f"Difference:    {np.abs(noisy_data[i] - decoded[i])}")
        print()
    
    # Show latent representations
    print("Latent representations (first 5 samples):")
    for i in range(5):
        print(f"Sample {i}: {encoded[i]}")
    
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Average reconstruction error: {np.mean(np.abs(noisy_data[:100] - autoencoder.forward(noisy_data[:100]))):.6f}")