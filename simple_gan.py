import numpy as np
from NeurRaw import Layer_Dense, Activation_ReLU, Activation_Softmax

class Activation_Tanh:
    def forward(self, inputs):
        self.output = np.tanh(inputs)

class Activation_LeakyReLU:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def forward(self, inputs):
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)

class SimpleGAN:
    def __init__(self, noise_dim=10, data_dim=2):
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        
        # Generator: noise -> data
        self.gen_layer1 = Layer_Dense(noise_dim, 16)
        self.gen_activation1 = Activation_ReLU()
        
        self.gen_layer2 = Layer_Dense(16, 16)
        self.gen_activation2 = Activation_ReLU()
        
        self.gen_layer3 = Layer_Dense(16, data_dim)
        self.gen_activation3 = Activation_Tanh()
        
        # Discriminator: data -> real/fake probability
        self.disc_layer1 = Layer_Dense(data_dim, 16)
        self.disc_activation1 = Activation_LeakyReLU(0.2)
        
        self.disc_layer2 = Layer_Dense(16, 16)
        self.disc_activation2 = Activation_LeakyReLU(0.2)
        
        self.disc_layer3 = Layer_Dense(16, 1)
        self.disc_activation3 = Activation_Softmax()  # Using softmax for binary classification
        
    def generate(self, noise):
        #generate data from noise
        self.gen_layer1.forward(noise)
        self.gen_activation1.forward(self.gen_layer1.output)
        
        self.gen_layer2.forward(self.gen_activation1.output)
        self.gen_activation2.forward(self.gen_layer2.output)
        
        self.gen_layer3.forward(self.gen_activation2.output)
        self.gen_activation3.forward(self.gen_layer3.output)
        
        return self.gen_activation3.output
    
    def discriminate(self, data):
        #Classify data as real or fake
        self.disc_layer1.forward(data)
        self.disc_activation1.forward(self.disc_layer1.output)
        
        self.disc_layer2.forward(self.disc_activation1.output)
        self.disc_activation2.forward(self.disc_layer2.output)
        
        self.disc_layer3.forward(self.disc_activation2.output)
        self.disc_activation3.forward(self.disc_layer3.output)
        
        return self.disc_activation3.output
    
    def train_step(self, real_data, batch_size=32):
        # Generate noise and fake data
        noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
        fake_data = self.generate(noise)
        
        real_predictions = self.discriminate(real_data[:batch_size])
        
        fake_predictions = self.discriminate(fake_data)
        
        # Calculate simple losses (very simplified)
        real_loss = np.mean((real_predictions - 1) ** 2)
        fake_loss = np.mean(fake_predictions ** 2)
        disc_loss = real_loss + fake_loss
        
        # Generator loss (wants discriminator to think fake data is real)
        gen_loss = np.mean((fake_predictions - 1) ** 2)
        
        return disc_loss, gen_loss, fake_data

def create_circle_data(n_samples=1000, radius=1.0, noise=0.1):
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    radii = np.random.normal(radius, noise, n_samples)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack([x, y])

if __name__ == "__main__":
    real_data = create_circle_data(1000, radius=2.0, noise=0.2)
    
    print(f"Real data shape: {real_data.shape}")
    print(f"Real data stats - Mean: {np.mean(real_data, axis=0)}, Std: {np.std(real_data, axis=0)}")
    
    gan = SimpleGAN(noise_dim=5, data_dim=2)
    
    epochs = 500
    batch_size = 64
    
    for epoch in range(epochs):
        indices = np.random.choice(len(real_data), batch_size, replace=False)
        batch_real = real_data[indices]
        
        disc_loss, gen_loss, fake_samples = gan.train_step(batch_real, batch_size)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}")
            print(f"  Discriminator Loss: {disc_loss:.4f}")
            print(f"  Generator Loss: {gen_loss:.4f}")
            print(f"  Sample fake data: {fake_samples[0]}")
            print()
    
    test_noise = np.random.normal(0, 1, (10, 5))
    generated_samples = gan.generate(test_noise)
    
    print("Generated samples after training:")
    for i, sample in enumerate(generated_samples):
        print(f"Sample {i}: [{sample[0]:.3f}, {sample[1]:.3f}]")
    
    print(f"\nGenerated data stats - Mean: {np.mean(generated_samples, axis=0)}, Std: {np.std(generated_samples, axis=0)}")
    print(f"Real data stats      - Mean: {np.mean(real_data, axis=0)}, Std: {np.std(real_data, axis=0)}")