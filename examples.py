# examples
import numpy as np
from NeurRaw import create_data, Layer_Dense, Activation_ReLU, Activation_Softmax, lcce

def demo_original_network():
    # spiral data
    X, y = create_data(100, 3)
    
    # network
    layer1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()
    layer2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()
    
    # Forward pass
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    
    # Calculate loss
    loss_function = lcce()
    loss = loss_function.calculate(activation2.output, y)
    
    print(f"Sample predictions: {activation2.output[:3]}")
    print(f"Loss: {loss:.4f}")
    print()

def demo_tokenizer():
    #Tokenizer demo
    
    try:
        from tokenizer import SimpleTokenizer
        
        # Sample texts
        texts = [
            "Hello world, this is neural networks",
            "Machine learning with raw python",
            "Building AI from scratch"
        ]
        
        tokenizer = SimpleTokenizer(vocab_size=50)
        tokenizer.build_vocab(texts)
        
        # Test encoding/decoding
        sample_text = "Hello machine learning"
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"Original: {sample_text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Vocabulary size: {len(tokenizer.vocab)}")
        print()
        
    except ImportError:
        print("Tokenizer module not found. Make sure tokenizer.py is in the same directory.")
        print()

def demo_data_generation():
    #different ways to generate data
    
    # Spiral data (from original)
    X_spiral, y_spiral = create_data(50, 2)
    print(f"Spiral data shape: {X_spiral.shape}, classes: {len(np.unique(y_spiral))}")
    
    # Simple linear data
    np.random.seed(42)
    X_linear = np.random.randn(100, 2)
    y_linear = (X_linear[:, 0] + X_linear[:, 1] > 0).astype(int)
    print(f"Linear data shape: {X_linear.shape}, classes: {len(np.unique(y_linear))}")
    
    # Circular data
    angles = np.linspace(0, 2*np.pi, 100)
    radius = 1 + 0.1 * np.random.randn(100)
    X_circle = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    y_circle = np.zeros(100)  # All same class
    print(f"Circle data shape: {X_circle.shape}")
    
    print()

if __name__ == "__main__":
    print("NeurRaw")
    print()
    
    # Run demos
    demo_original_network()
    demo_tokenizer()
    demo_data_generation()
    
    print("Works")