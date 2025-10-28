# NeurRaw

Neural Networks from Scratch

- **`NeurRaw.py`** - Basic neural network with:
  - Dense layers with random weight initialization
  - ReLU and Softmax activation functions
  - Categorical cross-entropy loss
  - Spiral dataset generator
  - Simple 2-layer classification example

### **Tokenizer** (`tokenizer.py`)
- Convert text to numbers for NLP
- Build vocabulary from text data
- Encode/decode with padding support
- Ready for text classification

### **Enhanced Network** (`enhanced_network.py`) 
- Dropout layers for regularization
- SGD optimizer with momentum and decay
- Modular network class
- Better training loop

### **Text Classifier** (`text_classifier.py`)
- Sentiment analysis example
- Combines tokenizer + neural network
- Simple embedding layer
- End-to-end text classification

### **Autoencoder** (`autoencoder.py`)
- Dimensionality reduction
- Data compression/decompression
- Noise removal capability
- Encoder-decoder architecture

### **Simple GAN** (`simple_gan.py`)
- Generative Adversarial Network
- Generator + Discriminator
- Creates new data samples
- Basic adversarial training

### **Examples** (`examples.py`)
- Demonstrations of all components
- Quick testing of functionality

