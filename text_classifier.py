import numpy as np
from tokenizer import SimpleTokenizer
from enhanced_network import NeuralNetwork, Layer_Dense, Activation_ReLU, Activation_Softmax, lcce, Optimizer_SGD

class TextClassifier:
    def __init__(self, vocab_size=1000, embedding_dim=32, hidden_dim=64, num_classes=2):
        self.tokenizer = SimpleTokenizer(vocab_size)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.network = None
        
    def create_embedding_layer(self, vocab_size, embedding_dim):
        return 0.1 * np.random.randn(vocab_size, embedding_dim)
        
    def embed_sequence(self, token_ids, embeddings):
        embedded = embeddings[token_ids]
        return np.mean(embedded, axis=0)
        
    def prepare_data(self, texts, labels, max_length=20):
        self.tokenizer.build_vocab(texts)
        
        self.embeddings = self.create_embedding_layer(len(self.tokenizer.vocab), self.embedding_dim)
        
        # Encode
        X = []
        for text in texts:
            token_ids = self.tokenizer.encode(text)
            # Truncate or pad
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids.extend([0] * (max_length - len(token_ids)))

            # Convert to embeddings and average
            embedded = self.embed_sequence(token_ids, self.embeddings)
            X.append(embedded)
            
        return np.array(X), np.array(labels)
        
    def build_network(self):
        self.network = NeuralNetwork()
        
        #Layers
        self.network.add(Layer_Dense(self.embedding_dim, self.hidden_dim))
        self.network.add(Activation_ReLU())
        
        self.network.add(Layer_Dense(self.hidden_dim, self.hidden_dim))
        self.network.add(Activation_ReLU())
        
        self.network.add(Layer_Dense(self.hidden_dim, self.num_classes))
        self.network.add(Activation_Softmax())
        
        self.network.set_loss(lcce())
        self.network.set_optimizer(Optimizer_SGD(learning_rate=0.1, decay=1e-5, momentum=0.9))
        
    def train(self, texts, labels, epochs=1000):
        X, y = self.prepare_data(texts, labels)
        
        self.build_network()
        
        # Train
        self.network.train(X, y, epochs=epochs, print_every=200)
        
    def predict(self, texts):
        #predict new texts
        X = []
        for text in texts:
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) > 20:
                token_ids = token_ids[:20]
            else:
                token_ids.extend([0] * (20 - len(token_ids)))
                
            embedded = self.embed_sequence(token_ids, self.embeddings)
            X.append(embedded)
            
        X = np.array(X)
        return self.network.predict(X)

# Example
if __name__ == "__main__":
    texts = [
        "I love this movie, it's amazing!",
        "This film is terrible and boring",
        "Great acting and wonderful story",
        "Worst movie I've ever seen",
        "Fantastic cinematography and direction",
        "Complete waste of time",
        "Beautiful and touching story",
        "Absolutely horrible acting",
        "I enjoyed every minute of it",
        "Could not wait for it to end"
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    
    # create n train classifier
    classifier = TextClassifier(vocab_size=100, num_classes=2)
    classifier.train(texts, labels, epochs=500)
    
    # predictions
    test_texts = [
        "This movie is amazing!",
        "I hate this film"
    ]
    
    predictions = classifier.predict(test_texts)
    
    for i, text in enumerate(test_texts):
        pred_class = np.argmax(predictions[i])
        confidence = predictions[i][pred_class]
        sentiment = "Positive" if pred_class == 1 else "Negative"
        print(f"Text: '{text}'")
        print(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
        print()