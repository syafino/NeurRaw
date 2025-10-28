import numpy as np
from collections import Counter
import re

class SimpleTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab = set()
        
    def build_vocab(self, texts):
        # Tokenize and count words // build vocab from text
        all_words = []
        for text in texts:
            words = self._tokenize(text)
            all_words.extend(words)
        
        # Get most frequent words
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(self.vocab_size - 2)  # -2 for UNK and PAD
        
        # Build mappings
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for i, (word, _) in enumerate(most_common, 2):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
            
        self.vocab = set(self.word_to_id.keys())
        
    def _tokenize(self, text):
        return re.findall(r'\w+|[^\w\s]', text.lower())
        
    def encode(self, text):
        #token ids from text
        tokens = self._tokenize(text)
        return [self.word_to_id.get(token, 1) for token in tokens]  # 1 is UNK
        
    def decode(self, token_ids):
        #convert token IDs back to text
        tokens = [self.id_to_word.get(id, '<UNK>') for id in token_ids]
        return ' '.join(tokens)
        
    def encode_batch(self, texts, max_length=None):
        #encode multiple texts
        encoded = [self.encode(text) for text in texts]
        
        if max_length:
            # Pad or truncate to max_length
            for i in range(len(encoded)):
                if len(encoded[i]) > max_length:
                    encoded[i] = encoded[i][:max_length]
                else:
                    encoded[i].extend([0] * (max_length - len(encoded[i])))
                    
        return np.array(encoded)

# Example usage
if __name__ == "__main__":
    texts = [
        "Hello world, this is a simple tokenizer",
        "Machine learning is fascinating",
        "Neural networks can learn complex patterns",
        "This tokenizer converts text to numbers"
    ]
    
    tokenizer = SimpleTokenizer(vocab_size=50)
    tokenizer.build_vocab(texts)

    sample_text = "Hello machine learning world"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Batch encoding
    batch_encoded = tokenizer.encode_batch(texts, max_length=10)
    print(f"\nBatch encoded shape: {batch_encoded.shape}")
    print(f"First encoded text: {batch_encoded[0]}")