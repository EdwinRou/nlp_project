import numpy as np
from typing import List, Dict
from tqdm import tqdm

class SimpleSentimentClassifier:
    def __init__(self, vocab: Dict[str, int], max_words: int = 10000):
        """
        Initialize a simple bag-of-words sentiment classifier
        Args:
            vocab: Dictionary mapping words to indices
            max_words: Maximum number of words to consider
        """
        self.vocab = vocab
        self.max_words = max_words
        self.word_weights = np.zeros(max_words)
        
    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        Convert text to bag-of-words representation
        Args:
            text: Input text string
        Returns:
            Bag-of-words vector
        """
        vector = np.zeros(self.max_words)
        for word in text.split():
            idx = self.vocab.get(word, self.vocab["<UNK>"])
            vector[idx] += 1
        return vector
    
    def train(self, texts: List[str], labels: List[int], learning_rate: float = 0.01, num_epochs: int = 10, batch_size: int = 32):
        """
        Train the classifier using batch gradient descent
        Args:
            texts: List of input texts
            labels: List of binary labels (0 or 1)
            learning_rate: Learning rate for gradient descent
            num_epochs: Number of training epochs
            batch_size: Number of examples per batch
        """
        num_examples = len(texts)
        indices = list(range(num_examples))
        
        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            total_loss = 0
            np.random.shuffle(indices)
            
            # Process mini-batches
            for start_idx in tqdm(range(0, num_examples, batch_size), desc=f"Epoch {epoch+1}", leave=False):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_size_actual = len(batch_indices)
                
                # Convert batch of texts to bag-of-words
                X = np.zeros((batch_size_actual, self.max_words))
                for i, idx in enumerate(batch_indices):
                    X[i] = self._text_to_bow(texts[idx])
                
                batch_labels = np.array([labels[idx] for idx in batch_indices])
                
                # Forward pass
                preds = self._sigmoid(np.dot(X, self.word_weights))
                
                # Compute loss
                loss = -np.mean(
                    batch_labels * np.log(preds + 1e-10) + 
                    (1 - batch_labels) * np.log(1 - preds + 1e-10)
                )
                total_loss += loss * batch_size_actual
                
                # Backward pass
                errors = preds - batch_labels
                grad = np.dot(X.T, errors) / batch_size_actual
                self.word_weights -= learning_rate * grad
            
            avg_loss = total_loss / num_examples
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        """
        Predict sentiment for input texts
        Args:
            texts: List of input texts
            batch_size: Number of examples per batch
        Returns:
            List of predicted labels (0 or 1)
        """
        predictions = []
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]
            batch_size_actual = len(batch_texts)
            
            # Convert batch of texts to bag-of-words
            X = np.zeros((batch_size_actual, self.max_words))
            for i, text in enumerate(batch_texts):
                X[i] = self._text_to_bow(text)
            
            # Get predictions
            preds = self._sigmoid(np.dot(X, self.word_weights))
            predictions.extend([1 if p >= 0.5 else 0 for p in preds])
        
        return predictions
    
    def _sigmoid(self, x: float) -> float:
        """Compute sigmoid function"""
        return 1 / (1 + np.exp(-x))
