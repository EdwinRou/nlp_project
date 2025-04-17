import numpy as np
from typing import List, Dict

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
    
    def train(self, texts: List[str], labels: List[int], learning_rate: float = 0.01, num_epochs: int = 10):
        """
        Train the classifier using simple logistic regression
        Args:
            texts: List of input texts
            labels: List of binary labels (0 or 1)
            learning_rate: Learning rate for gradient descent
            num_epochs: Number of training epochs
        """
        for epoch in range(num_epochs):
            total_loss = 0
            for text, label in zip(texts, labels):
                # Forward pass
                x = self._text_to_bow(text)
                pred = self._sigmoid(np.dot(x, self.word_weights))
                
                # Compute loss
                loss = -label * np.log(pred + 1e-10) - (1 - label) * np.log(1 - pred + 1e-10)
                total_loss += loss
                
                # Backward pass
                error = pred - label
                grad = error * x
                self.word_weights -= learning_rate * grad
            
            avg_loss = total_loss / len(texts)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict sentiment for input texts
        Args:
            texts: List of input texts
        Returns:
            List of predicted labels (0 or 1)
        """
        predictions = []
        for text in texts:
            x = self._text_to_bow(text)
            pred = self._sigmoid(np.dot(x, self.word_weights))
            predictions.append(1 if pred >= 0.5 else 0)
        return predictions
    
    def _sigmoid(self, x: float) -> float:
        """Compute sigmoid function"""
        return 1 / (1 + np.exp(-x))
