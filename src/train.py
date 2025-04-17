from data_loader import IMDBDataset
from model import SimpleSentimentClassifier
import random
from typing import List, Tuple

def create_validation_split(texts: List[str], labels: List[int], val_ratio: float = 0.1) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Split data into training and validation sets
    """
    # Create indices and shuffle
    indices = list(range(len(texts)))
    random.shuffle(indices)
    
    # Split indices
    val_size = int(len(texts) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Split data
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    return train_texts, train_labels, val_texts, val_labels

def evaluate(model: SimpleSentimentClassifier, texts: List[str], labels: List[int]) -> float:
    """
    Evaluate model accuracy
    """
    predictions = model.predict(texts)
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(texts)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load data
    print("Loading dataset...")
    dataset = IMDBDataset()
    texts, labels = dataset.load_data("train")
    
    # Create validation split
    train_texts, train_labels, val_texts, val_labels = create_validation_split(texts, labels)
    print(f"Train size: {len(train_texts)}")
    print(f"Validation size: {len(val_texts)}")
    
    # Build vocabulary
    vocab = dataset.get_vocabulary(train_texts)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Initialize and train model
    model = SimpleSentimentClassifier(vocab)
    print("Training model...")
    model.train(train_texts, train_labels)
    
    # Evaluate
    train_acc = evaluate(model, train_texts, train_labels)
    val_acc = evaluate(model, val_texts, val_labels)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
