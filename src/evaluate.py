from data_loader import IMDBDataset
from model import SimpleSentimentClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, texts, labels, split_name="Test"):
    """
    Evaluate model performance with detailed metrics
    """
    predictions = model.predict(texts)
    
    # Calculate and print detailed metrics
    print(f"\n{split_name} Set Metrics:")
    print(classification_report(labels, predictions))
    
    # Calculate and print confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:")
    print("              Predicted Negative  Predicted Positive")
    print(f"Actual Negative     {conf_matrix[0][0]:^14d}    {conf_matrix[0][1]:^14d}")
    print(f"Actual Positive     {conf_matrix[1][0]:^14d}    {conf_matrix[1][1]:^14d}")

def main():
    # Load the model and data
    dataset = IMDBDataset()
    train_texts, train_labels = dataset.load_data("train")
    test_texts, test_labels = dataset.load_data("test")
    
    # Build vocabulary from training data
    vocab = dataset.get_vocabulary(train_texts)
    
    # Initialize and train model
    model = SimpleSentimentClassifier(vocab)
    print("Training model...")
    model.train(train_texts, train_labels)
    
    # Evaluate on both training and test sets
    evaluate_model(model, train_texts, train_labels, "Training")
    evaluate_model(model, test_texts, test_labels, "Test")

if __name__ == "__main__":
    main()
