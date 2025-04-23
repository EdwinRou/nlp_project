from data_loader import IMDBDataset
from model import SimpleSentimentClassifier
from preprocessor import TextPreprocessor
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

def run_experiment(use_preprocessing: bool = True):
    """Run experiment with or without preprocessing"""
    # Load the model and data
    dataset = IMDBDataset()
    train_texts, train_labels = dataset.load_data("train")
    test_texts, test_labels = dataset.load_data("test")
    
    # Build vocabulary from training data (use preprocessed texts if enabled)
    if use_preprocessing:
        preprocessor = TextPreprocessor()
        processed_train_texts = preprocessor.preprocess(train_texts)
        vocab = dataset.get_vocabulary(processed_train_texts)
        print("\nUsing preprocessed texts for training...")
    else:
        vocab = dataset.get_vocabulary(train_texts)
        print("\nUsing raw texts for training...")
    
    # Initialize and train model
    model = SimpleSentimentClassifier(vocab, use_preprocessing=use_preprocessing)
    print("Training model...")
    model.train(train_texts, train_labels)
    
    # Evaluate on both training and test sets
    evaluate_model(model, train_texts, train_labels, "Training")
    evaluate_model(model, test_texts, test_labels, "Test")

def main():
    # Run experiments with and without preprocessing
    print("=== Experiment with Raw Text ===")
    run_experiment(use_preprocessing=False)
    
    print("\n=== Experiment with Preprocessed Text ===")
    run_experiment(use_preprocessing=True)

if __name__ == "__main__":
    main()
