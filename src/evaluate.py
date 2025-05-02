from data_loader import IMDBDataset
from model import SimpleSentimentClassifier
from bert_model import BERTSentimentClassifier
from preprocessor import TextPreprocessor
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer

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

def run_experiment(model_type: str = "simple", use_preprocessing: bool = True, num_samples: int = 1000):
    """
    Run experiment with specified model type
    Args:
        model_type: Type of model to use ("simple" or "bert")
        use_preprocessing: Whether to use preprocessing (only for simple model)
    """
    # Load the data with limited samples for testing
    dataset = IMDBDataset()
    # Load balanced datasets with specified limit
    train_texts, train_labels = dataset.load_data("train", limit=num_samples)
    test_texts, test_labels = dataset.load_data("test", limit=num_samples//5)
    
    # Initialize model based on type
    if model_type == "simple":
        # Build vocabulary for simple model
        if use_preprocessing:
            preprocessor = TextPreprocessor()
            processed_train_texts = preprocessor.preprocess(train_texts)
            vocab = dataset.get_vocabulary(processed_train_texts)
            print("\nUsing preprocessed texts for simple model...")
        else:
            vocab = dataset.get_vocabulary(train_texts)
            print("\nUsing raw texts for simple model...")
        
        # Initialize and train simple model
        model = SimpleSentimentClassifier(vocab, use_preprocessing=use_preprocessing)
        print("Training simple model...")
        model.train(train_texts, train_labels)
    
    else:  # bert model
        print("\nInitializing BERT model...")
        model = BERTSentimentClassifier()
        print("Training BERT model...")
        # Initialize and train BERT model using Hugging Face Trainer
        model.train_model(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=test_texts,
            val_labels=test_labels,
            batch_size=16,
            num_epochs=3,
            learning_rate=2e-5
        )
    
    # Evaluate on both training and test sets
    evaluate_model(model, train_texts, train_labels, f"{model_type.capitalize()} Model - Training")
    evaluate_model(model, test_texts, test_labels, f"{model_type.capitalize()} Model - Test")

def main():
    # Run experiments with both models
    # Run experiments with balanced datasets
    print("=== Simple Model with Raw Text ===")
    run_experiment("simple", use_preprocessing=False, num_samples=1000)
    
    print("\n=== Simple Model with Preprocessed Text ===")
    run_experiment("simple", use_preprocessing=True, num_samples=1000)
    
    print("\n=== BERT Model ===")
    run_experiment("bert", num_samples=1000)

if __name__ == "__main__":
    main()
