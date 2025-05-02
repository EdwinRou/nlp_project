import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments
)
from typing import List
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class SentimentDataset(Dataset):
    texts: List[str]
    labels: List[int]
    tokenizer: BertTokenizer
    max_length: int = 128
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTSentimentClassifier:
    def __init__(self, model_name: str = "bert-base-uncased"):
        """Simple BERT model for sentiment classification using Hugging Face"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        ).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
    
    def train_model(self, train_texts: List[str], train_labels: List[int],
                   val_texts: List[str] = None, val_labels: List[int] = None,
                   batch_size: int = 16, num_epochs: int = 3, learning_rate: float = 2e-5):
        """Train the model using Hugging Face's Trainer"""
        # Create datasets
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
        if val_texts:
            val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            do_eval=val_texts is not None,
            save_steps=500,
            save_total_limit=1,
            learning_rate=learning_rate,
            remove_unused_columns=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if val_texts else None
        )
        
        # Train the model
        trainer.train()
    
    def predict(self, texts: List[str], batch_size: int = 16) -> List[int]:
        """Predict sentiment labels for the input texts"""
        dataset = SentimentDataset(texts, [0] * len(texts), self.tokenizer)  # Dummy labels
        trainer = Trainer(model=self.model)
        predictions = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().tolist())
        
        return predictions
    
    def evaluate(self, texts: List[str], labels: List[int], batch_size: int = 16) -> float:
        """
        Evaluate model accuracy
        Args:
            texts: List of input texts
            labels: List of true labels
            batch_size: Batch size for evaluation
        Returns:
            Accuracy score
        """
        predictions = self.predict(texts, batch_size)
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return correct / len(texts)

def main():
    """Test the BERT classifier"""
    # Example texts
    texts = [
        "This movie is fantastic! I loved it.",
        "Terrible waste of time, awful movie.",
        "A great film that's worth watching.",
    ]
    labels = [1, 0, 1]  # 1 for positive, 0 for negative
    
    # Initialize model
    model = BERTSentimentClassifier()
    
    # Train model
    print("Training model...")
    model.train_model(texts, labels, num_epochs=1)  # Just for testing
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(texts)
    
    # Print results
    print("\nResults:")
    for text, pred in zip(texts, predictions):
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"Text: {text}")
        print(f"Prediction: {sentiment}\n")

if __name__ == "__main__":
    main()
