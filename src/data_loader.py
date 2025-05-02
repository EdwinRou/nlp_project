import os
from typing import Tuple, List, Dict
from torch.utils.data import Dataset
from collections import Counter

class IMDBDataset(Dataset):
    def __init__(self, data_dir: str = "data/aclImdb"):
        """
        Initialize the IMDB dataset loader
        Args:
            data_dir: Path to the IMDB dataset directory
        """
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, "train")
        self.test_dir = os.path.join(data_dir, "test")
    
    def _read_text(self, text: str) -> str:
        """
        Minimal text cleaning, let BERT tokenizer handle the rest
        """
        # Just remove HTML tags and extra whitespace
        text = text.replace('<br />', ' ')
        return ' '.join(text.split())

    def load_data(self, split: str = "train", limit: int = None) -> Tuple[List[str], List[int]]:
        """
        Load data from the specified split
        Args:
            split: Either "train" or "test"
        Returns:
            Tuple of (texts, labels)
        """
        data_dir = self.train_dir if split == "train" else self.test_dir
        texts, labels = [], []
        
        # Load positive reviews
        pos_dir = os.path.join(data_dir, "pos")
        for filename in os.listdir(pos_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    texts.append(self._read_text(text))
                    labels.append(1)
        
        # Load negative reviews
        neg_dir = os.path.join(data_dir, "neg")
        for filename in os.listdir(neg_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    texts.append(self._read_text(text))
                    labels.append(0)        # Optionally limit dataset size
        if limit:
            pos_indices = [i for i, label in enumerate(labels) if label == 1][:limit//2]
            neg_indices = [i for i, label in enumerate(labels) if label == 0][:limit//2]
            indices = pos_indices + neg_indices
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        return texts, labels

    def __len__(self):
        return len(self.texts) if hasattr(self, 'texts') else 0

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def get_vocabulary(self, texts: List[str], max_features: int = 10000) -> Dict[str, int]:
        """
        Build vocabulary from texts
        Args:
            texts: List of text strings
            max_features: Maximum number of words to include in vocabulary
        Returns:
            Dictionary mapping words to indices
        """
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())
        
        # Create vocabulary with most common words
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in word_counts.most_common(max_features - len(vocab)):
            vocab[word] = len(vocab)
        
        return vocab

def main():
    """Simple test of the dataset loader"""
    dataset = IMDBDataset()
    texts, labels = dataset.load_data("train")
    print(f"Loaded {len(texts)} training examples")
    print(f"Sample text:\n{texts[0][:200]}...")
    print(f"Sample label: {labels[0]}")
    
    vocab = dataset.get_vocabulary(texts)
    print(f"Vocabulary size: {len(vocab)}")

if __name__ == "__main__":
    main()
