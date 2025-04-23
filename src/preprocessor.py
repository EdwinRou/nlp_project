import re
import nltk
from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class TextPreprocessor:
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        """
        Initialize the text preprocessor
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Define sentiment-aware stopwords (words to keep)
        self.sentiment_words = {
            'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither',
            'hardly', 'barely', 'scarcely', 'rarely', 'seldom'
        }
        self.stop_words = self.stop_words - self.sentiment_words
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Handle contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'d", " would", text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess(self, texts: List[str], keep_sentence_structure: bool = False) -> List[str]:
        """
        Preprocess a list of texts
        Args:
            texts: List of text strings
            keep_sentence_structure: Whether to maintain sentence structure
        Returns:
            List of preprocessed texts
        """
        processed_texts = []
        
        for text in texts:
            # Clean text
            text = self.clean_text(text)
            
            if keep_sentence_structure:
                # Tokenize while maintaining sentence structure
                words = word_tokenize(text)
                
                if self.remove_stopwords:
                    words = [w for w in words if w not in self.stop_words]
                
                if self.lemmatize:
                    words = [self.lemmatizer.lemmatize(w) for w in words]
                
                processed_text = ' '.join(words)
            
            else:
                # Simple word splitting
                words = text.split()
                
                if self.remove_stopwords:
                    words = [w for w in words if w not in self.stop_words]
                
                if self.lemmatize:
                    words = [self.lemmatizer.lemmatize(w) for w in words]
                
                processed_text = ' '.join(words)
            
            processed_texts.append(processed_text)
        
        return processed_texts

def main():
    """Test the preprocessor"""
    # Example texts
    texts = [
        "This movie is fantastic! I've watched it twice.",
        "The acting wasn't good, and the plot was terrible.",
        "A great film that's worth watching.",
    ]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Process texts
    processed_texts = preprocessor.preprocess(texts)
    
    # Print results
    print("Original vs Processed texts:")
    for orig, proc in zip(texts, processed_texts):
        print(f"Original: {orig}")
        print(f"Processed: {proc}")
        print()

if __name__ == "__main__":
    main()
