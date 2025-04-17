import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from wordcloud import WordCloud
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from data_loader import IMDBDataset

# Download required NLTK data
nltk.download('punkt')

def create_analysis_dir():
    """Create directory for analysis outputs"""
    os.makedirs('analysis', exist_ok=True)
    os.makedirs('analysis/plots', exist_ok=True)

def analyze_label_distribution(texts: list, labels: list):
    """Analyze and visualize the distribution of labels"""
    plt.figure(figsize=(10, 5))
    
    # Create subplot for pie chart
    plt.subplot(1, 2, 1)
    label_counts = Counter(labels)
    plt.pie([label_counts[0], label_counts[1]], 
            labels=['Negative', 'Positive'],
            autopct='%1.1f%%',
            colors=['lightcoral', 'lightgreen'])
    plt.title('Distribution of Sentiment Labels')
    
    # Create subplot for bar plot
    plt.subplot(1, 2, 2)
    sentiment_df = pd.DataFrame({'sentiment': ['Negative' if l == 0 else 'Positive' for l in labels]})
    sns.countplot(data=sentiment_df,
                 x='sentiment',
                 hue='sentiment',
                 palette=['lightcoral', 'lightgreen'],
                 legend=False)
    plt.title('Count of Reviews by Sentiment')
    
    plt.tight_layout()
    plt.savefig('analysis/plots/label_distribution.png')
    plt.close()
    
    return {
        'negative_count': label_counts[0],
        'positive_count': label_counts[1],
        'total_reviews': len(texts)
    }

def analyze_text_statistics(texts: list):
    """Analyze text length and other statistical properties"""
    # Calculate lengths
    char_lengths = [len(text) for text in texts]
    word_lengths = [len(text.split()) for text in texts]
    
    stats = {
        'char_length': {
            'mean': np.mean(char_lengths),
            'median': np.median(char_lengths),
            'std': np.std(char_lengths),
            'min': min(char_lengths),
            'max': max(char_lengths)
        },
        'word_length': {
            'mean': np.mean(word_lengths),
            'median': np.median(word_lengths),
            'std': np.std(word_lengths),
            'min': min(word_lengths),
            'max': max(word_lengths)
        }
    }
    
    # Create visualizations
    plt.figure(figsize=(12, 5))
    
    # Character length distribution
    plt.subplot(1, 2, 1)
    sns.histplot(char_lengths, bins=50)
    plt.title('Distribution of Review Character Lengths')
    plt.xlabel('Number of Characters')
    plt.ylabel('Count')
    
    # Word length distribution
    plt.subplot(1, 2, 2)
    sns.histplot(word_lengths, bins=50)
    plt.title('Distribution of Review Word Counts')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('analysis/plots/length_distributions.png')
    plt.close()
    
    return stats

def analyze_vocabulary(texts: list, labels: list, min_freq: int = 100):
    """Analyze vocabulary and word frequencies"""
    # Split texts by sentiment
    positive_texts = [text for text, label in zip(texts, labels) if label == 1]
    negative_texts = [text for text, label in zip(texts, labels) if label == 0]
    
    # Get word frequencies
    def get_word_freq(texts):
        words = []
        for text in texts:
            # Simple word tokenization by splitting on whitespace and punctuation
            words.extend(text.lower().split())
        return Counter(words)
    
    pos_freq = get_word_freq(positive_texts)
    neg_freq = get_word_freq(negative_texts)
    all_freq = get_word_freq(texts)
    
    # Find most common words
    stats = {
        'total_unique_words': len(all_freq),
        'positive_unique_words': len(pos_freq),
        'negative_unique_words': len(neg_freq),
        'most_common_overall': all_freq.most_common(20),
        'most_common_positive': pos_freq.most_common(20),
        'most_common_negative': neg_freq.most_common(20)
    }
    
    # Create word frequency comparison plot
    plt.figure(figsize=(15, 5))
    
    words = [word for word, _ in all_freq.most_common(20)]
    pos_counts = [pos_freq[word] for word in words]
    neg_counts = [neg_freq[word] for word in words]
    
    x = np.arange(len(words))
    width = 0.35
    
    plt.bar(x - width/2, pos_counts, width, label='Positive', color='lightgreen')
    plt.bar(x + width/2, neg_counts, width, label='Negative', color='lightcoral')
    
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Most Common Words by Sentiment')
    plt.xticks(x, words, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('analysis/plots/word_frequencies.png')
    plt.close()
    
    # Analyze sentiment-specific words
    def get_sentiment_specific_words(pos_freq, neg_freq, min_freq):
        words = set(pos_freq.keys()) | set(neg_freq.keys())
        sentiment_words = {}
        
        for word in words:
            pos_count = pos_freq.get(word, 0)
            neg_count = neg_freq.get(word, 0)
            total = pos_count + neg_count
            
            if total >= min_freq:
                # Calculate sentiment score (-1 to 1)
                score = (pos_count - neg_count) / total
                sentiment_words[word] = {
                    'score': score,
                    'total': total,
                    'positive': pos_count,
                    'negative': neg_count
                }
        
        return sentiment_words
    
    sentiment_words = get_sentiment_specific_words(pos_freq, neg_freq, min_freq)
    
    # Sort words by absolute sentiment score
    sorted_words = sorted(sentiment_words.items(), 
                         key=lambda x: abs(x[1]['score']), 
                         reverse=True)
    
    # Plot top sentiment-indicating words
    plt.figure(figsize=(15, 6))
    
    # Split into positive and negative words
    pos_words = [(w, s['score']) for w, s in sorted_words if s['score'] > 0][:10]
    neg_words = [(w, s['score']) for w, s in sorted_words if s['score'] < 0][:10]
    
    # Plot positive words
    plt.subplot(1, 2, 1)
    words, scores = zip(*pos_words)
    plt.barh(range(len(words)), scores, color='lightgreen')
    plt.yticks(range(len(words)), words)
    plt.title('Top Positive-Sentiment Words')
    plt.xlabel('Sentiment Score')
    
    # Plot negative words
    plt.subplot(1, 2, 2)
    words, scores = zip(*neg_words)
    plt.barh(range(len(words)), scores, color='lightcoral')
    plt.yticks(range(len(words)), words)
    plt.title('Top Negative-Sentiment Words')
    plt.xlabel('Sentiment Score')
    
    plt.tight_layout()
    plt.savefig('analysis/plots/sentiment_words.png')
    plt.close()
    
    # Add sentiment-specific words to stats
    stats['sentiment_words'] = {
        'most_positive': sorted_words[:10],
        'most_negative': sorted(sentiment_words.items(), 
                              key=lambda x: x[1]['score'])[:10]
    }
    
    return stats

def create_word_clouds(texts: list, labels: list):
    """Generate word clouds for different subsets of the data"""
    def generate_word_cloud(texts, title, filename):
        text = ' '.join(texts)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.savefig(f'analysis/plots/{filename}.png')
        plt.close()
    
    # Generate word clouds
    positive_texts = [text for text, label in zip(texts, labels) if label == 1]
    negative_texts = [text for text, label in zip(texts, labels) if label == 0]
    
    generate_word_cloud(texts, 'All Reviews Word Cloud', 'wordcloud_all')
    generate_word_cloud(positive_texts, 'Positive Reviews Word Cloud', 'wordcloud_positive')
    generate_word_cloud(negative_texts, 'Negative Reviews Word Cloud', 'wordcloud_negative')

def perform_ngram_analysis(texts: list, labels: list, n_range=(1, 3)):
    """Analyze n-grams in the texts"""
    def get_ngrams(texts, n):
        all_ngrams = []
        for text in texts:
            tokens = text.lower().split()
            text_ngrams = list(ngrams(tokens, n))
            all_ngrams.extend(text_ngrams)
        return Counter(all_ngrams)
    
    positive_texts = [text for text, label in zip(texts, labels) if label == 1]
    negative_texts = [text for text, label in zip(texts, labels) if label == 0]
    
    stats = {}
    
    for n in range(n_range[0], n_range[1] + 1):
        pos_ngrams = get_ngrams(positive_texts, n)
        neg_ngrams = get_ngrams(negative_texts, n)
        
        stats[f'{n}-grams'] = {
            'positive': pos_ngrams.most_common(10),
            'negative': neg_ngrams.most_common(10)
        }
        
        # Visualize top n-grams
        plt.figure(figsize=(15, 5))
        
        # Get top n-grams overall
        all_ngrams = get_ngrams(texts, n)
        top_ngrams = [' '.join(ng) for ng, _ in all_ngrams.most_common(10)]
        pos_counts = [pos_ngrams[ng] for ng, _ in all_ngrams.most_common(10)]
        neg_counts = [neg_ngrams[ng] for ng, _ in all_ngrams.most_common(10)]
        
        x = np.arange(len(top_ngrams))
        width = 0.35
        
        plt.bar(x - width/2, pos_counts, width, label='Positive', color='lightgreen')
        plt.bar(x + width/2, neg_counts, width, label='Negative', color='lightcoral')
        
        plt.xlabel(f'{n}-grams')
        plt.ylabel('Frequency')
        plt.title(f'Most Common {n}-grams by Sentiment')
        plt.xticks(x, top_ngrams, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'analysis/plots/ngram_{n}_frequencies.png')
        plt.close()
    
    return stats

def main():
    # Create analysis directory
    create_analysis_dir()
    
    # Load data
    dataset = IMDBDataset()
    texts, labels = dataset.load_data("train")
    
    # Perform analyses
    stats = {}
    
    print("Analyzing label distribution...")
    stats['label_distribution'] = analyze_label_distribution(texts, labels)
    
    print("Analyzing text statistics...")
    stats['text_statistics'] = analyze_text_statistics(texts)
    
    print("Analyzing vocabulary...")
    stats['vocabulary'] = analyze_vocabulary(texts, labels)
    
    print("Creating word clouds...")
    create_word_clouds(texts, labels)
    
    print("Performing n-gram analysis...")
    stats['ngrams'] = perform_ngram_analysis(texts, labels)
    
    # Save statistics
    with open('analysis/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Analysis complete! Results saved in the 'analysis' directory.")

if __name__ == "__main__":
    main()
