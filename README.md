# Sentiment Analysis of Movie Reviews - ENSAE NLP Mini Project

This is my mini project for the ENSAE NLP 2025 course. It implements and compares traditional (Bag-of-Words) and modern (BERT) approaches for sentiment analysis on movie reviews using the IMDB dataset.

## Project Structure

- `src/`: Source code for models and data processing
  - `bert_model.py`: BERT-based classifier
  - `model.py`: Traditional Bag-of-Words model
  - `data_loader.py`: Data loading and preprocessing
  - `evaluate.py`: Model evaluation scripts
- `analysis/`: Data analysis and visualizations
- `report/`: NeurIPS-formatted paper with complete findings
- `data/`: IMDB movie review dataset

## Installation

```bash
# Clone the repository
git clone https://github.com/EdwinRou/nlp_project.git
cd nlp_project

# Install dependencies
pip install -r requirements.txt

# Download and extract the IMDB dataset
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

## Results

- Traditional Model (Bag-of-Words + Logistic Regression):
  - Raw Text: 71% accuracy
  - With Preprocessing: 73% accuracy

- BERT Model:
  - Accuracy: 86%
  - Balanced performance across positive/negative reviews
  - F1-Score: 0.86

The complete methodology and analysis can be found in the NeurIPS-formatted report at `report/sentiment_analysis.pdf`.
