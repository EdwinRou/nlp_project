# IMDB Movie Reviews Dataset Analysis

## Dataset Overview
- Total number of reviews: 25,000
- Perfect balance between classes:
  * Positive reviews: 12,500 (50%)
  * Negative reviews: 12,500 (50%)

## Text Statistics
### Review Length
- Character length:
  * Mean: 1,263 characters
  * Median: 935 characters
  * Standard deviation: 958 characters
  * Range: 51 to 13,308 characters
- Word length:
  * Mean: 237 words
  * Median: 177 words
  * Standard deviation: 176 words
  * Range: 10 to 2,487 words

## Vocabulary Analysis
- Total unique words: 74,218
- Vocabulary distribution:
  * Words in positive reviews: 55,143
  * Words in negative reviews: 53,816
  * Significant overlap between positive and negative vocabularies

### Sentiment-Specific Words
#### Most Positive-Associated Words (with sentiment scores):
1. "edie" (1.000) - Appears exclusively in positive reviews
2. "paulie" (0.983) - Strong positive association
3. "felix" (0.934) - Strong positive association
4. "polanski" (0.906) - Strong positive association

#### Most Negative-Associated Words (with sentiment scores):
1. "boll" (-0.986) - Strong negative association, refers to director Uwe Boll
2. "uwe" (-0.980) - Strong negative association
3. "seagal" (-0.949) - Strong negative association
4. "unwatchable" (-0.925) - Strong negative sentiment
5. "stinker" (-0.922) - Strong negative sentiment
6. "incoherent" (-0.899) - Common negative criticism
7. "unfunny" (-0.873) - Frequent criticism in negative reviews
8. "waste" (-0.865) - Commonly used in negative context

### Most Common Words Overall
1. Top 5 overall:
   - "the" (336,652 occurrences)
   - "and" (164,106 occurrences)
   - "a" (163,136 occurrences)
   - "of" (145,854 occurrences)
   - "to" (135,703 occurrences)

### Movie-related terms frequency:
- "movie" appears 44,030 times (more in negative reviews: 24,955 vs 19,075)
- "film" appears 40,147 times (slightly more in positive reviews: 20,934 vs 19,213)

## N-gram Analysis
### Most Common Trigrams
Positive Reviews:
1. "one of the" (2,944 occurrences)
2. "this is a" (1,536 occurrences)
3. "it s a" (1,442 occurrences)

Negative Reviews:
1. "one of the" (1,996 occurrences)
2. "i don t" (1,622 occurrences)
3. "this movie is" (1,515 occurrences)

## Key Insights
1. **Personal Names Impact**: Names of certain directors/actors (e.g., Uwe Boll, Seagal) show strong correlation with negative reviews, while others (e.g., Polanski) correlate with positive reviews

2. **Quality Descriptors**: 
   - Negative reviews frequently use words like "unwatchable", "stinker", "incoherent"
   - These terms have very high negative sentiment scores (> 0.85 negative correlation)
   - More specific in criticism compared to positive reviews

3. **Objective vs. Subjective Language**:
   - Negative reviews tend to use more directly critical terms
   - Positive reviews show more variety in praise terms
   - Both use similar basic language structures but differ in sentiment-specific vocabulary
1. **Review Length**: Reviews show considerable variation in length, with some being very short (10 words) and others quite extensive (2,487 words).

2. **Vocabulary Distribution**: 
   - There's a significant overlap in vocabulary between positive and negative reviews
   - Positive reviews tend to use a slightly more diverse vocabulary (55,143 vs 53,816 unique words)

3. **Common Patterns**:
   - The phrase "one of the" is the most common trigram in both positive and negative reviews
   - Negative reviews have more occurrences of "i don t", suggesting more negative expressions
   - The word "movie" appears more frequently in negative reviews, while "film" is more common in positive reviews

4. **Writing Style**:
   - Both positive and negative reviews use similar basic language structures (common articles, prepositions)
   - The main differences appear in the context and sentiment-specific vocabulary

## Visualizations
The following visualizations are available in the `analysis/plots` directory:
- Label distribution (`label_distribution.png`)
- Review length distributions (`length_distributions.png`)
- Word frequencies comparison (`word_frequencies.png`)
- Word clouds for positive and negative reviews (`wordcloud_positive.png`, `wordcloud_negative.png`)
- N-gram frequency plots (`ngram_1_frequencies.png`, `ngram_2_frequencies.png`, `ngram_3_frequencies.png`)
- Sentiment-specific words (`sentiment_words.png`)

### Understanding the Sentiment Visualizations
1. **Word Clouds**: 
   - Show the most frequent words in positive and negative reviews
   - Size corresponds to frequency
   - Provides quick visual insight into common terms

2. **Sentiment-Specific Words Plot**:
   - Split view showing most characteristic positive and negative words
   - Score ranges from -1 (completely negative) to +1 (completely positive)
   - Horizontal bars indicate strength of sentiment association
   - Color coding: green for positive, red for negative

3. **Interpretation Tips**:
   - Personal names (directors, actors) can be strong sentiment indicators
   - Technical terms ("incoherent", "unwatchable") are more common in negative reviews
   - Context is important: "film" vs "movie" usage patterns suggest different reviewer attitudes
