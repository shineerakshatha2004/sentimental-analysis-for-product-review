# Sentiment Analysis using NLTK - VADER


## Install the Required Packages

bash
pip install nltk


## Download VADER Lexicon from NLTK

python
import nltk
nltk.download('vader_lexicon')


## Usage

Here's a basic example of how to use the VADER model:

python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon if not already done
nltk.download('vader_lexicon')

# Initialize the analyzer
analyzer = SentimentIntensityAnalyzer()

# Example text
text = "I love this product! It's amazing and works great."

# Get sentiment scores
scores = analyzer.polarity_scores(text)

print("Sentiment Scores:", scores)


### Example Output:

python
Sentiment Scores: {
    'neg': 0.0,
    'neu': 0.315,
    'pos': 0.685,
    'compound': 0.8316
}


## Interpretation of Scores

- pos: Probability of positive sentiment.
- neu: Probability of neutral sentiment.
- neg: Probability of negative sentiment.
- compound: Normalized, weighted composite score:
  - Range from -1 (most negative) to +1 (most positive).
  - Common thresholds:
    - compound â‰¥ 0.05: Positive  
    - compound â‰¤ -0.05: Negative  
    - Otherwise: Neutral

## Applications

- Social media monitoring  
- Customer feedback analysis  
- Product review classification  
- Chatbot emotion detection

## License

This project is licensed under the MIT License.
