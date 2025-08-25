# src/utils.py

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    """
    Analyze sentiment polarity of a given text string.
    Returns a dictionary with VADER sentiment scores (pos, neu, neg, compound).
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

if __name__ == "__main__":
    sample_text = "Bitcoin prices are soaring with bullish market sentiment!"
    scores = analyze_sentiment(sample_text)
    print("Sentiment Scores:", scores)
