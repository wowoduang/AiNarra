import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure you have the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

class HighlightExtractor:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def extract_highlights(self, text):
        sentences = text.split('. ')
        highlights = []

        for sentence in sentences:
            if self.is_positive(sentence):
                highlights.append(sentence)

        return highlights

    def is_positive(self, sentence):
        score = self.sia.polarity_scores(sentence)
        return score['compound'] > 0.05  # Threshold for positive sentiment

# Example usage:
# extractor = HighlightExtractor()
# highlights = extractor.extract_highlights("This is an amazing project. It has some flaws, but overall great!")
# print(highlights)