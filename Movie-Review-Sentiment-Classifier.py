import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import html

class MovieReviewClassifier:
    def __init__(self):
        # Increased max_features since reviews are longer
        self.vectorizer = TfidfVectorizer(
            max_features=10000, 
            stop_words='english',
            ngram_range=(1, 2)  # Include both unigrams and bigrams
        )
        self.model = LogisticRegression(max_iter=1000)
        
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags and decode HTML entities
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def train(self, X_train, y_train):
        print("Preprocessing training data...")
        X_train_clean = [self.preprocess_text(text) for text in X_train]
        
        print("Converting text to TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train_clean)
        
        print("Training the model...")
        self.model.fit(X_train_tfidf, y_train)
        
    def predict(self, texts):
        texts_clean = [self.preprocess_text(text) for text in texts]
        texts_tfidf = self.vectorizer.transform(texts_clean)
        predictions = self.model.predict(texts_tfidf)
        probabilities = self.model.predict_proba(texts_tfidf)
        return predictions, probabilities

def train_movie_classifier(csv_path):
    # Load the CSV file
    print("Loading data from CSV...")
    df = pd.read_csv(csv_path)
    
    # Extract reviews and sentiments
    reviews = df['review']
    sentiments = df['sentiment']
    
    # Convert sentiment labels
    sentiment_map = {'positive': 1, 'negative': 0}
    sentiments = sentiments.map(sentiment_map)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, sentiments, test_size=0.2, random_state=42
    )
    
    # Create and train the classifier
    classifier = MovieReviewClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluate on test set
    predictions, probabilities = classifier.predict(X_test)
    
    # Print performance metrics
    print("\nModel Performance:")
    print(classification_report(y_test, predictions))
    
    return classifier

if __name__ == "__main__":
    # Replace with your CSV file path
    csv_path = "IMDB Dataset.csv"
    
    try:
        classifier = train_movie_classifier(csv_path)
        
        # Test with a sample review
        test_reviews = [
            """One of the best movies I've seen this year! The acting was superb and the plot kept me engaged throughout. Definitely recommend it!""",
            
            """This movie was a complete waste of time. Poor acting, terrible script, and the plot made no sense. Save your money.""",

            """This is the best Movie I have watched this year.""",

            """Watching this movie was a Mistake. You people don't make the same mistake."""
        ]
        
        predictions, probabilities = classifier.predict(test_reviews)
        
        print("\nSample Predictions:")
        for review, pred, prob in zip(test_reviews, predictions, probabilities):
            sentiment = "Positive" if pred == 1 else "Negative"
            confidence = prob.max() * 100
            print(f"\nReview: {review[:100]}...")  # Show first 100 chars
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.2f}%")
            
    except Exception as e:
        print(f"Error: {str(e)}")