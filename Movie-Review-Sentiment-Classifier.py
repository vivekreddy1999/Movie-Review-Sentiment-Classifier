import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re

class SentimentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(max_iter=1000)
        
    def preprocess_text(self, text):
        # Handle missing values
        if pd.isna(text):
            return ""
        
        text = str(text)  # Convert to string in case of non-string input
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
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

def load_and_train_model(csv_path):
    # Load the CSV file
    print("Loading data from CSV...")
    df = pd.read_csv(csv_path)
    
    # Get column names
    columns = df.columns.tolist()
    print(f"Found columns: {columns}")
    
    # Extract reviews and sentiments
    reviews = df[columns[0]]  # First column - reviews
    sentiments = df[columns[1]]  # Second column - sentiments
    
    # Convert sentiment labels to numerical values if they're text
    if sentiments.dtype == 'object':
        sentiment_map = {'positive': 1, 'negative': 0}
        sentiments = sentiments.map(sentiment_map)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, sentiments, test_size=0.2, random_state=42
    )
    
    # Create and train the classifier
    classifier = SentimentClassifier()
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
        classifier = load_and_train_model(csv_path)
        
        # Test with some sample reviews
        test_reviews = [
            "This movie was absolutely fantastic!",
            "I really hated this movie, it was terrible.",
            "The movie was okay, nothing special.",
            "This movie gave me headache",
            "Wonderfull Movie. Must watch"
        ]
        
        predictions, probabilities = classifier.predict(test_reviews)
        
        print("\nSample Predictions:")
        for review, pred, prob in zip(test_reviews, predictions, probabilities):
            sentiment = "Positive" if pred == 1 else "Negative"
            confidence = prob.max() * 100
            print(f"\nReview: {review}")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.2f}%")
            
    except Exception as e:
        print(f"Error: {str(e)}")