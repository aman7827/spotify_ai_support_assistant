"""
src/intent_classifier.py
-------------------------
Intent Classification module for SpotifyCares AI Support Assistant.
Implements:
1. Baseline 1: Most Frequent Class Predictor (Majority Vote)
2. Baseline 2: TF-IDF Vectorizer + Logistic Regression
3. Final Model: Sentence Transformer Embeddings (all-MiniLM-L6-v2) + Logistic Regression

Strict constraints applied: No deep neural network training, no fine-tuning.
100% local, explainable execution.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

INTENT_TAXONOMY = [
    "Login & Auth Issue",
    "Subscription & Premium",
    "Billing & Refund Request",
    "Playback & Audio Bugs",
    "App Crash & Performance",
    "Account Compromise & Security",
    "Playlist & Library Issues",
    "General Query & Feature Request"
]

class MostFrequentClassifier:
    """Baseline 1: Always predicts the most frequent class seen during training."""
    def __init__(self):
        self.most_frequent_class = INTENT_TAXONOMY[0]

    def fit(self, X, y):
        counts = pd.Series(y).value_counts()
        self.most_frequent_class = counts.index[0]
        return self

    def predict(self, X):
        return [self.most_frequent_class] * len(X)


class TfidfLogRegClassifier:
    """Baseline 2: TF-IDF feature extraction with a linear Logistic Regression model."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, stop_words='english')
        self.model = LogisticRegression(C=1.0, max_iter=500, random_state=42)

    def fit(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        return self

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)


class SentenceTransformerClassifier:
    """
    Final Model: Encodes customer messages using pre-trained all-MiniLM-L6-v2 sentence embeddings
    and classifies intents using Logistic Regression.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(C=2.0, max_iter=500, random_state=42)

    def fit(self, X, y):
        embeddings = self.encoder.encode(X, show_progress_bar=False, convert_to_numpy=True)
        self.classifier.fit(embeddings, y)
        return self

    def predict(self, X):
        embeddings = self.encoder.encode(list(X), show_progress_bar=False, convert_to_numpy=True)
        return self.classifier.predict(embeddings)
