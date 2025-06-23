import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.manifold import TSNE
import pickle
import os

class TextVectorizer:
    def __init__(self, vectorization_method='tfidf'):
        """
        Initialize text vectorizer
        
        Args:
            vectorization_method (str): 'tfidf', 'count', 'lda', or 'lsa'
        """
        self.vectorization_method = vectorization_method
        self.vectorizer = None
        self.feature_names = None
        
    def create_vectorizer(self, max_features=5000, ngram_range=(1, 2), 
                         min_df=2, max_df=0.95, stop_words='english'):
        """
        Create vectorizer based on the specified method
        """
        if self.vectorization_method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words,
                lowercase=True
            )
        elif self.vectorization_method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words,
                lowercase=True
            )
        elif self.vectorization_method == 'lda':
            # First create TF-IDF vectorizer, then apply LDA
            tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words,
                lowercase=True
            )
            self.vectorizer = tfidf_vectorizer
            self.lda = LatentDirichletAllocation(
                n_components=100,
                random_state=42,
                max_iter=10
            )
        elif self.vectorization_method == 'lsa':
            # First create TF-IDF vectorizer, then apply LSA
            tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words,
                lowercase=True
            )
            self.vectorizer = tfidf_vectorizer
            self.lsa = TruncatedSVD(
                n_components=100,
                random_state=42
            )
        else:
            raise ValueError("vectorization_method must be 'tfidf', 'count', 'lda', or 'lsa'")
    
    def fit_transform(self, texts, **kwargs):
        """
        Fit the vectorizer and transform the texts
        """
        if self.vectorizer is None:
            self.create_vectorizer(**kwargs)
        
        # Fit and transform with the base vectorizer
        if self.vectorization_method in ['tfidf', 'count']:
            X = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
        elif self.vectorization_method == 'lda':
            X_tfidf = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            X = self.lda.fit_transform(X_tfidf)
        elif self.vectorization_method == 'lsa':
            X_tfidf = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            X = self.lsa.fit_transform(X_tfidf)
        
        return X
    
    def transform(self, texts):
        """
        Transform new texts using the fitted vectorizer
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted before transforming")
        
        if self.vectorization_method in ['tfidf', 'count']:
            return self.vectorizer.transform(texts)
        elif self.vectorization_method == 'lda':
            X_tfidf = self.vectorizer.transform(texts)
            return self.lda.transform(X_tfidf)
        elif self.vectorization_method == 'lsa':
            X_tfidf = self.vectorizer.transform(texts)
            return self.lsa.transform(X_tfidf)
    
    def get_feature_names(self):
        """
        Get feature names (words/ngrams)
        """
        return self.feature_names
    
    def get_top_features(self, n=10):
        """
        Get top features by importance
        """
        if self.feature_names is None:
            return []
        
        if self.vectorization_method == 'tfidf':
            # Get mean TF-IDF scores across all documents
            mean_scores = np.mean(self.vectorizer.idf_)
            top_indices = np.argsort(self.vectorizer.idf_)[:n]
            return [(self.feature_names[i], self.vectorizer.idf_[i]) for i in top_indices]
        elif self.vectorization_method == 'count':
            # Get most frequent features
            feature_counts = np.sum(self.vectorizer.transform([]).toarray(), axis=0)
            top_indices = np.argsort(feature_counts)[-n:][::-1]
            return [(self.feature_names[i], feature_counts[i]) for i in top_indices]
        else:
            return self.feature_names[:n]
    
    def save_vectorizer(self, filepath):
        """
        Save the fitted vectorizer to a file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_vectorizer(cls, filepath):
        """
        Load a fitted vectorizer from a file
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class AdvancedTextVectorizer:
    def __init__(self):
        self.vectorizers = {}
        
    def create_multiple_vectorizers(self, texts, methods=['tfidf', 'count']):
        """
        Create multiple vectorizers for ensemble methods
        """
        for method in methods:
            vectorizer = TextVectorizer(method)
            vectorizer.fit_transform(texts)
            self.vectorizers[method] = vectorizer
    
    def get_combined_features(self, texts):
        """
        Combine features from multiple vectorizers
        """
        combined_features = []
        
        for method, vectorizer in self.vectorizers.items():
            features = vectorizer.transform(texts)
            if hasattr(features, 'toarray'):
                features = features.toarray()
            combined_features.append(features)
        
        return np.hstack(combined_features)

def create_sample_vectorization_demo():
    """
    Create a demo of text vectorization
    """
    # Sample texts
    texts = [
        "I love this product it is amazing",
        "This product is terrible and awful",
        "The product is okay nothing special",
        "Absolutely fantastic best purchase ever",
        "Very disappointed with the quality"
    ]
    
    # Test different vectorization methods
    methods = ['tfidf', 'count']
    
    for method in methods:
        print(f"\n=== {method.upper()} Vectorization ===")
        vectorizer = TextVectorizer(method)
        X = vectorizer.fit_transform(texts, max_features=100)
        
        print(f"Shape: {X.shape}")
        print(f"Feature names (first 10): {vectorizer.get_feature_names()[:10]}")
        
        if method == 'tfidf':
            print(f"Top features: {vectorizer.get_top_features(5)}")

if __name__ == "__main__":
    create_sample_vectorization_demo() 