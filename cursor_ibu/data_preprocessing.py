import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
    def clean_text(self, text):
        """
        Clean text by removing special characters, numbers, and extra spaces
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from tokenized text
        """
        return [word for word in tokens if word not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens to their base form
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def stem_tokens(self, tokens):
        """
        Stem tokens to their root form
        """
        return [self.stemmer.stem(word) for word in tokens]
    
    def tokenize_text(self, text):
        """
        Tokenize text into words
        """
        return word_tokenize(text)
    
    def preprocess_text(self, text, remove_stopwords=True, use_lemmatization=True, use_stemming=False):
        """
        Complete text preprocessing pipeline
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize or stem
        if use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        elif use_stemming:
            tokens = self.stem_tokens(tokens)
        
        return ' '.join(tokens)
    
    def preprocess_dataset(self, df, text_column, target_column=None, remove_stopwords=True, 
                          use_lemmatization=True, use_stemming=False):
        """
        Preprocess entire dataset
        """
        df_processed = df.copy()
        
        # Preprocess text column
        df_processed['processed_text'] = df_processed[text_column].apply(
            lambda x: self.preprocess_text(x, remove_stopwords, use_lemmatization, use_stemming)
        )
        
        # Remove empty processed texts
        df_processed = df_processed[df_processed['processed_text'].str.len() > 0]
        
        return df_processed

def create_sample_data():
    """
    Create sample data for testing
    """
    sample_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst purchase I've ever made. Terrible quality!",
        "The product is okay, nothing special but gets the job done.",
        "Absolutely fantastic! Best decision ever to buy this.",
        "Disappointed with the service. Very poor customer support.",
        "Great value for money. Highly recommend!",
        "Not worth the price. Mediocre at best.",
        "Excellent quality and fast delivery. Very satisfied!",
        "Could be better. Some issues with the design.",
        "Perfect! Exactly what I was looking for."
    ]
    
    sample_labels = ['positive', 'negative', 'neutral', 'positive', 'negative', 
                     'positive', 'negative', 'positive', 'neutral', 'positive']
    
    return pd.DataFrame({
        'text': sample_texts,
        'sentiment': sample_labels
    })

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    # Create sample data
    sample_df = create_sample_data()
    print("Original Data:")
    print(sample_df.head())
    
    # Preprocess the data
    processed_df = preprocessor.preprocess_dataset(sample_df, 'text', 'sentiment')
    print("\nProcessed Data:")
    print(processed_df[['text', 'processed_text', 'sentiment']].head()) 