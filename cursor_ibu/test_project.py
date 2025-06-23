#!/usr/bin/env python3
"""
Test script for the Sentiment Analysis & Emotion Detection Project
This script tests all major components to ensure they work correctly.
"""

import sys
import traceback
from sklearn.model_selection import train_test_split

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import nltk
        print("✅ All basic imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from data_preprocessing import TextPreprocessor
        from text_vectorization import TextVectorizer
        from sentiment_models import SentimentClassifier, EmotionClassifier, ModelEvaluator
        print("✅ All project module imports successful")
    except ImportError as e:
        print(f"❌ Project module import error: {e}")
        return False
    
    return True

def test_preprocessing():
    """Test text preprocessing functionality"""
    print("\n🔍 Testing text preprocessing...")
    
    try:
        from data_preprocessing import TextPreprocessor, create_sample_data
        
        # Create preprocessor
        preprocessor = TextPreprocessor()
        
        # Test single text preprocessing
        test_text = "I love this product! It's amazing and works perfectly. Check it out at https://example.com"
        processed = preprocessor.preprocess_text(test_text)
        print(f"✅ Single text preprocessing: '{test_text}' -> '{processed}'")
        
        # Test dataset preprocessing
        sample_df = create_sample_data()
        processed_df = preprocessor.preprocess_dataset(sample_df, 'text', 'sentiment')
        print(f"✅ Dataset preprocessing: {len(processed_df)} samples processed")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        traceback.print_exc()
        return False

def test_vectorization():
    """Test text vectorization functionality"""
    print("\n🔍 Testing text vectorization...")
    
    try:
        from text_vectorization import TextVectorizer
        
        # Test TF-IDF vectorization
        texts = [
            "I love this product it is amazing",
            "This product is terrible and awful",
            "The product is okay nothing special"
        ]
        
        vectorizer = TextVectorizer('tfidf')
        X = vectorizer.fit_transform(texts, max_features=100)
        print(f"✅ TF-IDF vectorization: {X.shape}")
        
        # Test Count vectorization
        count_vectorizer = TextVectorizer('count')
        X_count = count_vectorizer.fit_transform(texts, max_features=100)
        print(f"✅ Count vectorization: {X_count.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Vectorization error: {e}")
        traceback.print_exc()
        return False

def test_sentiment_models():
    """Test sentiment analysis models"""
    print("\n🔍 Testing sentiment analysis models...")
    
    try:
        from data_preprocessing import TextPreprocessor, create_sample_data
        from text_vectorization import TextVectorizer
        from sentiment_models import SentimentClassifier, ModelEvaluator
        
        # Prepare data
        preprocessor = TextPreprocessor()
        sample_df = create_sample_data()
        processed_df = preprocessor.preprocess_dataset(sample_df, 'text', 'sentiment')
        
        vectorizer = TextVectorizer('tfidf')
        X = vectorizer.fit_transform(processed_df['processed_text'])
        y = processed_df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Test different models
        models = ['logistic_regression', 'random_forest', 'naive_bayes']
        evaluator = ModelEvaluator()
        
        for model_type in models:
            print(f"  Testing {model_type}...")
            classifier = SentimentClassifier(model_type)
            classifier.fit(X_train, y_train)
            
            # Evaluate
            results = evaluator.evaluate_model(classifier, X_test, y_test, model_type)
            print(f"    ✅ {model_type}: Accuracy = {results['accuracy']:.3f}, F1 = {results['f1_score']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Sentiment models error: {e}")
        traceback.print_exc()
        return False

def test_emotion_models():
    """Test emotion detection models"""
    print("\n🔍 Testing emotion detection models...")
    
    try:
        from sentiment_models import EmotionClassifier, create_sample_emotion_data
        from data_preprocessing import TextPreprocessor
        from text_vectorization import TextVectorizer
        
        # Prepare emotion data
        preprocessor = TextPreprocessor()
        sample_df = create_sample_emotion_data()
        processed_df = preprocessor.preprocess_dataset(sample_df, 'text', 'emotion')
        
        vectorizer = TextVectorizer('tfidf')
        X = vectorizer.fit_transform(processed_df['processed_text'])
        y = processed_df['emotion']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Test emotion classifier
        emotion_classifier = EmotionClassifier('logistic_regression')
        emotion_classifier.fit(X_train, y_train)
        
        predictions = emotion_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"✅ Emotion detection: Accuracy = {accuracy:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Emotion models error: {e}")
        traceback.print_exc()
        return False

def test_flask_app():
    """Test Flask app functionality"""
    print("\n🔍 Testing Flask app...")
    
    try:
        from app import app
        print("✅ Flask app can be imported")
        
        # Test app configuration
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/api/health')
            print(f"✅ Health endpoint: {response.status_code}")
            
            # Test model info endpoint
            response = client.get('/api/model_info')
            print(f"✅ Model info endpoint: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Flask app error: {e}")
        traceback.print_exc()
        return False

def test_end_to_end():
    """Test end-to-end functionality"""
    print("\n🔍 Testing end-to-end functionality...")
    
    try:
        from data_preprocessing import TextPreprocessor, create_sample_data
        from text_vectorization import TextVectorizer
        from sentiment_models import SentimentClassifier, EmotionClassifier
        
        # Complete pipeline test
        print("  Running complete pipeline...")
        
        # 1. Data preprocessing
        preprocessor = TextPreprocessor()
        sample_df = create_sample_data()
        processed_df = preprocessor.preprocess_dataset(sample_df, 'text', 'sentiment')
        
        # 2. Vectorization
        vectorizer = TextVectorizer('tfidf')
        X = vectorizer.fit_transform(processed_df['processed_text'])
        y_sentiment = processed_df['sentiment']
        
        # 3. Sentiment analysis
        sentiment_classifier = SentimentClassifier('logistic_regression')
        sentiment_classifier.fit(X, y_sentiment)
        
        # 4. Test prediction
        test_text = "I love this amazing product!"
        processed_text = preprocessor.preprocess_text(test_text)
        X_test = vectorizer.transform([processed_text])
        prediction = sentiment_classifier.predict(X_test)[0]
        print(f"    ✅ Sentiment prediction: '{test_text}' -> {prediction}")
        
        # 5. Emotion detection
        emotion_df = create_sample_emotion_data()
        emotion_processed = preprocessor.preprocess_dataset(emotion_df, 'text', 'emotion')
        X_emotion = vectorizer.fit_transform(emotion_processed['processed_text'])
        y_emotion = emotion_processed['emotion']
        
        emotion_classifier = EmotionClassifier('logistic_regression')
        emotion_classifier.fit(X_emotion, y_emotion)
        
        emotion_prediction = emotion_classifier.predict(X_test)[0]
        print(f"    ✅ Emotion prediction: '{test_text}' -> {emotion_prediction}")
        
        return True
    except Exception as e:
        print(f"❌ End-to-end test error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Sentiment Analysis Project Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Preprocessing", test_preprocessing),
        ("Vectorization", test_vectorization),
        ("Sentiment Models", test_sentiment_models),
        ("Emotion Models", test_emotion_models),
        ("Flask App", test_flask_app),
        ("End-to-End", test_end_to_end)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        print("\n🚀 To start the web application:")
        print("   python app.py")
        print("   Then open http://localhost:5000 in your browser")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 