#!/usr/bin/env python3
"""
Demo script for the Sentiment Analysis & Emotion Detection Project
This script demonstrates the main features and capabilities of the project.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def demo_preprocessing():
    """Demonstrate text preprocessing capabilities"""
    print("🔧 TEXT PREPROCESSING DEMO")
    print("=" * 40)
    
    from data_preprocessing import TextPreprocessor
    
    # Create preprocessor
    preprocessor = TextPreprocessor()
    
    # Sample texts with various issues
    sample_texts = [
        "I LOVE this product! It's AMAZING!!! Check it out at https://example.com",
        "This is the WORST purchase I've ever made. Terrible quality! Contact me at user@email.com",
        "The product is okay, nothing special but gets the job done. Price: $99.99",
        "Absolutely fantastic! Best decision ever to buy this. #amazing #loveit",
        "Disappointed with the service. Very poor customer support. :("
    ]
    
    print("Original texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    print("\nAfter preprocessing:")
    for i, text in enumerate(sample_texts, 1):
        processed = preprocessor.preprocess_text(text)
        print(f"{i}. {processed}")
    
    print("\n" + "=" * 40)

def demo_vectorization():
    """Demonstrate text vectorization methods"""
    print("📊 TEXT VECTORIZATION DEMO")
    print("=" * 40)
    
    from text_vectorization import TextVectorizer
    
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
        print(f"\n{method.upper()} Vectorization:")
        vectorizer = TextVectorizer(method)
        X = vectorizer.fit_transform(texts, max_features=50)
        
        print(f"Shape: {X.shape}")
        print(f"Feature names (first 10): {vectorizer.get_feature_names()[:10]}")
        
        if method == 'tfidf':
            top_features = vectorizer.get_top_features(5)
            print(f"Top features: {top_features}")
    
    print("\n" + "=" * 40)

def demo_sentiment_analysis():
    """Demonstrate sentiment analysis capabilities"""
    print("😊 SENTIMENT ANALYSIS DEMO")
    print("=" * 40)
    
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
    
    print("Training and evaluating different models:")
    for model_type in models:
        print(f"\n{model_type.replace('_', ' ').title()}:")
        classifier = SentimentClassifier(model_type)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        results = evaluator.evaluate_model(classifier, X_test, y_test, model_type)
        print(f"  Accuracy: {results['accuracy']:.3f}")
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall: {results['recall']:.3f}")
        print(f"  F1 Score: {results['f1_score']:.3f}")
    
    # Test predictions
    print("\nTesting predictions:")
    test_texts = [
        "I love this amazing product!",
        "This is terrible, I hate it!",
        "It's okay, nothing special."
    ]
    
    classifier = SentimentClassifier('logistic_regression')
    classifier.fit(X, y)
    
    for text in test_texts:
        processed_text = preprocessor.preprocess_text(text)
        X_test = vectorizer.transform([processed_text])
        prediction = classifier.predict(X_test)[0]
        probabilities = classifier.predict_proba(X_test)[0]
        
        print(f"  '{text}' -> {prediction} (confidence: {max(probabilities):.3f})")
    
    print("\n" + "=" * 40)

def demo_emotion_detection():
    """Demonstrate emotion detection capabilities"""
    print("😍 EMOTION DETECTION DEMO")
    print("=" * 40)
    
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
    
    # Train emotion classifier
    emotion_classifier = EmotionClassifier('logistic_regression')
    emotion_classifier.fit(X, y)
    
    # Test emotion predictions
    test_texts = [
        "I'm so happy today! Everything is going great!",
        "I feel really sad and depressed about what happened.",
        "I'm so angry at this situation! This is unacceptable!",
        "I'm scared about what might happen next.",
        "Wow! I can't believe this amazing news!",
        "This is disgusting, I can't stand it.",
        "I'm feeling neutral about this situation."
    ]
    
    print("Testing emotion detection:")
    for text in test_texts:
        processed_text = preprocessor.preprocess_text(text)
        X_test = vectorizer.transform([processed_text])
        prediction = emotion_classifier.predict(X_test)[0]
        probabilities = emotion_classifier.predict_proba(X_test)[0]
        
        print(f"  '{text}' -> {prediction} (confidence: {max(probabilities):.3f})")
    
    print("\n" + "=" * 40)

def demo_model_comparison():
    """Demonstrate model comparison capabilities"""
    print("🔍 MODEL COMPARISON DEMO")
    print("=" * 40)
    
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
    
    # Compare models
    models = ['logistic_regression', 'random_forest', 'naive_bayes']
    evaluator = ModelEvaluator()
    
    for model_type in models:
        classifier = SentimentClassifier(model_type)
        classifier.fit(X_train, y_train)
        evaluator.evaluate_model(classifier, X_test, y_test, model_type)
    
    # Get comparison results
    comparison_df = evaluator.compare_models()
    print("Model Comparison Results:")
    print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 40)

def demo_api_endpoints():
    """Demonstrate API endpoint functionality"""
    print("🌐 API ENDPOINTS DEMO")
    print("=" * 40)
    
    try:
        from app import app
        
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/api/health')
            print(f"Health Check: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"  Status: {data['status']}")
                print(f"  Sentiment Models: {data['sentiment_models_loaded']}")
                print(f"  Emotion Models: {data['emotion_models_loaded']}")
            
            # Test model info endpoint
            response = client.get('/api/model_info')
            print(f"\nModel Info: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"  Sentiment Models: {data['sentiment_analysis']['available_models']}")
                print(f"  Emotion Models: {data['emotion_detection']['available_models']}")
            
            # Test sentiment analysis endpoint
            response = client.post('/api/sentiment', 
                                 json={'text': 'I love this product!', 'model_type': 'logistic_regression'})
            print(f"\nSentiment Analysis: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"  Sentiment: {data['sentiment']}")
                print(f"  Confidence: {data['confidence']:.3f}")
            
            # Test emotion detection endpoint
            response = client.post('/api/emotion', 
                                 json={'text': 'I am so happy today!', 'model_type': 'logistic_regression'})
            print(f"\nEmotion Detection: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"  Emotion: {data['emotion']}")
                print(f"  Confidence: {data['confidence']:.3f}")
    
    except Exception as e:
        print(f"API demo error: {e}")
    
    print("\n" + "=" * 40)

def main():
    """Run all demos"""
    print("🚀 SENTIMENT ANALYSIS & EMOTION DETECTION PROJECT DEMO")
    print("=" * 60)
    print("This demo showcases all the main features of the project.\n")
    
    demos = [
        ("Text Preprocessing", demo_preprocessing),
        ("Text Vectorization", demo_vectorization),
        ("Sentiment Analysis", demo_sentiment_analysis),
        ("Emotion Detection", demo_emotion_detection),
        ("Model Comparison", demo_model_comparison),
        ("API Endpoints", demo_api_endpoints)
    ]
    
    for demo_name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 DEMO COMPLETED!")
    print("\nTo start the web application:")
    print("  python app.py")
    print("  Then open http://localhost:5000 in your browser")
    print("\nTo run tests:")
    print("  python test_project.py")

if __name__ == "__main__":
    main() 