#!/usr/bin/env python3
"""
Generate Visualizations for Sentiment Analysis Project
This script creates comprehensive visualizations including AUC-ROC plots for the trained models.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import our project modules
from data_preprocessing import TextPreprocessor
from text_vectorization import TextVectorizer
from sentiment_models import SentimentClassifier, EmotionClassifier, create_sample_emotion_data
from model_visualization import ModelVisualizer

def prepare_data_for_visualization():
    """
    Prepare data and models for visualization
    """
    print("🔄 Preparing data and models for visualization...")
    
    # Create sample data
    print("📊 Creating sample dataset...")
    sample_df = create_sample_emotion_data()
    
    # Initialize preprocessor and vectorizer
    print("🔧 Initializing preprocessor and vectorizer...")
    preprocessor = TextPreprocessor()
    vectorizer = TextVectorizer('tfidf')
    
    # Preprocess data
    print("🧹 Preprocessing text data...")
    processed_df = preprocessor.preprocess_dataset(sample_df, 'text', 'emotion')
    
    # Vectorize data
    print("📝 Vectorizing text data...")
    X = vectorizer.fit_transform(processed_df['processed_text'])
    
    # Prepare labels
    y_sentiment = processed_df['emotion'].map({
        'joy': 'positive', 'sadness': 'negative', 'anger': 'negative',
        'fear': 'negative', 'surprise': 'positive', 'disgust': 'negative', 'neutral': 'neutral'
    })
    y_emotion = processed_df['emotion']
    
    # Split data for training and testing
    print("✂️ Splitting data into train/test sets...")
    X_train, X_test, y_sentiment_train, y_sentiment_test = train_test_split(
        X, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
    )
    
    _, _, y_emotion_train, y_emotion_test = train_test_split(
        X, y_emotion, test_size=0.2, random_state=42, stratify=y_emotion
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_sentiment_train': y_sentiment_train,
        'y_sentiment_test': y_sentiment_test,
        'y_emotion_train': y_emotion_train,
        'y_emotion_test': y_emotion_test,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor
    }

def train_models_for_visualization(data_dict):
    """
    Train models for visualization
    """
    print("🤖 Training models for visualization...")
    
    X_train = data_dict['X_train']
    y_sentiment_train = data_dict['y_sentiment_train']
    y_emotion_train = data_dict['y_emotion_train']
    
    # Train sentiment models
    print("📊 Training sentiment analysis models...")
    sentiment_models = {}
    sentiment_model_types = ['logistic_regression', 'random_forest', 'naive_bayes']
    
    for model_type in sentiment_model_types:
        print(f"   Training {model_type} for sentiment...")
        model = SentimentClassifier(model_type)
        model.fit(X_train, y_sentiment_train)
        sentiment_models[model_type] = model
    
    # Train emotion models
    print("😊 Training emotion detection models...")
    emotion_models = {}
    emotion_model_types = ['logistic_regression', 'random_forest', 'naive_bayes']
    
    for model_type in emotion_model_types:
        print(f"   Training {model_type} for emotion...")
        model = EmotionClassifier(model_type)
        model.fit(X_train, y_emotion_train)
        emotion_models[model_type] = model
    
    return sentiment_models, emotion_models

def generate_visualizations(sentiment_models, emotion_models, data_dict):
    """
    Generate comprehensive visualizations
    """
    print("🎨 Generating comprehensive visualizations...")
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Create plots directory
    plots_dir = 'model_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get feature names from vectorizer
    feature_names = None
    try:
        feature_names = data_dict['vectorizer'].get_feature_names_out()
    except:
        pass
    
    # Generate sentiment analysis visualizations
    print("\n📊 Generating sentiment analysis visualizations...")
    try:
        visualizer.create_comprehensive_report(
            models=sentiment_models,
            X_train=data_dict['X_train'],
            X_test=data_dict['X_test'],
            y_train=data_dict['y_sentiment_train'],
            y_test=data_dict['y_sentiment_test'],
            task_type='sentiment',
            feature_names=feature_names,
            save_dir=os.path.join(plots_dir, 'sentiment')
        )
    except Exception as e:
        print(f"❌ Error generating sentiment visualizations: {e}")
    
    # Generate emotion detection visualizations
    print("\n😊 Generating emotion detection visualizations...")
    try:
        visualizer.create_comprehensive_report(
            models=emotion_models,
            X_train=data_dict['X_train'],
            X_test=data_dict['X_test'],
            y_train=data_dict['y_emotion_train'],
            y_test=data_dict['y_emotion_test'],
            task_type='emotion',
            feature_names=feature_names,
            save_dir=os.path.join(plots_dir, 'emotion')
        )
    except Exception as e:
        print(f"❌ Error generating emotion visualizations: {e}")

def print_model_summary(sentiment_models, emotion_models, data_dict):
    """
    Print a summary of model performance
    """
    print("\n📋 Model Performance Summary")
    print("=" * 50)
    
    # Sentiment models summary
    print("\n📊 Sentiment Analysis Models:")
    print("-" * 30)
    for model_name, model in sentiment_models.items():
        try:
            y_pred = model.predict(data_dict['X_test'])
            accuracy = accuracy_score(data_dict['y_sentiment_test'], y_pred)
            print(f"   {model_name.replace('_', ' ').title()}: {accuracy:.3f} accuracy")
        except Exception as e:
            print(f"   {model_name.replace('_', ' ').title()}: Error - {e}")
    
    # Emotion models summary
    print("\n😊 Emotion Detection Models:")
    print("-" * 30)
    for model_name, model in emotion_models.items():
        try:
            y_pred = model.predict(data_dict['X_test'])
            accuracy = accuracy_score(data_dict['y_emotion_test'], y_pred)
            print(f"   {model_name.replace('_', ' ').title()}: {accuracy:.3f} accuracy")
        except Exception as e:
            print(f"   {model_name.replace('_', ' ').title()}: Error - {e}")

def main():
    """
    Main function to generate all visualizations
    """
    print("🚀 Sentiment Analysis Model Visualization Generator")
    print("=" * 60)
    
    try:
        # Step 1: Prepare data
        data_dict = prepare_data_for_visualization()
        
        # Step 2: Train models
        sentiment_models, emotion_models = train_models_for_visualization(data_dict)
        
        # Step 3: Print summary
        print_model_summary(sentiment_models, emotion_models, data_dict)
        
        # Step 4: Generate visualizations
        generate_visualizations(sentiment_models, emotion_models, data_dict)
        
        print("\n✅ Visualization generation completed successfully!")
        print("\n📁 Generated files are saved in the 'model_plots' directory:")
        print("   - sentiment/ - Sentiment analysis visualizations")
        print("   - emotion/ - Emotion detection visualizations")
        
        print("\n🎯 Key visualizations generated:")
        print("   📈 AUC-ROC curves")
        print("   📊 Confusion matrices")
        print("   📋 Model comparison charts")
        print("   🔍 Feature importance plots")
        print("   📚 Learning curves")
        
    except Exception as e:
        print(f"\n❌ Error during visualization generation: {e}")
        print("🔧 Please check your dependencies and try again.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All visualizations generated successfully!")
        print("💡 You can now view the plots in the 'model_plots' directory.")
    else:
        print("\n💥 Visualization generation failed.")
        sys.exit(1) 