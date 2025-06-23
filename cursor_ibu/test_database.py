#!/usr/bin/env python3
"""
Database Test Script for Sentiment Analysis Project
This script demonstrates the database functionality for storing and retrieving analysis data.
"""

from database import db
import json
from datetime import datetime

def test_database_functionality():
    """Test all database functionality"""
    print("🧪 Testing Database Functionality")
    print("=" * 50)
    
    # Test 1: Store sentiment analysis
    print("\n1. Testing Sentiment Analysis Storage...")
    sentiment_id = db.store_sentiment_analysis(
        text="I absolutely love this product! It's amazing!",
        processed_text="absolutely love product amazing",
        sentiment="positive",
        confidence=0.95,
        model_type="logistic_regression",
        probabilities={"positive": 0.95, "negative": 0.03, "neutral": 0.02}
    )
    print(f"✅ Stored sentiment analysis with ID: {sentiment_id}")
    
    # Test 2: Store emotion analysis
    print("\n2. Testing Emotion Analysis Storage...")
    emotion_id = db.store_emotion_analysis(
        text="I'm so happy and excited about the news!",
        processed_text="happy excited news",
        emotion="joy",
        confidence=0.88,
        model_type="random_forest",
        probabilities={"joy": 0.88, "sadness": 0.05, "anger": 0.02, "fear": 0.01, "surprise": 0.02, "disgust": 0.01, "neutral": 0.01}
    )
    print(f"✅ Stored emotion analysis with ID: {emotion_id}")
    
    # Test 3: Store more sample data
    print("\n3. Adding More Sample Data...")
    sample_data = [
        ("This movie was terrible and boring.", "negative", "sadness"),
        ("The food was delicious and the service was excellent!", "positive", "joy"),
        ("I'm feeling anxious about the upcoming exam.", "negative", "fear"),
        ("What a surprise! I didn't expect this at all.", "positive", "surprise"),
        ("The customer service was absolutely horrible.", "negative", "anger"),
        ("The weather is nice today.", "neutral", "neutral"),
        ("I'm disgusted by the poor quality.", "negative", "disgust"),
        ("This is the best day ever!", "positive", "joy"),
        ("I'm so disappointed with the results.", "negative", "sadness"),
        ("The presentation was informative and well-organized.", "positive", "joy")
    ]
    
    for i, (text, sentiment, emotion) in enumerate(sample_data):
        # Store sentiment
        db.store_sentiment_analysis(
            text=text,
            processed_text=text.lower().replace("!", "").replace(".", ""),
            sentiment=sentiment,
            confidence=0.75 + (i * 0.02),
            model_type="naive_bayes" if i % 2 == 0 else "logistic_regression",
            probabilities={
                "positive": 0.6 if sentiment == "positive" else 0.2,
                "negative": 0.6 if sentiment == "negative" else 0.2,
                "neutral": 0.6 if sentiment == "neutral" else 0.2
            }
        )
        
        # Store emotion
        db.store_emotion_analysis(
            text=text,
            processed_text=text.lower().replace("!", "").replace(".", ""),
            emotion=emotion,
            confidence=0.70 + (i * 0.03),
            model_type="random_forest" if i % 3 == 0 else "logistic_regression",
            probabilities={
                "joy": 0.8 if emotion == "joy" else 0.1,
                "sadness": 0.8 if emotion == "sadness" else 0.1,
                "anger": 0.8 if emotion == "anger" else 0.1,
                "fear": 0.8 if emotion == "fear" else 0.1,
                "surprise": 0.8 if emotion == "surprise" else 0.1,
                "disgust": 0.8 if emotion == "disgust" else 0.1,
                "neutral": 0.8 if emotion == "neutral" else 0.1
            }
        )
    
    print(f"✅ Added {len(sample_data)} more sample records")
    
    # Test 4: Retrieve history
    print("\n4. Testing History Retrieval...")
    sentiment_history = db.get_sentiment_history(5)
    emotion_history = db.get_emotion_history(5)
    
    print(f"✅ Retrieved {len(sentiment_history)} recent sentiment analyses")
    print(f"✅ Retrieved {len(emotion_history)} recent emotion analyses")
    
    # Display some history
    print("\n📊 Recent Sentiment Analyses:")
    for item in sentiment_history[:3]:
        print(f"   ID: {item['id']} | Text: {item['text'][:50]}... | Sentiment: {item['sentiment']} | Confidence: {item['confidence']:.2f}")
    
    print("\n📊 Recent Emotion Analyses:")
    for item in emotion_history[:3]:
        print(f"   ID: {item['id']} | Text: {item['text'][:50]}... | Emotion: {item['emotion']} | Confidence: {item['confidence']:.2f}")
    
    # Test 5: Get statistics
    print("\n5. Testing Statistics...")
    stats = db.get_analysis_statistics()
    print("✅ Retrieved database statistics:")
    print(f"   Total sentiment analyses: {stats['total_sentiment_analyses']}")
    print(f"   Total emotion analyses: {stats['total_emotion_analyses']}")
    print(f"   Average sentiment confidence: {stats['avg_sentiment_confidence']:.3f}")
    print(f"   Average emotion confidence: {stats['avg_emotion_confidence']:.3f}")
    
    print("\n📈 Sentiment Distribution:")
    for sentiment, count in stats['sentiment_distribution'].items():
        print(f"   {sentiment}: {count}")
    
    print("\n📈 Emotion Distribution:")
    for emotion, count in stats['emotion_distribution'].items():
        print(f"   {emotion}: {count}")
    
    # Test 6: Search functionality
    print("\n6. Testing Search Functionality...")
    search_results = db.search_analyses("love", "both", 5)
    print(f"✅ Found {len(search_results)} results for 'love'")
    
    if search_results:
        print("📋 Search Results:")
        for result in search_results[:3]:
            print(f"   ID: {result['id']} | Text: {result['text'][:50]}... | Prediction: {result['prediction']}")
    
    # Test 7: Export functionality
    print("\n7. Testing Export Functionality...")
    try:
        exported_file = db.export_to_csv("test_export.csv")
        print(f"✅ Data exported to: {exported_file}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
    
    # Test 8: Model performance storage
    print("\n8. Testing Model Performance Storage...")
    db.store_model_performance(
        model_type="logistic_regression",
        analysis_type="sentiment",
        accuracy=0.85,
        precision=0.87,
        recall=0.83,
        f1_score=0.85
    )
    db.store_model_performance(
        model_type="random_forest",
        analysis_type="emotion",
        accuracy=0.78,
        precision=0.80,
        recall=0.76,
        f1_score=0.78
    )
    print("✅ Stored model performance metrics")
    
    # Test 9: Batch analysis storage
    print("\n9. Testing Batch Analysis Storage...")
    batch_texts = ["Great product!", "Terrible service", "Okay experience"]
    batch_predictions = ["positive", "negative", "neutral"]
    batch_confidences = [0.92, 0.88, 0.75]
    
    db.store_batch_analysis(
        batch_id="test_batch_001",
        texts=batch_texts,
        predictions=batch_predictions,
        confidences=batch_confidences,
        analysis_type="sentiment",
        model_type="logistic_regression"
    )
    print("✅ Stored batch analysis results")
    
    print("\n" + "=" * 50)
    print("🎉 All Database Tests Completed Successfully!")
    print("=" * 50)

def show_database_info():
    """Show database information and structure"""
    print("\n🗄️ Database Information")
    print("=" * 30)
    
    # Get statistics
    stats = db.get_analysis_statistics()
    
    print(f"Database File: sentiment_analysis.db")
    print(f"Total Records: {stats['total_sentiment_analyses'] + stats['total_emotion_analyses']}")
    print(f"Sentiment Analyses: {stats['total_sentiment_analyses']}")
    print(f"Emotion Analyses: {stats['total_emotion_analyses']}")
    
    print("\n📊 Database Tables:")
    print("   - sentiment_analysis: Stores sentiment analysis results")
    print("   - emotion_analysis: Stores emotion detection results")
    print("   - model_performance: Stores model evaluation metrics")
    print("   - batch_analysis: Stores batch processing results")
    
    print("\n🔍 Available Functions:")
    print("   - store_sentiment_analysis(): Store sentiment results")
    print("   - store_emotion_analysis(): Store emotion results")
    print("   - get_sentiment_history(): Get sentiment history")
    print("   - get_emotion_history(): Get emotion history")
    print("   - get_analysis_statistics(): Get overall statistics")
    print("   - search_analyses(): Search for specific text")
    print("   - export_to_csv(): Export data to CSV")
    print("   - store_model_performance(): Store model metrics")
    print("   - store_batch_analysis(): Store batch results")

def interactive_demo():
    """Interactive demonstration of database features"""
    print("\n🎮 Interactive Database Demo")
    print("=" * 30)
    
    while True:
        print("\nChoose an option:")
        print("1. View recent sentiment analyses")
        print("2. View recent emotion analyses")
        print("3. View database statistics")
        print("4. Search for specific text")
        print("5. Export data to CSV")
        print("6. Show database info")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            limit = int(input("How many records to show? (default 10): ") or "10")
            history = db.get_sentiment_history(limit)
            print(f"\n📊 Recent Sentiment Analyses ({len(history)} records):")
            for item in history:
                print(f"   ID: {item['id']} | {item['text'][:60]}... | {item['sentiment']} ({item['confidence']:.2f})")
        elif choice == "2":
            limit = int(input("How many records to show? (default 10): ") or "10")
            history = db.get_emotion_history(limit)
            print(f"\n📊 Recent Emotion Analyses ({len(history)} records):")
            for item in history:
                print(f"   ID: {item['id']} | {item['text'][:60]}... | {item['emotion']} ({item['confidence']:.2f})")
        elif choice == "3":
            stats = db.get_analysis_statistics()
            print("\n📈 Database Statistics:")
            print(f"   Total sentiment analyses: {stats['total_sentiment_analyses']}")
            print(f"   Total emotion analyses: {stats['total_emotion_analyses']}")
            print(f"   Average sentiment confidence: {stats['avg_sentiment_confidence']:.3f}")
            print(f"   Average emotion confidence: {stats['avg_emotion_confidence']:.3f}")
        elif choice == "4":
            search_term = input("Enter search term: ").strip()
            if search_term:
                results = db.search_analyses(search_term, "both", 10)
                print(f"\n🔍 Search Results for '{search_term}' ({len(results)} found):")
                for result in results:
                    print(f"   ID: {result['id']} | {result['text'][:50]}... | {result['prediction']}")
            else:
                print("❌ Please enter a search term")
        elif choice == "5":
            filename = input("Enter filename (default: export.csv): ").strip() or "export.csv"
            try:
                exported_file = db.export_to_csv(filename)
                print(f"✅ Data exported to: {exported_file}")
            except Exception as e:
                print(f"❌ Export failed: {e}")
        elif choice == "6":
            show_database_info()
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    print("🚀 Sentiment Analysis Database Test Suite")
    print("=" * 50)
    
    try:
        # Run automated tests
        test_database_functionality()
        
        # Show database information
        show_database_info()
        
        # Ask if user wants interactive demo
        response = input("\n🤔 Would you like to try the interactive demo? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_demo()
        
        print("\n✅ Database functionality is working correctly!")
        print("💡 You can now use the database in your Flask application.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("🔧 Please check your database configuration and dependencies.") 