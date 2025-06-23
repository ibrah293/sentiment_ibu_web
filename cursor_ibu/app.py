from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import uuid
from datetime import datetime
from data_preprocessing import TextPreprocessor
from text_vectorization import TextVectorizer
from sentiment_models import SentimentClassifier, EmotionClassifier, ModelEvaluator
from database import db

app = Flask(__name__)
CORS(app)

# Global variables to store models and preprocessors
sentiment_models = {}
emotion_models = {}
preprocessor = None
vectorizer = None

def load_or_create_models():
    """
    Load existing models or create new ones if they don't exist
    """
    global sentiment_models, emotion_models, preprocessor, vectorizer
    
    # Initialize preprocessor and vectorizer
    preprocessor = TextPreprocessor()
    vectorizer = TextVectorizer('tfidf')
    
    # Create sample data for training
    from sentiment_models import create_sample_emotion_data
    sample_df = create_sample_emotion_data()
    
    # Preprocess data
    processed_df = preprocessor.preprocess_dataset(sample_df, 'text', 'emotion')
    
    # Vectorize data
    X = vectorizer.fit_transform(processed_df['processed_text'])
    y_sentiment = processed_df['emotion'].map({
        'joy': 'positive', 'sadness': 'negative', 'anger': 'negative',
        'fear': 'negative', 'surprise': 'positive', 'disgust': 'negative', 'neutral': 'neutral'
    })
    y_emotion = processed_df['emotion']
    
    # Train sentiment models
    sentiment_model_types = ['logistic_regression', 'random_forest', 'naive_bayes']
    for model_type in sentiment_model_types:
        model = SentimentClassifier(model_type)
        model.fit(X, y_sentiment)
        sentiment_models[model_type] = model
    
    # Train emotion models
    emotion_model_types = ['logistic_regression', 'random_forest', 'naive_bayes']
    for model_type in emotion_model_types:
        model = EmotionClassifier(model_type)
        model.fit(X, y_emotion)
        emotion_models[model_type] = model

@app.route('/')
def index():
    """
    Serve the main HTML page
    """
    return render_template('index.html')

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    """
    Analyze sentiment of input text
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_type = data.get('model_type', 'logistic_regression')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess text
        processed_text = preprocessor.preprocess_text(text)
        
        # Vectorize text
        X = vectorizer.transform([processed_text])
        
        # Get sentiment prediction
        if model_type in sentiment_models:
            model = sentiment_models[model_type]
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Store in database
            db_id = db.store_sentiment_analysis(
                text=text,
                processed_text=processed_text,
                sentiment=prediction,
                confidence=float(max(probabilities)),
                model_type=model_type,
                probabilities={label: float(prob) for label, prob in zip(model.label_encoder.classes_, probabilities)}
            )
            
            # Create response
            response = {
                'id': db_id,
                'text': text,
                'processed_text': processed_text,
                'sentiment': prediction,
                'confidence': float(max(probabilities)),
                'model_type': model_type,
                'probabilities': {
                    label: float(prob) for label, prob in zip(model.label_encoder.classes_, probabilities)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
        else:
            return jsonify({'error': f'Model type {model_type} not available'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotion', methods=['POST'])
def analyze_emotion():
    """
    Analyze emotion of input text
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_type = data.get('model_type', 'logistic_regression')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess text
        processed_text = preprocessor.preprocess_text(text)
        
        # Vectorize text
        X = vectorizer.transform([processed_text])
        
        # Get emotion prediction
        if model_type in emotion_models:
            model = emotion_models[model_type]
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Store in database
            db_id = db.store_emotion_analysis(
                text=text,
                processed_text=processed_text,
                emotion=prediction,
                confidence=float(max(probabilities)),
                model_type=model_type,
                probabilities={label: float(prob) for label, prob in zip(model.label_encoder.classes_, probabilities)}
            )
            
            # Create response
            response = {
                'id': db_id,
                'text': text,
                'processed_text': processed_text,
                'emotion': prediction,
                'confidence': float(max(probabilities)),
                'model_type': model_type,
                'probabilities': {
                    label: float(prob) for label, prob in zip(model.label_encoder.classes_, probabilities)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
        else:
            return jsonify({'error': f'Model type {model_type} not available'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare_models', methods=['POST'])
def compare_models():
    """
    Compare different models on the same text
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        analysis_type = data.get('analysis_type', 'sentiment')  # 'sentiment' or 'emotion'
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess text
        processed_text = preprocessor.preprocess_text(text)
        
        # Vectorize text
        X = vectorizer.transform([processed_text])
        
        results = {}
        
        if analysis_type == 'sentiment':
            models = sentiment_models
        else:
            models = emotion_models
        
        for model_type, model in models.items():
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            results[model_type] = {
                'prediction': prediction,
                'confidence': float(max(probabilities)),
                'probabilities': {
                    label: float(prob) for label, prob in zip(model.label_encoder.classes_, probabilities)
                }
            }
        
        response = {
            'text': text,
            'processed_text': processed_text,
            'analysis_type': analysis_type,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_analysis', methods=['POST'])
def batch_analysis():
    """
    Analyze multiple texts at once
    """
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        analysis_type = data.get('analysis_type', 'sentiment')
        model_type = data.get('model_type', 'logistic_regression')
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        # Preprocess texts
        processed_texts = [preprocessor.preprocess_text(text) for text in texts]
        
        # Vectorize texts
        X = vectorizer.transform(processed_texts)
        
        # Get predictions
        if analysis_type == 'sentiment':
            model = sentiment_models.get(model_type)
        else:
            model = emotion_models.get(model_type)
        
        if not model:
            return jsonify({'error': f'Model type {model_type} not available'}), 400
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Store batch results in database
        confidences = [float(max(probs)) for probs in probabilities]
        db.store_batch_analysis(batch_id, texts, predictions, confidences, analysis_type, model_type)
        
        # Create response
        results = []
        for i, (text, processed_text, pred, probs) in enumerate(zip(texts, processed_texts, predictions, probabilities)):
            results.append({
                'id': i,
                'text': text,
                'processed_text': processed_text,
                'prediction': pred,
                'confidence': float(max(probs)),
                'probabilities': {
                    label: float(prob) for label, prob in zip(model.label_encoder.classes_, probs)
                }
            })
        
        response = {
            'batch_id': batch_id,
            'analysis_type': analysis_type,
            'model_type': model_type,
            'total_texts': len(texts),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """
    Get analysis history
    """
    try:
        analysis_type = request.args.get('type', 'both')  # 'sentiment', 'emotion', or 'both'
        limit = int(request.args.get('limit', 50))
        
        if analysis_type == 'sentiment':
            history = db.get_sentiment_history(limit)
        elif analysis_type == 'emotion':
            history = db.get_emotion_history(limit)
        else:
            # Get both types
            sentiment_history = db.get_sentiment_history(limit // 2)
            emotion_history = db.get_emotion_history(limit // 2)
            history = sentiment_history + emotion_history
            # Sort by timestamp
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            history = history[:limit]
        
        return jsonify({
            'history': history,
            'total': len(history),
            'type': analysis_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """
    Get analysis statistics
    """
    try:
        stats = db.get_analysis_statistics()
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_analyses():
    """
    Search for analyses containing specific text
    """
    try:
        search_term = request.args.get('q', '')
        analysis_type = request.args.get('type', 'both')
        limit = int(request.args.get('limit', 50))
        
        if not search_term:
            return jsonify({'error': 'No search term provided'}), 400
        
        results = db.search_analyses(search_term, analysis_type, limit)
        
        return jsonify({
            'search_term': search_term,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['GET'])
def export_data():
    """
    Export analysis data to CSV
    """
    try:
        filename = f"sentiment_analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        exported_file = db.export_to_csv(filename)
        
        return jsonify({
            'message': 'Data exported successfully',
            'filename': exported_file
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """
    Get information about available models
    """
    sentiment_info = {
        'available_models': list(sentiment_models.keys()),
        'supported_classes': list(sentiment_models['logistic_regression'].label_encoder.classes_) if sentiment_models else []
    }
    
    emotion_info = {
        'available_models': list(emotion_models.keys()),
        'supported_classes': list(emotion_models['logistic_regression'].label_encoder.classes_) if emotion_models else []
    }
    
    return jsonify({
        'sentiment_analysis': sentiment_info,
        'emotion_detection': emotion_info
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'sentiment_models_loaded': len(sentiment_models),
        'emotion_models_loaded': len(emotion_models),
        'preprocessor_ready': preprocessor is not None,
        'vectorizer_ready': vectorizer is not None,
        'database_ready': True
    })

if __name__ == '__main__':
    # Load or create models on startup
    print("Loading models...")
    load_or_create_models()
    print("Models loaded successfully!")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 