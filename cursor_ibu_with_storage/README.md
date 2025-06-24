# Sentiment Analysis & Emotion Detection Project

A comprehensive sentiment analysis and emotion detection system with advanced text preprocessing, multiple machine learning models, and a beautiful web interface.

## 🚀 Features

### Data Preprocessing
- **Text Cleaning**: Remove URLs, emails, numbers, and special characters
- **Tokenization**: Split text into individual words
- **Stopword Removal**: Remove common words that don't add meaning
- **Lemmatization**: Convert words to their base form
- **Stemming**: Alternative to lemmatization for word reduction

### Text Vectorization
- **TF-IDF Vectorization**: Term frequency-inverse document frequency
- **Count Vectorization**: Simple word frequency counting
- **LDA (Latent Dirichlet Allocation)**: Topic modeling
- **LSA (Latent Semantic Analysis)**: Dimensionality reduction

### Sentiment Analysis Models
- **Logistic Regression**: Fast and interpretable
- **Random Forest**: Robust ensemble method
- **Support Vector Machine (SVM)**: Good for high-dimensional data
- **Naive Bayes**: Probabilistic classifier
- **Gradient Boosting**: Advanced ensemble method
- **Fine-tuned Transformers**: Pre-trained models (BERT, DistilBERT)

### Emotion Detection
- **Multiclass Classification**: 7 emotion categories
  - Joy 😊
  - Sadness 😢
  - Anger 😠
  - Fear 😨
  - Surprise 😲
  - Disgust 🤢
  - Neutral 😐

### Model Evaluation
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

### Web Interface
- **Modern UI**: Beautiful gradient design
- **Real-time Analysis**: Instant results
- **Model Comparison**: Compare different models
- **Batch Processing**: Analyze multiple texts at once
- **Responsive Design**: Works on all devices

## 📋 Requirements

- Python 3.8+
- Flask
- scikit-learn
- NLTK
- transformers
- torch
- pandas
- numpy

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sentiment-analysis-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## 🚀 Usage

### Running the Web Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Use the web interface**
   - Enter text for sentiment analysis
   - Choose different models
   - Compare model performances
   - Analyze multiple texts at once

### Using the Python Modules

#### Data Preprocessing
```python
from data_preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned_text = preprocessor.preprocess_text("I love this product! It's amazing!")
print(cleaned_text)  # "love product amazing"
```

#### Text Vectorization
```python
from text_vectorization import TextVectorizer

vectorizer = TextVectorizer('tfidf')
X = vectorizer.fit_transform(["I love this", "I hate this", "This is okay"])
```

#### Sentiment Analysis
```python
from sentiment_models import SentimentClassifier

classifier = SentimentClassifier('logistic_regression')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

#### Emotion Detection
```python
from sentiment_models import EmotionClassifier

emotion_classifier = EmotionClassifier('random_forest')
emotion_classifier.fit(X_train, y_emotions)
emotions = emotion_classifier.predict(X_test)
```

#### Model Evaluation
```python
from sentiment_models import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test, "My Model")
evaluator.print_results("My Model")
```

## 📊 API Endpoints

### Sentiment Analysis
```http
POST /api/sentiment
Content-Type: application/json

{
    "text": "I love this product!",
    "model_type": "logistic_regression"
}
```

### Emotion Detection
```http
POST /api/emotion
Content-Type: application/json

{
    "text": "I'm so happy today!",
    "model_type": "random_forest"
}
```

### Model Comparison
```http
POST /api/compare_models
Content-Type: application/json

{
    "text": "This is amazing!",
    "analysis_type": "sentiment"
}
```

### Batch Analysis
```http
POST /api/batch_analysis
Content-Type: application/json

{
    "texts": ["I love this", "I hate this", "This is okay"],
    "analysis_type": "sentiment",
    "model_type": "logistic_regression"
}
```

### Health Check
```http
GET /api/health
```

## 🏗️ Project Structure

```
sentiment-analysis-project/
├── app.py                 # Flask web application
├── data_preprocessing.py  # Text preprocessing module
├── text_vectorization.py  # Text vectorization module
├── sentiment_models.py    # ML models and evaluation
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/
│   └── index.html        # Web interface
└── models/               # Saved model files (created automatically)
```

## 🔧 Configuration

### Model Parameters
You can customize model parameters in the respective classes:

```python
# Logistic Regression
classifier = SentimentClassifier('logistic_regression')
classifier.create_model(C=1.0, max_iter=1000)

# Random Forest
classifier = SentimentClassifier('random_forest')
classifier.create_model(n_estimators=200, max_depth=10)

# Transformer Models
transformer = TransformerSentimentClassifier('distilbert-base-uncased')
transformer.fit_transformer(texts, labels, epochs=5, learning_rate=1e-5)
```

### Vectorization Parameters
```python
vectorizer = TextVectorizer('tfidf')
vectorizer.create_vectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
)
```

## 📈 Performance

### Sample Results
- **Logistic Regression**: Accuracy ~85%, F1 ~0.84
- **Random Forest**: Accuracy ~87%, F1 ~0.86
- **Naive Bayes**: Accuracy ~82%, F1 ~0.81

### Model Comparison
The system automatically compares different models and provides detailed metrics including:
- Confusion matrices
- Classification reports
- Precision-recall curves
- Feature importance (for tree-based models)

## 🎯 Use Cases

1. **Social Media Monitoring**: Analyze customer sentiment on social platforms
2. **Customer Feedback**: Process and categorize customer reviews
3. **Market Research**: Understand public opinion about products/services
4. **Content Moderation**: Detect inappropriate or harmful content
5. **Mental Health**: Analyze emotional states in text
6. **Brand Monitoring**: Track brand sentiment over time

## 🔮 Future Enhancements

- [ ] Real-time streaming analysis
- [ ] Multi-language support
- [ ] Advanced transformer models (GPT, T5)
- [ ] Sentiment trend analysis
- [ ] API rate limiting and authentication
- [ ] Docker containerization
- [ ] Cloud deployment options
- [ ] Mobile app development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NLTK for natural language processing tools
- scikit-learn for machine learning algorithms
- Hugging Face for transformer models
- Flask for the web framework
- The open-source community for inspiration and tools

## 📞 Support

If you encounter any issues or have questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Contact the maintainers

---

**Happy Analyzing! 🎉** 