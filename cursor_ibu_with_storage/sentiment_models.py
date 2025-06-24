import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

class SentimentClassifier:
    def __init__(self, model_type='logistic_regression'):
        """
        Initialize sentiment classifier
        
        Args:
            model_type (str): 'logistic_regression', 'random_forest', 'svm', 'naive_bayes', 'gradient_boosting', or 'transformer'
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def create_model(self, **kwargs):
        """
        Create the specified model
        """
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **kwargs
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                **kwargs
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                random_state=42,
                probability=True,
                **kwargs
            )
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB(**kwargs)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                **kwargs
            )
        elif self.model_type == 'transformer':
            # This will be handled separately for transformer models
            print("Warning: Transformer models are not available without torch and transformers packages")
            pass
        else:
            raise ValueError("model_type must be one of: 'logistic_regression', 'random_forest', 'svm', 'naive_bayes', 'gradient_boosting', 'transformer'")
    
    def fit(self, X, y):
        """
        Fit the model
        """
        if self.model_type != 'transformer':
            if self.model is None:
                self.create_model()
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Fit the model
            self.model.fit(X, y_encoded)
            self.is_fitted = True
        else:
            raise ValueError("For transformer models, use fit_transformer method")
    
    def predict(self, X):
        """
        Make predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath):
        """
        Save the fitted model
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a fitted model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls(model_data['model_type'])
        classifier.model = model_data['model']
        classifier.label_encoder = model_data['label_encoder']
        classifier.is_fitted = model_data['is_fitted']
        
        return classifier

# Comment out the TransformerSentimentClassifier class for now
# class TransformerSentimentClassifier:
#     def __init__(self, model_name='distilbert-base-uncased'):
#         """
#         Initialize transformer-based sentiment classifier
#         
#         Args:
#             model_name (str): HuggingFace model name
#         """
#         self.model_name = model_name
#         self.tokenizer = None
#         self.model = None
#         self.label_encoder = LabelEncoder()
#         self.is_fitted = False
        
#     def setup_model(self, num_labels=3):
#         """
#         Setup the transformer model
#         """
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             self.model_name, 
#             num_labels=num_labels
#         )
    
#     def tokenize_data(self, texts, max_length=128):
#         """
#         Tokenize the input texts
#         """
#         return self.tokenizer(
#             texts,
#             truncation=True,
#             padding=True,
#             max_length=max_length,
#             return_tensors="pt"
#         )
    
#     def fit_transformer(self, texts, labels, epochs=3, batch_size=16, learning_rate=2e-5):
#         """
#         Fine-tune the transformer model
#         """
#         # Setup model if not already done
#         if self.model is None:
#             unique_labels = list(set(labels))
#             self.setup_model(num_labels=len(unique_labels))
        
#         # Encode labels
#         y_encoded = self.label_encoder.fit_transform(labels)
        
#         # Create dataset
#         dataset = Dataset.from_dict({
#             'text': texts,
#             'label': y_encoded
#         })
        
#         # Tokenize dataset
#         def tokenize_function(examples):
#             return self.tokenizer(
#                 examples['text'],
#                 truncation=True,
#                 padding=True,
#                 max_length=128
#             )
        
#         tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
#         # Training arguments
#         training_args = TrainingArguments(
#             output_dir="./results",
#             num_train_epochs=epochs,
#             per_device_train_batch_size=batch_size,
#             per_device_eval_batch_size=batch_size,
#             warmup_steps=500,
#             weight_decay=0.01,
#             logging_dir='./logs',
#             logging_steps=10,
#             learning_rate=learning_rate,
#             save_strategy="epoch",
#             evaluation_strategy="epoch",
#             load_best_model_at_end=True,
#         )
        
#         # Create trainer
#         trainer = Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=tokenized_dataset,
#             eval_dataset=tokenized_dataset,
#         )
        
#         # Train the model
#         trainer.train()
#         self.is_fitted = True
    
#     def predict_transformer(self, texts):
#         """
#         Make predictions with transformer model
#         """
#         if not self.is_fitted:
#             raise ValueError("Model must be fitted before making predictions")
        
#         self.model.eval()
#         predictions = []
        
#         with torch.no_grad():
#             for text in texts:
#                 inputs = self.tokenize_data([text])
#                 outputs = self.model(**inputs)
#                 pred = torch.argmax(outputs.logits, dim=1)
#                 predictions.append(pred.item())
        
#         return self.label_encoder.inverse_transform(predictions)

class EmotionClassifier:
    def __init__(self, model_type='logistic_regression'):
        """
        Initialize emotion classifier for multiclass classification
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
        
    def create_model(self, **kwargs):
        """
        Create the specified model for emotion classification
        """
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='multinomial',
                **kwargs
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                **kwargs
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                random_state=42,
                probability=True,
                **kwargs
            )
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB(**kwargs)
        else:
            raise ValueError("model_type must be one of: 'logistic_regression', 'random_forest', 'svm', 'naive_bayes'")
    
    def fit(self, X, y):
        """
        Fit the emotion classifier
        """
        if self.model is None:
            self.create_model()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Fit the model
        self.model.fit(X, y_encoded)
        self.is_fitted = True
    
    def predict(self, X):
        """
        Predict emotions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X):
        """
        Get emotion prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)

    def save_model(self, filepath):
        """
        Save the fitted emotion model
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filepath):
        """
        Load a fitted emotion model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        classifier = cls(model_data['model_type'])
        classifier.model = model_data['model']
        classifier.label_encoder = model_data['label_encoder']
        classifier.is_fitted = model_data['is_fitted']
        return classifier

class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate a model and return metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Get detailed classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_rep,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        return self.results[model_name]
    
    def print_results(self, model_name="Model", model=None):
        """
        Print evaluation results
        """
        if model_name not in self.results:
            print(f"No results found for {model_name}")
            return
        
        results = self.results[model_name]
        
        print(f"\n=== {model_name} Evaluation Results ===")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        
        print(f"\nClassification Report:")
        if model and hasattr(model, 'label_encoder'):
            print(classification_report(
                y_test, 
                results['predictions'], 
                target_names=model.label_encoder.classes_
            ))
        else:
            print(classification_report(
                y_test, 
                results['predictions']
            ))
    
    def compare_models(self, model_names=None):
        """
        Compare multiple models
        """
        if model_names is None:
            model_names = list(self.results.keys())
        
        comparison_data = []
        for name in model_names:
            if name in self.results:
                results = self.results[name]
                comparison_data.append({
                    'Model': name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1 Score': results['f1_score']
                })
        
        return pd.DataFrame(comparison_data)

def create_sample_emotion_data():
    """
    Create sample emotion data for testing
    """
    # Create more diverse sample texts for each emotion
    emotion_texts = {
        'joy': [
            "I'm so happy today! Everything is going great!",
            "What a wonderful day! I'm overjoyed!",
            "This is fantastic news! I'm thrilled!",
            "I'm feeling so joyful and excited!",
            "What an amazing experience! I'm delighted!",
            "I'm so grateful for this wonderful moment!",
            "This makes me so happy and content!",
            "I'm feeling absolutely ecstatic!",
            "What a beautiful day! I'm so pleased!",
            "I'm over the moon with joy!"
        ],
        'sadness': [
            "I feel really sad and depressed about what happened.",
            "I'm devastated by this terrible loss.",
            "This is so heartbreaking and painful.",
            "I'm feeling so down and hopeless.",
            "This makes me so sad and miserable.",
            "I'm overwhelmed with grief and sorrow.",
            "This is such a depressing situation.",
            "I feel so lonely and sad right now.",
            "This loss is unbearable and tragic.",
            "I'm feeling so blue and melancholic."
        ],
        'anger': [
            "I'm so angry at this situation! This is unacceptable!",
            "I'm furious about this injustice!",
            "This makes me absolutely livid!",
            "I'm so mad about what happened!",
            "This is infuriating and outrageous!",
            "I'm boiling with rage right now!",
            "This makes me so angry and frustrated!",
            "I'm absolutely furious about this!",
            "This is so aggravating and maddening!",
            "I'm seething with anger!"
        ],
        'fear': [
            "I'm scared about what might happen next.",
            "I'm terrified of what's coming.",
            "This is so frightening and alarming.",
            "I'm feeling so anxious and afraid.",
            "This makes me so scared and worried.",
            "I'm petrified of what might happen.",
            "This is absolutely terrifying!",
            "I'm so afraid of the consequences.",
            "This fills me with dread and fear.",
            "I'm feeling so panicked and scared."
        ],
        'surprise': [
            "Wow! I can't believe this amazing news!",
            "Incredible! This is beyond my expectations!",
            "Oh my goodness! This is unbelievable!",
            "I'm completely shocked by this!",
            "This is absolutely astonishing!",
            "I'm so surprised and amazed!",
            "This is incredible and unexpected!",
            "I'm totally blown away by this!",
            "This is so surprising and shocking!",
            "I'm absolutely stunned by this news!"
        ],
        'disgust': [
            "This is disgusting, I can't stand it.",
            "This is revolting, I feel sick.",
            "This is absolutely repulsive!",
            "I'm so disgusted by this!",
            "This makes me feel nauseous.",
            "This is so gross and disgusting!",
            "I'm absolutely appalled by this!",
            "This is so vile and repugnant!",
            "I'm completely disgusted!",
            "This is so offensive and disgusting!"
        ],
        'neutral': [
            "I'm feeling neutral about this situation.",
            "I'm indifferent to this matter.",
            "This doesn't really affect me either way.",
            "I'm feeling quite neutral about it.",
            "This is neither good nor bad to me.",
            "I'm feeling balanced and neutral.",
            "This doesn't really matter to me.",
            "I'm feeling so-so about this.",
            "This is just okay, nothing special.",
            "I'm feeling neither positive nor negative."
        ]
    }
    
    # Create balanced dataset
    texts = []
    emotions = []
    
    for emotion, emotion_text_list in emotion_texts.items():
        for text in emotion_text_list:
            texts.append(text)
            emotions.append(emotion)
    
    return pd.DataFrame({
        'text': texts,
        'emotion': emotions
    })

if __name__ == "__main__":
    from data_preprocessing import TextPreprocessor
    from text_vectorization import TextVectorizer
    from sklearn.model_selection import GridSearchCV
    import pandas as pd

    # Load Sentiment140 cleaned dataset
    DATA_PATH = 'sentiment140_cleaned.csv'
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    # For faster testing, use a sample (set to None to use all data)
    SAMPLE_SIZE = 100000  # Set to None to use the full dataset
    if SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)
    print(f"Loaded {len(df)} rows.")

    # Preprocess data
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_dataset(df, 'text', 'sentiment')

    # Vectorize data with n-grams (unigrams + bigrams)
    vectorizer = TextVectorizer('tfidf')
    X = vectorizer.fit_transform(
        processed_df['processed_text'],
        ngram_range=(1, 2),  # Enable unigrams and bigrams
        max_features=5000,
        min_df=1,
        max_df=1.0,
        stop_words=None  # Already preprocessed
    )
    y = processed_df['sentiment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define models and their hyperparameter grids
    model_grids = {
        'logistic_regression': {
            'model': SentimentClassifier('logistic_regression'),
            'params': {
                'model__C': [0.01, 0.1, 1, 10],
                'model__penalty': ['l2'],
                'model__solver': ['lbfgs']
            }
        },
        'random_forest': {
            'model': SentimentClassifier('random_forest'),
            'params': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5]
            }
        },
        'naive_bayes': {
            'model': SentimentClassifier('naive_bayes'),
            'params': {
                'model__alpha': [0.5, 1.0, 1.5]
            }
        }
    }

    from sklearn.pipeline import Pipeline
    evaluator = ModelEvaluator()
    best_score = 0
    best_model_name = None
    best_model = None
    best_params = None

    for model_name, grid in model_grids.items():
        print(f"\nTuning {model_name}...")
        # Always create a fresh model for the pipeline
        grid['model'].create_model()
        pipe = Pipeline([
            ('model', grid['model'].model)
        ])
        gs = GridSearchCV(pipe, grid['params'], cv=3, scoring='accuracy', n_jobs=-1)
        # Fit label encoder for y
        grid['model'].label_encoder.fit(y_train)
        y_train_enc = grid['model'].label_encoder.transform(y_train)
        gs.fit(X_train, y_train_enc)
        score = gs.best_score_
        print(f"Best CV accuracy for {model_name}: {score:.4f}")
        print(f"Best params: {gs.best_params_}")
        # Evaluate on test set
        y_test_enc = grid['model'].label_encoder.transform(y_test)
        test_acc = gs.score(X_test, y_test_enc)
        print(f"Test accuracy: {test_acc:.4f}")
        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_model = gs.best_estimator_
            best_params = gs.best_params_

    print(f"\nBest model: {best_model_name}")
    print(f"Best cross-validation accuracy: {best_score:.4f}")
    print(f"Best parameters: {best_params}")

    # Save the best model and vectorizer
    import joblib
    # Save the SentimentClassifier object with label_encoder
    best_classifier = SentimentClassifier(best_model_name)
    best_classifier.model = best_model.named_steps['model'] if hasattr(best_model, 'named_steps') else best_model
    # Fit the label encoder on all data
    best_classifier.label_encoder.fit(y)
    best_classifier.is_fitted = True
    best_classifier.save_model('best_sentiment_model.pkl')
    print('Best SentimentClassifier object saved to best_sentiment_model.pkl')
    vectorizer.save_vectorizer('sentiment_vectorizer.pkl')
    print('Vectorizer saved to sentiment_vectorizer.pkl') 