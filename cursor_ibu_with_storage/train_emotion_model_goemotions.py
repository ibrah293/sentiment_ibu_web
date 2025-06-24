import os
import pandas as pd
from datasets import load_dataset
from data_preprocessing import TextPreprocessor
from text_vectorization import TextVectorizer
from sentiment_models import EmotionClassifier

# 1. Download and prepare GoEmotions dataset
print('Loading GoEmotions dataset...')
dataset = load_dataset('go_emotions')
df = pd.DataFrame(dataset['train'])

# Use the first label for each sample (single-label classification)
df['emotion'] = df['labels'].apply(lambda x: x[0] if len(x) > 0 else -1)
label_map = {i: l for i, l in enumerate(dataset['train'].features['labels'].feature.names)}
df['emotion'] = df['emotion'].map(label_map)
df = df[df['emotion'].notnull()]

# Save cleaned CSV for reference
df[['text', 'emotion']].to_csv('goemotions_cleaned.csv', index=False)
print('GoEmotions cleaned data saved to goemotions_cleaned.csv')

# 2. Preprocess text
print('Preprocessing text...')
preprocessor = TextPreprocessor()
df_processed = preprocessor.preprocess_dataset(df, 'text', 'emotion')

# 3. Vectorize text
print('Vectorizing text...')
vectorizer = TextVectorizer('tfidf')
X = vectorizer.fit_transform(
    df_processed['processed_text'],
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2,
    max_df=0.95,
    stop_words=None
)
y = df_processed['emotion']

# 4. Train emotion classifier
print('Training emotion classifier...')
model = EmotionClassifier('logistic_regression')
model.fit(X, y)

# 5. Save model and vectorizer
print('Saving model and vectorizer...')
model.save_model('best_emotion_model.pkl')
vectorizer.save_vectorizer('emotion_vectorizer.pkl')
print('Emotion model saved to best_emotion_model.pkl')
print('Vectorizer saved to emotion_vectorizer.pkl') 