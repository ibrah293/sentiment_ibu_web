import os
import requests
import zipfile
import pandas as pd

# Download URL and filenames
DATA_URL = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
ZIP_FILE = 'trainingandtestdata.zip'
CSV_FILE = 'training.1600000.processed.noemoticon.csv'
OUTPUT_CSV = 'sentiment140_cleaned.csv'

# Download the dataset
if not os.path.exists(ZIP_FILE):
    print('Downloading Sentiment140 dataset...')
    r = requests.get(DATA_URL)
    with open(ZIP_FILE, 'wb') as f:
        f.write(r.content)
else:
    print('Zip file already exists.')

# Extract the CSV
if not os.path.exists(CSV_FILE):
    print('Extracting zip file...')
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall('.')
else:
    print('CSV file already extracted.')

# Load and clean the data
print('Loading and cleaning data...')
df = pd.read_csv(CSV_FILE, encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['sentiment', 'text']

# Map sentiment labels: 0=negative, 2=neutral, 4=positive
label_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
df['sentiment'] = df['sentiment'].map(label_map)

# Drop any rows with missing values
df = df.dropna(subset=['text', 'sentiment'])

# Save cleaned CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f'Cleaned data saved to {OUTPUT_CSV}') 