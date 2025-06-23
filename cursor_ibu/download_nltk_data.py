#!/usr/bin/env python3
"""
Script to download all required NLTK data for the sentiment analysis project
"""

import nltk

def download_nltk_data():
    """Download all required NLTK data"""
    print("Downloading NLTK data...")
    
    # List of required NLTK data
    required_data = [
        'punkt',
        'stopwords', 
        'wordnet',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]
    
    for data in required_data:
        try:
            print(f"Downloading {data}...")
            nltk.download(data, quiet=True)
            print(f"✅ {data} downloaded successfully")
        except Exception as e:
            print(f"❌ Error downloading {data}: {e}")
    
    print("\n🎉 NLTK data download completed!")
    print("You can now run the sentiment analysis application.")

if __name__ == "__main__":
    download_nltk_data() 