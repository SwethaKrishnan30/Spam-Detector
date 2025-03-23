"""
Setup script for downloading required NLTK data
Run this script before starting the application to download necessary NLTK resources
"""

import nltk

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')
    
    print("NLTK resources downloaded successfully!")

if __name__ == "__main__":
    download_nltk_data()