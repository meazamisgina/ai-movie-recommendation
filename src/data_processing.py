import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import string
import re

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Basic text cleaning"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text):
    """Apply NLTK lemmatization"""
    # Tokenize the text
    tokens = word_tokenize(text)
    # Lemmatize each token
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmas)

# Load data
df = pd.read_csv('data/raw/imdb_top_1000.csv')

def create_metatext(row):
    """Combine title, genre, plot, and cast into single text field"""
    text = f"{row['Series_Title']} {row['Genre']} {row['Overview']} "
    text += f"{row['Star1']} {row['Star2']} {row['Star3']} {row['Star4']}"
    return text

# Create MetaText field
df['MetaText'] = df.apply(create_metatext, axis=1)

# Clean text
df['MetaText'] = df['MetaText'].apply(clean_text)

# Apply lemmatization with progress bar
tqdm.pandas(desc="Lemmatizing")
df['MetaText'] = df['MetaText'].progress_apply(lemmatize_text)

# Save processed data
df.to_csv('data/processed/imdb_processed.csv', index=False)
print("Data processing completed!")