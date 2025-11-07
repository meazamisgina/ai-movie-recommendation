from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import os

# Load processed data
df = pd.read_csv('data/processed/imdb_processed.csv')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=384)  # Using 384 features to match the original embedding size

# Fit and transform the MetaText column
print("Generating TF-IDF embeddings...")
tfidf_matrix = vectorizer.fit_transform(df['MetaText'])

# Save the vectorizer and the matrix
os.makedirs('data/embeddings', exist_ok=True)
with open('data/embeddings/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save the TF-IDF matrix (we'll convert it to a dense array for simplicity)
tfidf_dense = tfidf_matrix.toarray()
with open('data/embeddings/tfidf_embeddings.pkl', 'wb') as f:
    pickle.dump(tfidf_dense, f)

print("TF-IDF embeddings generated and saved!")