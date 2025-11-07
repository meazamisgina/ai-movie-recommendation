import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        self.embeddings = None
        self.vectorizer = None
        self.metadata = []
    
    def add_embeddings(self, embeddings_path, vectorizer_path, data_path):
        try:
            # Load embeddings and vectorizer
            with open(embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            df = pd.read_csv(data_path)
            
            # Store metadata
            for i, row in df.iterrows():
                self.metadata.append({
                    "title": row['Series_Title'],
                    "genre": row['Genre'],
                    "plot": row['Overview'],
                    "cast": [row['Star1'], row['Star2'], row['Star3'], row['Star4']]
                })
            
            logger.info("Embeddings and metadata added successfully")
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            raise
    
    def query(self, query_text, n_results=5):
        try:
            # Transform the query text using the loaded vectorizer
            query_embedding = self.vectorizer.transform([query_text]).toarray()
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top n results
            top_indices = np.argsort(similarities)[::-1][:n_results]
            
            # Format results
            results = {
                "metadatas": [self.metadata[i] for i in top_indices],
                "distances": [1 - similarities[i] for i in top_indices]
            }
            
            return results
        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            raise
    
    def save(self, path):
        try:
            # Save embeddings, vectorizer, and metadata
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(f"{path}_embeddings.pkl", "wb") as f:
                pickle.dump(self.embeddings, f)
            with open(f"{path}_vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
            with open(f"{path}_metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Vector database saved to {path}")
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
            raise
    
    @classmethod
    def load(cls, path):
        try:
            db = cls()
            with open(f"{path}_embeddings.pkl", "rb") as f:
                db.embeddings = pickle.load(f)
            with open(f"{path}_vectorizer.pkl", "rb") as f:
                db.vectorizer = pickle.load(f)
            with open(f"{path}_metadata.pkl", "rb") as f:
                db.metadata = pickle.load(f)
            logger.info(f"Vector database loaded from {path}")
            return db
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            raise