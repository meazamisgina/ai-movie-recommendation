from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.vector_db import VectorDB
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize components
try:
    # Load the vector database
    vector_db = VectorDB.load('data/vector_db/movie_db')
    logger.info("Vector database loaded successfully!")
except Exception as e:
    logger.error(f"Error loading vector database: {e}")
    # Create a new vector database if loading fails
    vector_db = VectorDB()
    logger.error("Created a new vector database instance (this may cause errors)")

# Initialize the LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    logger.info("LLM initialized successfully!")
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")
    llm = None

# Chain-of-Thought prompt template
prompt_template = """
You are a movie recommendation expert. Based on the user's description:
"{query}"

Think step-by-step:
1. Identify key themes, genres, and elements in the description
2. Consider the retrieved movie options and their relevance
3. Select the most fitting movies and explain why each matches the request

Retrieved Movies:
{retrieved_movies}

Provide your top 3 recommendations with explanations:
"""

def get_recommendations(query):
    try:
        # Check if vectorizer is loaded
        if vector_db.vectorizer is None:
            logger.error("Vector database not properly initialized")
            return "Error: Vector database not properly initialized. Please try again later."
        
        # Check if LLM is initialized
        if llm is None:
            logger.error("LLM not properly initialized")
            return "Error: Language model not properly initialized. Please try again later."
        
        # Retrieve relevant movies
        results = vector_db.query(query, n_results=5)
        retrieved_movies = []
        
        for metadata in results["metadatas"]:
            retrieved_movies.append(
                f"{metadata['title']} ({metadata['genre']}): {metadata['plot']}"
            )
        
        # Generate prompt
        prompt = PromptTemplate.from_template(prompt_template).format(query=query, retrieved_movies="\n".join(retrieved_movies))
        
        # Get LLM response
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return f"Error generating recommendations: {str(e)}"