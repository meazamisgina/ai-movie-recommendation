# AI-Powered Movie Recommendation System

## Overview
This project is a semantic movie recommendation system that allows users to describe the type of movie they want and receive personalized recommendations that closely match their descriptions. The system uses natural language processing and generative AI to understand user queries and provide detailed, contextual recommendations.

## Features
- Natural language query processing
- Semantic search using TF-IDF vectorization
- Custom vector database for efficient similarity search
- Retrieval-Augmented Generation (RAG) using Google's Gemini LLM
- RESTful API for easy integration
- Detailed explanations for recommendations

## Architecture
1. **Data Processing**: Movie data is processed and cleaned using NLTK for lemmatization
2. **Embedding Generation**: TF-IDF vectorization is used to create embeddings for movie descriptions
3. **Vector Database**: Custom implementation using scikit-learn for efficient similarity search
4. **RAG Pipeline**: Retrieves relevant movies and uses Google Gemini to generate detailed recommendations
5. **API Layer**: FastAPI provides a RESTful interface for the system

## Installation
1. Clone the repository:
```bash
git clone https://github.com/meazamisgina/ai-movie-recommendation
cd ai-movie-recommendation
