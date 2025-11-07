import pytest
from src.rag_pipeline import get_recommendations

def test_recommendation_quality():
    query = "I want a sci-fi movie with time travel and a twist ending"
    result = get_recommendations(query)
    
    # Basic checks
    assert "recommendation" in result.lower()
    assert len(result) > 100  # Ensure detailed response
    assert "sci-fi" in result.lower() or "science fiction" in result.lower()