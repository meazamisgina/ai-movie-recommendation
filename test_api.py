import os
import google.generativeai as genai

# Set your API key directly 
api_key = "AIzaSyDBGxZsyrWvMN0tHtJXBSZItl9bED-SEEk"
os.environ["GOOGLE_API_KEY"] = api_key

# Configure the generative AI module
genai.configure(api_key=api_key)

# Test the API with the correct model name
try:
    # Use the gemini-2.5-flash model
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Hello, please confirm this API key is working.")
    print("API key is working!")
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
