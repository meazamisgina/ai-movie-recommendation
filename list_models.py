
import os
import google.generativeai as genai

# Set your API key directly (replace with your actual key)
api_key = "AIzaSyDBGxZsyrWvMN0tHtJXBSZItl9bED-SEEk"
os.environ["GOOGLE_API_KEY"] = api_key

# Configure the generative AI module
genai.configure(api_key=api_key)

# List available models
print("Available models:")
for model in genai.list_models():
    print(f"Name: {model.name}, Supported Methods: {model.supported_generation_methods}")



