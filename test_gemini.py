import google.generativeai as genai
import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API KEY:", api_key)

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    "models/gemini-1.5-flash"
)

response = model.generate_content(
    "What is antenna?"
)

print(response.text)