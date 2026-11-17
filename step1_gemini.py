import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY", "")
client = genai.Client(api_key=API_KEY)

print("Sending message to gemini")

response = client.models.generate_content(
   model="gemini-3.8-flash",
    contents="Hello! I'm a mobile developer and I've just written my first AI code in Python. How can I use my mobile development skills in the AI world? Keep it short and motivational.",
    config=types.GenerateContentConfig(
        # Define the AI's persona and behavior (System Instruction)
        system_instruction="You are a senior AI coach who responds in a friendly, professional, and English language.",
        temperature=0.7, # Controls model creativity (ranges from 0 to 2)
    )
)

print("\n Gemini resoinse:")
print(response.text)