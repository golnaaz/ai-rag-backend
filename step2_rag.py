import os
from google import genai
from google.genai import types

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
client = genai.Client(api_key=API_KEY)

print("1. Reading the text document...")

with open("knowledge.txt", "r", encoding="utf-8") as f:
    document_content = f.read()

#  Design the RAG prompt (combining context + user question)
user_question = "What is Golnaz's background and what is her future plan?"

prompt = f"""
You are a smart assistant. Your task is to answer the user's question **strictly and only** based on the provided reference information below. 
If the answer cannot be found in the text, say "I could not find this information in the text" and do not make anything up.

--- Reference Information ---
{document_content}
-----------------------------

User Question: {user_question}
"""

print("2. Sending the context and question to Gemini...")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.1,  # Low temperature ensures the model sticks strictly to the text without hallucinating
    )
)

print("\n--- AI Response Based on Document ---")
print(response.text)