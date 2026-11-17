import os
from google import genai
from google.genai import types
from langchain_text_splitters import CharacterTextSplitter

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
client = genai.Client(api_key=API_KEY)

print("1. Reading the large document...")
with open("knowledge.txt", "r", encoding="utf-8") as f:
    document_content = f.read()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,  # Max characters per chunk
    chunk_overlap=20   # Overlap to keep context between chunks
)
chunks = text_splitter.split_text(document_content)
print(f"2. Document successfully split into {len(chunks)} chunks using LangChain!")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)

# 4. For now, let's combine the chunks back or select the most relevant one for our prompt
# (In the next steps, we will use Vector Databases to automatically find the best chunk)
combined_chunks = "\n".join(chunks)

user_question = "What is Golnaz's goal?"

prompt = f"""
You are a smart assistant. Answer the user's question strictly based on the reference chunks below.

--- Reference Chunks ---
{combined_chunks}
------------------------

User Question: {user_question}
"""

print("\n3. Sending chunks and question to Gemini...")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.1,
    )
)

print("\n--- AI Response Using LangChain Chunks ---")
print(response.text)

