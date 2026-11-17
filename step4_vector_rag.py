import os
import time
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings



# 1. Initialize Gemini Client for final generation
API_KEY = os.environ.get("GOOGLE_API_KEY", "")
client = genai.Client(api_key=API_KEY)

print("1. Reading the document...")

# 2. Read the knowledge file
with open("knowledge.txt", "r", encoding="utf-8") as f:
    document_content = f.read()

# 3. Smart Chunking (Industry Standard: Recursive Splitter)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,     # Max characters per chunk
    chunk_overlap=30,   # Overlap to preserve context between chunks
    separators=["\n\n", "\n", " ", ""]
)
docs = text_splitter.create_documents([document_content])
print(f"2. Document successfully split into {len(docs)} smart chunks.")

print("3. Converting chunks to Embeddings and storing in Chroma Vector DB...")

# 4. Initialize Google Embeddings model
# This transforms text into mathematical vectors that capture semantic meaning
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=API_KEY
)

# 5. Create a local Vector Database using Chroma
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)
# 6. User Question
user_question = "What is Golnaz's background?"

print(f"4. Searching Vector DB for the most relevant chunk to: '{user_question}'")

# 7. Semantic Search (Retrieval): Find the most relevant chunk automatically
retriever = vector_store.as_retriever(
    search_kwargs={"k": 1}  # Retrieve only the top 1 most relevant chunk
)
relevant_docs = retriever.invoke(user_question)

# Extract the text from the retrieved chunk
retrieved_context = relevant_docs[0].page_content
print(f"\n[Retrieved Context Found]: {retrieved_context}")

# 8. Construct the final RAG Prompt using the retrieved context
prompt = f"""
You are a smart assistant. Answer the user's question strictly based on the reference context below.

--- Reference Context ---
{retrieved_context}
-------------------------

User Question: {user_question}
"""
print("5. Sending retrieved context and question to Gemini...")

# 9. Generate final response from Gemini with Retry mechanism (to handle 503 high demand errors)
max_retries = 3
response = None

for attempt in range(max_retries):
  try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",  
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
        ),
    )
    break 
  except Exception as e:
    print(
        f"Attempt {attempt+1} failed due to server load. Retrying in 5"
        f" seconds... Error: {e}"
    )
    time.sleep(5)

# 10. Print the final response if successful
if response:
  print("\n--- Final AI Response (Powered by Vector Search & RAG) ---")
  print(response.text)
else:
  print(
      "\nFailed to get a response after multiple attempts due to high server"
      " demand. Please try again later."
  )