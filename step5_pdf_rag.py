import os
import time
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


# 1. Initialize Gemini Client for final generation
API_KEY = os.environ.get("GOOGLE_API_KEY", "")
client = genai.Client(api_key=API_KEY)

pdf_path = "sample.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Please place a '{pdf_path}' file in the project folder first!")

print(f"1. Loading PDF document: {pdf_path}...")

# 2. Load the PDF page by page (Industry Standard Loader)
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"   Successfully loaded {len(pages)} pages from the PDF.")

# 3. Smart Chunking across PDF pages
# Note: Chunk size is larger for real PDFs (1000 chars) to keep full paragraphs intact
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

docs = text_splitter.split_documents(pages)
print(f"2. Split PDF into {len(docs)} smart chunks.")

print("3. Creating Embeddings and storing in Chroma Vector DB...")

# 4. Initialize Embeddings with the latest supported Gemini model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=API_KEY
)
# 5. Store chunks in Chroma Vector Database
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

# 6. User Question 
user_question = "Summarize the main points or skills mentioned in this document."

print(f"4. Searching Vector DB for top 3 relevant chunks...")

# 7. Retrieve top 3 most relevant chunks and join them together
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke(user_question)

# Combine the retrieved chunks AND include page numbers from metadata!
retrieved_context = "\n\n".join(
    [f"[Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}" for doc in relevant_docs]
)

print("\n--- Retrieved Context Snippets ---")
print(retrieved_context[:400] + "...\n")

# 8. Construct the RAG Prompt
prompt = f"""
You are a professional document assistant. Answer the user's question strictly based on the reference context provided below. 
If possible, mention which page number the information came from.

--- Reference Context ---
{retrieved_context}
-------------------------

User Question: {user_question}
"""

print("5. Sending context to Gemini with Retry protection...")

# 9. Generate response with Retry logic for 503 High Demand errors
max_retries = 3
response = None

for attempt in range(max_retries):
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        break
    except Exception as e:
        print(f"   Attempt {attempt+1} failed due to server load. Retrying in 5 seconds... ({e})")
        time.sleep(5)

        # 10. Output Result
if response:
    print("\n--- Final AI Response ---")
    print(response.text)
else:
    print("\nFailed to generate response after multiple attempts.")