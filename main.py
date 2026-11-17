import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="AI Engineering RAG Backend", version="1.0")

# 1. Initialize Gemini Client & Load Vector Store on Startup
API_KEY = os.environ.get("GOOGLE_API_KEY", "")
client = genai.Client(api_key=API_KEY)

pdf_path = "sample.pdf"
vector_store = None

print("Initializing RAG Pipeline and Vector Database...")
if os.path.exists(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(pages)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=API_KEY
    )

    vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)
    print("Vector Database initialized successfully!")
else:
    print(f"Warning: {pdf_path} not found. Please place it in the folder.")   


class QueryRequest(BaseModel):
    question: str

# 2. Define API Endpoint for Chat/RAG
@app.post("/ask")
async def ask_ai(request: QueryRequest):
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector database is not initialized.")
    try:
        # جستجو در Chroma
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(request.question)
        
        retrieved_context = "\n\n".join(
            [f"[Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}" for doc in relevant_docs]
        )
        
        prompt = f"""
        You are a professional document assistant. Answer the user's question strictly based on the reference context provided below.
        
        --- Reference Context ---
        {retrieved_context}
        -------------------------

        User Question: {request.question}
        """
        
        # ارسال به مدل با مدیریت خطا
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        
        return {
            "status": "success",
            "answer": response.text,
            "sources": [doc.metadata.get('page', 0) + 1 for doc in relevant_docs]
        }   
    except Exception as e:
        # مدیریت هوشمند ارورهای سرور برای اینکه اپلیکیشن کاتلین کرش نکند
        raise HTTPException(status_code=503, detail=f"AI model service is currently busy or failed: {str(e)}")    
    
    @app.get("/")
    def health_check():
     return {"status": "Backend is running smoothly!"}    