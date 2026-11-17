# AI RAG Backend

A Python FastAPI backend that answers questions about PDF documents using Retrieval-Augmented Generation (RAG) powered by Google Gemini.

## How it works

On startup the server loads a PDF, splits it into chunks, converts them into embeddings using `gemini-embedding-2-preview`, and stores them in a Chroma vector database. When a question arrives it retrieves the most relevant chunks and sends them as context to Gemini to generate an answer.

```
POST /ask
Body:     { "question": "your question here" }
Response: { "status": "success", "answer": "...", "sources": [1, 2, 3] }
```

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) — REST API server
- [Google Gemini](https://ai.google.dev/) — LLM for generation + embeddings
- [LangChain](https://www.langchain.com/) — document loading and text splitting
- [Chroma](https://www.trychroma.com/) — local vector database
- Python 3.11+

## Setup

```bash
git clone https://github.com/golnaaz/ai-rag-backend.git
cd ai-rag-backend

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install fastapi uvicorn google-genai langchain langchain-community \
            langchain-google-genai langchain-chroma chromadb pypdf python-dotenv

cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

cp /path/to/your/file.pdf sample.pdf

uvicorn main:app --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`.

## Android Client

This backend is consumed by [AiRagApp](https://github.com/golnaaz/AiRagApp) — a Kotlin Multiplatform app built with Compose Multiplatform.