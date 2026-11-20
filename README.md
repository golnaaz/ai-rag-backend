# AI RAG Backend

A Python FastAPI backend that answers questions about uploaded PDF documents using Retrieval-Augmented Generation (RAG) powered by Google Gemini and Qdrant vector search.

Used by [AiRagApp](https://github.com/golnaaz/AiRagApp) — a Kotlin Multiplatform client for Android, iOS, Desktop, and Web.

---

## How It Works

1. **Upload** — a PDF is split into chunks, embedded with `gemini-embedding-2-preview`, and stored in a local Qdrant collection
2. **Query** — the question is embedded, the top-k most relevant chunks are retrieved
3. **Answer** — chunks are injected as context into a Gemini prompt; the answer streams back token-by-token over SSE

---

## API

### `POST /upload`

Upload a PDF document.

```
Content-Type: multipart/form-data
Field: file  (PDF binary)
```

Response:
```json
{
  "doc_id": "abc123",
  "filename": "document.pdf",
  "pages_count": 4,
  "chunks_count": 18,
  "status": "indexed"
}
```

### `POST /ask-stream`

Ask a question about an uploaded document. Streams the answer as Server-Sent Events.

```json
{ "question": "What is the main topic?", "doc_id": "abc123" }
```

SSE event types:
```
data: {"type": "token",  "content": "The main ..."}
data: {"type": "done",   "content": null}
data: {"type": "error",  "message": "..."}
```

API docs available at `http://localhost:8000/docs` when running locally.

---

## Setup

```bash
git clone https://github.com/golnaaz/ai-rag-backend.git
cd ai-rag-backend

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set:  GOOGLE_API_KEY=your_key_here
```

Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) — REST API + SSE streaming
- [Google Gemini](https://ai.google.dev/) — LLM generation + text embeddings
- [Qdrant](https://qdrant.tech/) — local vector database (in-memory)
- [LangChain](https://www.langchain.com/) — PDF loading and text splitting
- Python 3.11+

---

## Security

The `GOOGLE_API_KEY` is loaded from a `.env` file (never hardcoded). The `.env` file is git-ignored.