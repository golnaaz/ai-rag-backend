# AI RAG Backend

A Python FastAPI backend that answers questions about uploaded PDF documents using Retrieval-Augmented Generation (RAG) powered by Google Gemini and Qdrant vector search.

Used by [ai-rag-project](https://github.com/golnaaz/ai-rag-project) — a Kotlin Multiplatform client for Android, iOS, Desktop, and Web.

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

## RAG Quality & Faithfulness

**Faithfulness** measures whether every claim in the generated answer is grounded in the retrieved context — not invented by the model. It is the most critical quality signal for a RAG system because it distinguishes a trustworthy answer from a hallucination.

The system is designed with faithfulness as the primary goal:

- The system prompt explicitly instructs Gemini to answer **only from the provided context** and to state "the document does not contain enough information" rather than speculate.
- Retrieved chunks include page-number metadata so answers can be traced back to their source pages (returned in the `done` SSE event as `sources`).
- Retrieval is scoped per `doc_id` so chunks from unrelated documents never contaminate the context window.

### Planned Evaluation (Work in Progress)

A formal evaluation pipeline using [RAGAS](https://docs.ragas.io/) is planned. RAGAS measures:

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Fraction of answer claims that are entailed by the retrieved context (target: > 0.9) |
| **Answer Relevancy** | How directly the answer addresses the question |
| **Context Precision** | Whether the top-ranked chunks are actually relevant |
| **Context Recall** | Whether all information needed to answer is present in the retrieved chunks |

---

## Validation (Work in Progress)

Current validation is minimal — only file extension and empty-file checks are enforced. The following improvements are planned:

- **File size limit** — reject uploads above a configurable threshold (e.g. 20 MB) before reading into memory
- **MIME-type verification** — inspect magic bytes, not just the `.pdf` extension
- **Chunk quality filter** — skip chunks that are too short (e.g. < 50 characters) or consist only of whitespace/numbers, which add noise to the vector store
- **Question sanitisation** — strip or truncate excessively long queries before embedding to avoid token-limit errors
- **Per-document isolation check** — return a clear error when a `doc_id` does not exist in Qdrant instead of silently falling back to the full collection

---

## Security

The `GOOGLE_API_KEY` is loaded from a `.env` file (never hardcoded). The `.env` file is git-ignored.