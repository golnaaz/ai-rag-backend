import json
import os
import tempfile
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types

from rag_service import build_rag_prompt, ingest_pdf_file, init_startup_document
from schemas import QueryRequest, UploadResponse

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=GOOGLE_API_KEY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Qdrant and optional default PDF on startup
    init_startup_document()
    yield


app = FastAPI(title="AI Document Intelligence Backend", version="2.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    # Validate file extension before reading into temporary storage
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path: str | None = None
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            temp_path = tmp_file.name

        result = ingest_pdf_file(file_path=temp_path, filename=file.filename)
        return UploadResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(exc)}")
    finally:
        # Clean up temporary file regardless of ingestion outcome
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/ask-stream")
async def ask_ai_stream(request: QueryRequest):
    try:
        system_instruction, user_prompt, sources = build_rag_prompt(
            question=request.question,
            doc_id=request.doc_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    def event_generator():
        try:
            # Stream generated tokens over Server-Sent Events (SSE)
            stream = client.models.generate_content_stream(
                model=CHAT_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                ),
            )

            for chunk in stream:
                if chunk.text:
                    payload = json.dumps({"type": "token", "content": chunk.text}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

            # Emit citation page numbers after streaming completes
            done_payload = json.dumps({"type": "done", "sources": sources}, ensure_ascii=False)
            yield f"data: {done_payload}\n\n"

        except Exception as exc:
            error_payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/")
def health_check():
    return {"status": "Backend is running smoothly!"}