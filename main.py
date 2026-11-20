import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types

from rag_service import build_rag_prompt, vector_store
from schemas import QueryRequest

load_dotenv()

app = FastAPI(title="AI Document Intelligence Backend", version="2.0")

# Initialize Gemini client for streaming generation
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
client = genai.Client(api_key=GOOGLE_API_KEY)

@app.post("/ask-stream")
async def ask_ai_stream(request: QueryRequest):
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector database is not initialized.")

    try:
        system_instruction, user_prompt, sources = build_rag_prompt(request.question)
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

            # Send citation page numbers once the stream finishes
            done_payload = json.dumps({"type": "done", "sources": sources}, ensure_ascii=False)
            yield f"data: {done_payload}\n\n"

        except Exception as exc:
            error_payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
def health_check():
    return {"status": "Backend is running smoothly!"}