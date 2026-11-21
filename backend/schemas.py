from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question about the document")
    doc_id: str | None = None

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages_count: int
    chunks_count: int
    status: str
    
class StreamEvent(BaseModel):
    type: str
    content: str | None = None
    sources: list[int] | None = None
    message: str | None = None