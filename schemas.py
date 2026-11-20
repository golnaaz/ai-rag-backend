from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question about the document")

class StreamEvent(BaseModel):
    type: str
    content: str | None = None
    sources: list[int] | None = None
    message: str | None = None