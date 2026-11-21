import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
DEFAULT_PDF_PATH = "sample.pdf"
QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "documents_collection"

_embeddings: GoogleGenerativeAIEmbeddings | None = None
_vector_store: QdrantVectorStore | None = None


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _embeddings


def get_vector_store() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    embeddings = _get_embeddings()
    sample_vector = embeddings.embed_query("dimension probe")
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=len(sample_vector), distance=Distance.COSINE),
        )

    _vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return _vector_store


def ingest_pdf_file(file_path: str, filename: str, doc_id: str | None = None) -> dict:
    # Generate a unique document ID when not explicitly provided
    resolved_doc_id = doc_id or str(uuid.uuid4())

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(pages)

    # Attach document metadata to every chunk for multi-document filtering and citations
    for chunk in chunks:
        page_idx = chunk.metadata.get("page", 0)
        chunk.metadata["doc_id"] = resolved_doc_id
        chunk.metadata["filename"] = filename
        chunk.metadata["page_number"] = page_idx + 1

    store = get_vector_store()
    store.add_documents(chunks)

    return {
        "doc_id": resolved_doc_id,
        "filename": filename,
        "pages_count": len(pages),
        "chunks_count": len(chunks),
        "status": "indexed",
    }


def init_startup_document() -> None:
    # Ensure Qdrant collection is ready even before any upload happens
    get_vector_store()
    if os.path.exists(DEFAULT_PDF_PATH):
        ingest_pdf_file(DEFAULT_PDF_PATH, filename=DEFAULT_PDF_PATH, doc_id="default-sample")
        print("Default sample document indexed successfully.")


def build_rag_prompt(question: str, doc_id: str | None = None) -> tuple[str, str, list[int]]:
    store = get_vector_store()

    # Apply metadata filter when querying a specific uploaded PDF
    search_kwargs: dict = {"k": 4}
    if doc_id:
        search_kwargs["filter"] = Filter(
            must=[FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_id))]
        )

    retriever = store.as_retriever(search_kwargs=search_kwargs)
    relevant_docs = retriever.invoke(question)

    retrieved_context = "\n\n".join(
        f"[Page {doc.metadata.get('page_number', doc.metadata.get('page', 0) + 1)}]: {doc.page_content}"
        for doc in relevant_docs
    )
    sources = sorted(
        {doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1) for doc in relevant_docs}
    )

    system_instruction = (
        "You are a professional document intelligence assistant. "
        "Answer strictly based on the provided reference context. "
        "If the answer cannot be found in the context, state clearly that the document does not contain enough information."
    )
    user_prompt = (
        f"--- Reference Context ---\n"
        f"{retrieved_context}\n"
        f"-------------------------\n\n"
        f"User Question: {question}"
    )
    return system_instruction, user_prompt, sources