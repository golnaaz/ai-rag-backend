import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

# Load environment configuration for RAG pipeline
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
PDF_PATH = "sample.pdf"
QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "documents_collection"

def init_vector_store() -> QdrantVectorStore | None:
    # Skip initialization if no sample PDF is present yet
    if not os.path.exists(PDF_PATH):
        print(f"Warning: '{PDF_PATH}' not found. Place the file in the project directory.")
        return None

    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    # Probe embedding dimension to create Qdrant collection cleanly
    sample_vector = embeddings.embed_query("dimension probe")
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=len(sample_vector), distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(docs)
    print("Qdrant vector store initialized successfully.")
    return vector_store

vector_store = init_vector_store()

def build_rag_prompt(question: str) -> tuple[str, str, list[int]]:
    if not vector_store:
        raise RuntimeError("Vector database is not initialized.")

    # Retrieve top-k chunks with source page metadata
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)

    retrieved_context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}"
        for doc in relevant_docs
    )
    sources = sorted({doc.metadata.get("page", 0) + 1 for doc in relevant_docs})

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