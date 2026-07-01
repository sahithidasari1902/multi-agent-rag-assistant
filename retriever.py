"""
Loads the persisted Chroma vector store and exposes a simple retrieval
function used by the retriever agent node in the LangGraph.
"""
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vectordb = None


def get_vector_store() -> Chroma:
    """Lazily load the vector store so the app can start even before ingestion."""
    global _vectordb
    if _vectordb is None:
        _vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=_embeddings,
        )
    return _vectordb


def retrieve_context(query: str, k: int = 4) -> str:
    """Return the top-k most relevant chunks for a query, joined as one string."""
    vectordb = get_vector_store()
    results = vectordb.similarity_search(query, k=k)
    if not results:
        return ""
    return "\n\n---\n\n".join(doc.page_content for doc in results)
