"""
Loads the persisted Chroma vector store and exposes a simple retrieval
function used by the retriever agent node in the LangGraph.

Uses a local, custom FastEmbed wrapper (see embeddings.py) — free, no API
key, no cost, and avoids a known bug in langchain_community's own
FastEmbedEmbeddings wrapper.
"""
import os
from dotenv import load_dotenv
from embeddings import LocalFastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

_embeddings = LocalFastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
_vectordb = None


def get_vector_store() -> Chroma:
    global _vectordb
    if _vectordb is None:
        _vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)
    return _vectordb


def retrieve_context(query: str, k: int = 4) -> str:
    vectordb = get_vector_store()
    results = vectordb.similarity_search(query, k=k)
    if not results:
        return ""
    return "\n\n---\n\n".join(doc.page_content for doc in results)
