import os
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
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
