"""
Ingestion pipeline: reads PDFs/text files from DOCS_DIR, splits them into
chunks, embeds them, and stores them in a persistent Chroma vector database.

Run this once (or whenever you add new documents):
    python ingest.py
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", ".")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")


def load_documents(docs_dir: str):
    """Load every PDF and .txt file in docs_dir into LangChain Document objects."""
    documents = []
    for filename in os.listdir(docs_dir):
        path = os.path.join(docs_dir, filename)
        if filename.lower().endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())
        elif filename.lower().endswith(".txt"):
            documents.extend(TextLoader(path).load())
    return documents


def build_vector_store():
    os.makedirs(DOCS_DIR, exist_ok=True)
    documents = load_documents(DOCS_DIR)

    if not documents:
        print(f"No documents found in {DOCS_DIR}. Add a PDF or .txt file and re-run.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} document(s), split into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()
    print(f"Vector store built and persisted at {PERSIST_DIR}")


if __name__ == "__main__":
    build_vector_store()
