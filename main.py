"""
FastAPI layer exposing the multi-agent RAG system as a REST API,
plus a simple chat UI served at the root URL.

Run locally:
    uvicorn main:app --reload

Endpoints:
    GET  /        -> simple chat UI (index.html)
    POST /ask     -> ask the agent a question
    GET  /health  -> health check
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graph import run_agent
from ingest import build_vector_store

app = FastAPI(
    title="Multi-Agent RAG Assistant",
    description="A FastAPI service backed by a LangGraph agent orchestration "
                 "pipeline (router -> retriever -> answer) with RAG over a "
                 "Chroma vector database. Uses Groq's free API for the LLM.",
    version="1.0.0",
)


@app.on_event("startup")
def ingest_on_startup():
    """
    Automatically builds the vector store when the app boots, if it doesn't
    already exist. This lets free-tier hosting (no Shell access) still get
    a working /ask endpoint without any manual step.
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    if not os.path.isdir(persist_dir) or not os.listdir(persist_dir):
        print("No existing vector store found — running ingestion on startup...")
        build_vector_store()
    else:
        print("Vector store already exists — skipping ingestion.")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    route_taken: str
    used_retrieval: bool
    answer: str


@app.get("/")
def serve_ui():
    """Serves the simple chat UI so the live link isn't just raw API docs."""
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = run_agent(request.question)
    return result
