"""
FastAPI layer exposing the multi-agent RAG system as a REST API.

Run locally:
    uvicorn main:app --reload

Endpoints:
    POST /ask      -> ask the agent a question
    GET  /health   -> health check
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph import run_agent

app = FastAPI(
    title="Multi-Agent RAG Assistant",
    description="A FastAPI service backed by a LangGraph agent orchestration "
                 "pipeline (router -> retriever -> answer) with RAG over a "
                 "Chroma vector database.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    route_taken: str
    used_retrieval: bool
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = run_agent(request.question)
    return result
