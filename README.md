# Multi-Agent RAG Assistant

A FastAPI service backed by a **LangGraph** multi-agent orchestration pipeline
with **Retrieval-Augmented Generation (RAG)** over a **Chroma vector database**.

## How it works

```
User question
     |
     v
 Router Agent -- decides "retrieve" or "direct"
     |
     +-- retrieve --> Retriever Agent (searches Chroma DB) --> Answer Agent
     |
     +-- direct  -----------------------------------------> Answer Agent
                                                                   |
                                                                   v
                                                            Final response
```

The **routing decision is made by the LLM at runtime** — the graph's execution
path changes dynamically based on the question, which is what makes this
agent orchestration rather than a fixed pipeline.

## Files

```
main.py                 FastAPI app, /ask and /health endpoints
graph.py                 LangGraph: router -> retriever -> answer agent orchestration
retriever.py             Vector similarity search
ingest.py                Document loading + chunking + embedding into Chroma
sample_leave_policy.txt  Sample document to test retrieval against
Dockerfile
requirements.txt
render.yaml              Render deployment blueprint
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your OPENAI_API_KEY
```

## Ingest documents (build the vector database)

Any `.txt` or `.pdf` file sitting in this folder gets indexed (a sample file
is already included):

```bash
python ingest.py
```

## Run locally

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive Swagger API docs.

Test it:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of annual leave do employees get?"}'
```

## Run with Docker

```bash
docker build -t rag-agent .
docker run -p 8000:8000 --env-file .env rag-agent
```

## Deploy to Render

1. Push this repo to GitHub.
2. On [render.com](https://render.com): **New +** → **Web Service** → connect the repo.
3. Render detects `render.yaml` and pre-fills settings (Docker environment).
4. Paste your `OPENAI_API_KEY` when prompted.
5. Click **Create Web Service**.
6. Once live: `https://<your-service-name>.onrender.com/health`

**Note:** Render's free tier has an ephemeral filesystem, so run `python ingest.py`
once via Render's **Shell** tab after your first deploy to build the vector store
on the live server.

## What this project demonstrates (for resume/interview purposes)

| Skill | Where |
|---|---|
| FastAPI / REST API design | `main.py` |
| LangGraph agent orchestration | `graph.py` |
| Multi-agent systems | Router, Retriever, and Answer agents as separate graph nodes |
| RAG (Retrieval-Augmented Generation) | `retriever.py`, `ingest.py` |
| Vector database (ChromaDB) | `ingest.py`, `retriever.py` |
| LLM integration | `ChatOpenAI` calls in `graph.py` |
| Docker containerization | `Dockerfile` |
| Cloud deployment | Render deployment steps above |

## Suggested resume bullet once this is live

> "Built and deployed a multi-agent RAG assistant using LangGraph, FastAPI, and
> ChromaDB; implemented dynamic LLM-based routing between direct-answer and
> document-retrieval agent paths; containerized with Docker and deployed to Render."

Only add this once you've actually run it, ingested a real document, and
deployed it — the bullet should describe something you can walk through live
in an interview.
