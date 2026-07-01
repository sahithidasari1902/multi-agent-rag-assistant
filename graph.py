"""
Multi-agent orchestration graph.

Three agents, each a node in a LangGraph StateGraph:

  1. Router agent   - decides whether the question needs document retrieval
                       or can be answered directly.
  2. Retriever agent - pulls relevant chunks from the vector database (RAG).
  3. Answer agent    - generates the final response, grounded in retrieved
                       context when available.

This is the "autonomous agent orchestration" piece: the ROUTING DECISION is
made by the LLM itself at runtime, not hardcoded if/else logic — the graph's
path through the nodes changes based on what the router agent decides.
"""
import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from retriever import retrieve_context

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---- Shared state passed between agent nodes ----
class AgentState(TypedDict):
    question: str
    route: str          # "retrieve" or "direct"
    context: str         # filled in by the retriever agent
    answer: str          # filled in by the answer agent


# ---- Agent 1: Router ----
def router_agent(state: AgentState) -> AgentState:
    """Decides whether this question needs document lookup."""
    routing_prompt = (
        "You are a routing agent. Decide if the user's question requires "
        "searching internal documents to answer accurately, or if it can be "
        "answered directly from general knowledge. "
        "Reply with exactly one word: 'retrieve' or 'direct'.\n\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke([HumanMessage(content=routing_prompt)])
    decision = response.content.strip().lower()
    route = "retrieve" if "retrieve" in decision else "direct"
    return {**state, "route": route}


def route_decision(state: AgentState) -> Literal["retriever", "answer"]:
    """Edge function: tells the graph which node to go to next."""
    return "retriever" if state["route"] == "retrieve" else "answer"


# ---- Agent 2: Retriever ----
def retriever_agent(state: AgentState) -> AgentState:
    """Pulls relevant chunks from the vector database."""
    context = retrieve_context(state["question"])
    return {**state, "context": context}


# ---- Agent 3: Answer ----
def answer_agent(state: AgentState) -> AgentState:
    """Generates the final answer, grounded in retrieved context if present."""
    if state.get("context"):
        system = (
            "You are a helpful assistant. Answer the question using ONLY the "
            "context below. If the answer isn't in the context, say so honestly.\n\n"
            f"Context:\n{state['context']}"
        )
    else:
        system = "You are a helpful, direct assistant."

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["question"]),
    ])
    return {**state, "answer": response.content}


# ---- Build the graph ----
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("answer_generator", answer_agent)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "retriever": "retriever",
        "answer": "answer_generator",
    })
    graph.add_edge("retriever", "answer_generator")
    graph.add_edge("answer_generator", END)

    return graph.compile()


agent_graph = build_graph()


def run_agent(question: str) -> dict:
    """Entry point used by the FastAPI route."""
    result = agent_graph.invoke({
        "question": question,
        "route": "",
        "context": "",
        "answer": "",
    })
    return {
        "question": question,
        "route_taken": result["route"],
        "used_retrieval": bool(result.get("context")),
        "answer": result["answer"],
    }
