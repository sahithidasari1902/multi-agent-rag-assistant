import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from retriever import retrieve_context

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


class AgentState(TypedDict):
    question: str
    route: str
    context: str
    answer: str


def router_agent(state: AgentState) -> AgentState:
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
    return "retriever" if state["route"] == "retrieve" else "answer"


def retriever_agent(state: AgentState) -> AgentState:
    context = retrieve_context(state["question"])
    return {**state, "context": context}


def answer_agent(state: AgentState) -> AgentState:
    if state.get("context"):
        system = (
            "You are a helpful assistant. Answer the question using ONLY the "
            "context below. If the answer isn't in the context, say so honestly.\n\n"
            f"Context:\n{state['context']}"
        )
    else:
        system = "You are a helpful, direct assistant."
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=state["question"])])
    return {**state, "answer": response.content}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("answer_generator", answer_agent)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {"retriever": "retriever", "answer": "answer_generator"})
    graph.add_edge("retriever", "answer_generator")
    graph.add_edge("answer_generator", END)
    return graph.compile()


agent_graph = build_graph()


def run_agent(question: str) -> dict:
    result = agent_graph.invoke({"question": question, "route": "", "context": "", "answer": ""})
    return {
        "question": question,
        "route_taken": result["route"],
        "used_retrieval": bool(result.get("context")),
        "answer": result["answer"],
    }
