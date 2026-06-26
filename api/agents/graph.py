"""
LangGraph agentic RAG workflow.
Design doc Section 16 — Self-correcting retrieval agent.

The graph implements: classify → retrieve → evaluate → decide → generate
with up to 2 requery loops for quality improvement.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from api.agents.agents import (
    evaluate_retrieval_quality,
    generate_answer,
    refine_query,
    search_documents,
)
from api.pipeline.router import QueryClass, Strategy, classify_query

logger = logging.getLogger(__name__)

MAX_REQUERY = 2


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Shared state passed between graph nodes."""
    query: str
    original_query: str
    query_class: str
    strategy: str
    chunks: list[dict]
    answer: str
    quality_score: float
    requery_count: int
    sources: list[dict]
    metadata: dict[str, Any]


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def classify_node(state: AgentState) -> AgentState:
    """Classify the query and determine initial retrieval strategy."""
    query = state["query"]
    qclass = classify_query(query)

    strategy_map = {
        QueryClass.KEYWORD: Strategy.SPARSE,
        QueryClass.CONCEPTUAL: Strategy.DENSE,
        QueryClass.MIXED: Strategy.HYBRID,
    }

    state["query_class"] = qclass.value
    state["strategy"] = strategy_map[qclass].value
    state["original_query"] = query

    logger.info(f"Classified query as {qclass.value} → strategy={strategy_map[qclass].value}")
    return state


async def retrieve_node(state: AgentState) -> AgentState:
    """Execute retrieval using the selected strategy."""
    chunks = await search_documents(
        query=state["query"],
        strategy=state["strategy"],
        top_k=10,
    )

    state["chunks"] = chunks
    logger.info(f"Retrieved {len(chunks)} chunks using {state['strategy']}")
    return state


async def evaluate_node(state: AgentState) -> AgentState:
    """Evaluate retrieval quality to decide if requery is needed."""
    quality = await evaluate_retrieval_quality(
        query=state["query"],
        chunks=state["chunks"],
    )

    state["quality_score"] = quality
    logger.info(f"Retrieval quality score: {quality:.3f}")
    return state


async def requery_node(state: AgentState) -> AgentState:
    """Refine the query and increment requery counter."""
    refined = await refine_query(
        original_query=state["original_query"],
        current_query=state["query"],
        chunks=state["chunks"],
    )

    state["query"] = refined
    state["requery_count"] = state.get("requery_count", 0) + 1
    logger.info(f"Requery #{state['requery_count']}: '{refined[:80]}...'")
    return state


async def generate_node(state: AgentState) -> AgentState:
    """Generate the final answer from retrieved context."""
    result = await generate_answer(
        query=state["original_query"],
        chunks=state["chunks"],
    )

    state["answer"] = result["text"]
    state["sources"] = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "content": c.get("content", "")[:500],
            "document_id": c.get("document_id", ""),
            "source": c.get("source", ""),
            "score": c.get("score", 0.0),
        }
        for c in state["chunks"][:5]
    ]
    state["metadata"] = {
        "model": result.get("model", ""),
        "requery_count": state.get("requery_count", 0),
        "quality_score": state.get("quality_score", 0.0),
        "strategy": state["strategy"],
    }

    logger.info(f"Generated answer ({len(state['answer'])} chars)")
    return state


# ── Conditional edges ─────────────────────────────────────────────────────────

def should_requery(state: AgentState) -> str:
    """Decide whether to requery or proceed to generation."""
    quality = state.get("quality_score", 0.0)
    requery_count = state.get("requery_count", 0)
    num_chunks = len(state.get("chunks", []))

    # Force generation if:
    # 1. Quality is good enough (>= 0.7)
    # 2. Already requeried MAX_REQUERY times
    # 3. No chunks found at all (nothing to improve)
    if quality >= 0.7 or requery_count >= MAX_REQUERY or num_chunks == 0:
        return "generate"

    return "requery"


# ── Graph construction ────────────────────────────────────────────────────────

def build_rag_agent_graph() -> StateGraph:
    """Build and compile the agentic RAG workflow graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("requery", requery_node)
    graph.add_node("generate", generate_node)

    # Define edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "evaluate")

    # Conditional: evaluate → requery OR generate
    graph.add_conditional_edges(
        "evaluate",
        should_requery,
        {
            "requery": "requery",
            "generate": "generate",
        },
    )

    # Requery loops back to retrieve
    graph.add_edge("requery", "retrieve")

    # Generate is the terminal node
    graph.add_edge("generate", END)

    return graph


# ── Compiled graph (singleton) ────────────────────────────────────────────────

_compiled_graph = None


def get_rag_agent():
    """Get the compiled RAG agent graph."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_rag_agent_graph()
        _compiled_graph = graph.compile()
    return _compiled_graph


async def run_agent_query(query: str) -> dict:
    """
    Run a query through the agentic RAG workflow.
    Returns the final state with answer, sources, and metadata.
    """
    agent = get_rag_agent()

    initial_state: AgentState = {
        "query": query,
        "original_query": query,
        "query_class": "",
        "strategy": "",
        "chunks": [],
        "answer": "",
        "quality_score": 0.0,
        "requery_count": 0,
        "sources": [],
        "metadata": {},
    }

    result = await agent.ainvoke(initial_state)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "strategy_used": result.get("strategy", "hybrid"),
        "query_class": result.get("query_class", "mixed"),
        "metadata": result.get("metadata", {}),
    }
