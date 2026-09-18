"""
graph/graph.py — assembles the LangGraph StateGraph and compiles it.

Why LangGraph StateGraph (not raw LangChain chains)?
  A StateGraph models the pipeline as an explicit directed graph of nodes and
  edges.  This gives us:
    1. Conditional routing (completeness_check → ask_user OR risk_classification)
    2. Human-in-the-loop interrupt() in ask_user (implemented Phase 3)
    3. Built-in streaming — .stream() yields events as each node completes,
       which we forward as SSE chunks to the frontend (Phase 5)
    4. Checkpointing — LangGraph persists state between HTTP requests, which
       is how interrupt/resume works across two separate API calls

Graph topology:
  START
    ↓
  parse_document
    ↓
  extract_fields  ←──────────────────────────────────────┐
    ↓                                                      │
  completeness_check ──(score < threshold)──→ ask_user ───┘
    ↓ (score ≥ threshold)                    (interrupt here; resume injects
  risk_classification                         user_clarification → re-extract)
    ↓
  duplicate_detection
    ↓
  capa_node
    ↓
  summary_node
    ↓
  END

Checkpointer — MemorySaver:
  MemorySaver stores checkpoints in a plain Python dict (in-process memory).
  Each checkpoint is keyed by (thread_id, checkpoint_id).  When .invoke() or
  .stream() is called with {"configurable": {"thread_id": "<id>"}}, LangGraph
  looks up the latest checkpoint for that thread and either starts fresh (no
  checkpoint found) or resumes from the interrupt point.

  Trade-off vs. a Postgres checkpointer:
    MemorySaver is lost on process restart — fine for a single-process dev
    server, but not for production with multiple workers or restarts.
    Phase 7 README documents how to swap this for langgraph-checkpoint-postgres
    for production without changing any node code.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from app.graph.state import ComplaintState
from app.graph.nodes import (
    parse_document,
    extract_fields,
    completeness_check,
    ask_user,
    risk_classification,
    duplicate_detection,
    capa_recommendation,
    summary_node,
)
from app.config import get_settings


def _route_after_completeness(state: ComplaintState) -> str:
    """
    Conditional edge function called after completeness_check.

    Returns the name of the next node to run.  LangGraph uses the return value
    as a key into the path_map dict passed to add_conditional_edges().

    Why a named function (not a lambda)?
      LangGraph serialises the routing function name into the checkpoint for
      reproducibility.  Named functions serialise cleanly; lambdas do not.
    """
    settings = get_settings()
    score = state.get("completeness_score", 0.0)

    if score < settings.completeness_threshold:
        return "ask_user"
    return "risk_classification"


def build_graph() -> StateGraph:
    """
    Construct the StateGraph.  Called once; the result is compiled below.

    Note: the graph topology is the same as Phase 1.  The only structural
    change in Phase 3 is that .compile() now receives a checkpointer, which
    enables interrupt() to actually persist and resume state.  Without a
    checkpointer, calling interrupt() raises a runtime error.
    """
    graph = StateGraph(ComplaintState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("parse_document",      parse_document)
    graph.add_node("extract_fields",      extract_fields)
    graph.add_node("completeness_check",  completeness_check)
    graph.add_node("ask_user",            ask_user)
    graph.add_node("risk_classification", risk_classification)
    graph.add_node("duplicate_detection", duplicate_detection)
    graph.add_node("capa_node",           capa_recommendation)
    graph.add_node("summary_node",        summary_node)

    # ── Deterministic edges ───────────────────────────────────────────────────
    graph.add_edge(START,                 "parse_document")
    graph.add_edge("parse_document",      "extract_fields")
    graph.add_edge("extract_fields",      "completeness_check")

    # ── Conditional edge: completeness gate ───────────────────────────────────
    graph.add_conditional_edges(
        "completeness_check",
        _route_after_completeness,
        {
            "ask_user":            "ask_user",
            "risk_classification": "risk_classification",
        },
    )

    # ask_user → extract_fields loop:
    # After the interrupt resumes, ask_user returns normally (the user_clarification
    # branch), and the graph continues to extract_fields which re-runs with the
    # clarification appended to raw_text.
    graph.add_edge("ask_user",            "extract_fields")

    graph.add_edge("risk_classification", "duplicate_detection")
    graph.add_edge("duplicate_detection", "capa_node")
    graph.add_edge("capa_node",           "summary_node")
    graph.add_edge("summary_node",        END)

    return graph


# ── Checkpointer ──────────────────────────────────────────────────────────────
# Module-level singleton so it is shared across all requests in the process.
# Sharing is essential — if each request created its own MemorySaver, the
# checkpoint written by /extract would not be visible to /chat.
_checkpointer = MemorySaver()

# ── Compiled graph ────────────────────────────────────────────────────────────
# Compiled once at import time; reused for every request.
# The checkpointer is what makes interrupt() work — LangGraph writes the
# checkpoint to _checkpointer before surfacing the interrupt event.
complaint_graph = build_graph().compile(checkpointer=_checkpointer)


def get_thread_config(thread_id: str) -> dict:
    """
    Build the LangGraph run config dict for a given thread_id.

    Every .invoke() / .stream() call that should share state (i.e. the
    /extract call that starts the graph AND the /chat call that resumes it)
    must pass the same thread_id in the configurable dict.

    LangGraph uses thread_id as the primary key into the checkpointer store.
    Using the session UUID as thread_id means each complaint processing run
    is isolated — two concurrent uploads don't interfere with each other.
    """
    return {"configurable": {"thread_id": str(thread_id)}}
