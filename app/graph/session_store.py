"""
graph/session_store.py — in-process store that tracks per-session metadata.

Why do we need this alongside the LangGraph checkpointer?
  The LangGraph MemorySaver stores the *graph execution state* (all the
  ComplaintState fields).  But the route handlers also need to know:
    - Is this session currently interrupted (waiting for user input)?
    - What was the clarifying question that was asked?
    - Is the graph complete (no interrupt pending)?

  We could query the checkpointer directly for this, but the LangGraph
  checkpoint API is low-level and varies between checkpointer backends.
  A thin session store gives us a stable, backend-agnostic interface that
  the route handlers can read without knowing checkpoint internals.

  Think of it as the "control plane" (what state is the session in?) vs.
  the MemorySaver's "data plane" (what are the field values?).

Lifetime:
  Like MemorySaver, this lives in process memory — cleared on restart.
  For production, Phase 7 notes how to migrate both to a Redis or Postgres
  backend simultaneously.

Thread safety:
  FastAPI runs handlers in an async event loop (single OS thread by default
  for sync route handlers, or as coroutines for async ones).  A plain dict
  is safe for concurrent async access because Python's GIL ensures dict
  operations are atomic.  If you move to multiple uvicorn workers you will
  need to replace this with a shared backend (Redis, Postgres).
  ASSUMPTION: single-worker deployment for now — flagged for Phase 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID


@dataclass
class SessionRecord:
    """
    Metadata for one complaint-processing session.

    interrupted    : True while the graph is paused at ask_user, waiting
                     for the user to answer the clarifying question.
    clarifying_question : The question text surfaced to the user.
    missing_field  : The specific field name that triggered the question
                     (used by /chat to inject the answer back in the right
                     context).
    is_complete    : True once the graph has run all the way to END without
                     an interrupt — i.e. the complaint is fully processed.
    """
    session_id: UUID
    interrupted: bool = False
    clarifying_question: Optional[str] = None
    missing_field: Optional[str] = None
    is_complete: bool = False


# Module-level store: { str(session_id) → SessionRecord }
# Keyed by string (not UUID) because dict lookups are faster on strings and
# all callers already have the UUID serialised as a string for the thread_id.
_store: dict[str, SessionRecord] = {}


def create_session(session_id: UUID) -> SessionRecord:
    """Register a new session and return its record."""
    record = SessionRecord(session_id=session_id)
    _store[str(session_id)] = record
    return record


def get_session(session_id: UUID) -> Optional[SessionRecord]:
    """Return the record for an existing session, or None if not found."""
    return _store.get(str(session_id))


def mark_interrupted(
    session_id: UUID,
    clarifying_question: str,
    missing_field: str,
) -> None:
    """
    Called by the /extract route handler when .stream() yields an interrupt
    event.  Updates the record so /chat knows the session is waiting.
    """
    record = _store.get(str(session_id))
    if record:
        record.interrupted = True
        record.clarifying_question = clarifying_question
        record.missing_field = missing_field


def mark_resumed(session_id: UUID) -> None:
    """
    Called by the /chat route handler just before injecting the user's
    answer and resuming the graph.  Clears the interrupted flag.
    """
    record = _store.get(str(session_id))
    if record:
        record.interrupted = False
        record.clarifying_question = None
        record.missing_field = None


def mark_complete(session_id: UUID) -> None:
    """Called when the graph reaches END without an interrupt."""
    record = _store.get(str(session_id))
    if record:
        record.is_complete = True
        record.interrupted = False
