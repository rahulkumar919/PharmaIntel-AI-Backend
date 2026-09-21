"""
graph/streamer.py — async generator that drives the SSE stream.

Architecture
────────────
The browser opens a persistent GET /api/complaints/extract/stream connection.
FastAPI uses sse-starlette's EventSourceResponse, which calls our async
generator and forwards each yielded dict as one SSE frame:

    data: {"event": "node_progress", "node": "extract_fields", ...}\n\n

Why an async generator instead of a sync one?
  complaint_graph.stream() is synchronous (LangGraph 0.3.x has no native async
  streaming).  We run it in a thread-pool executor via run_in_executor so the
  event loop stays free while the graph executes CPU/IO-bound work (LLM calls,
  embedding inference, DB queries).  The generator bridges the sync graph loop
  into async SSE frames.

SSE event sequence for a normal run:
  node_progress  parse_document    started
  node_progress  parse_document    completed
  node_progress  extract_fields    started
  node_progress  extract_fields    completed
  node_progress  completeness_check started
  node_progress  completeness_check completed
  node_progress  risk_classification started
  ...
  node_progress  summary_node      completed
  done           (ExtractResponse JSON in detail)

SSE event sequence when ask_user fires:
  ...
  node_progress  completeness_check completed
  node_progress  ask_user           started
  interrupt      (clarifying_question in detail)
  → stream closes; browser shows the question in the chat box
  → user answers via POST /chat → graph resumes → new SSE stream or sync response

Why emit both "started" and "completed" per node?
  The "started" event lets the frontend show a spinner immediately; the
  "completed" event lets it check off the step.  Without "started", there is
  a perceptible delay between checkmarks while a slow LLM call runs.

Thread-safety note:
  run_in_executor runs the sync graph loop in a separate OS thread.  The
  async queue bridges the two threads safely — the producer thread puts items;
  the consumer (event loop) gets them.  asyncio.Queue is designed for this
  cross-thread pattern when combined with loop.call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator
from uuid import UUID

from app.graph.graph import complaint_graph, get_thread_config
from app.graph.session_store import (
    mark_complete,
    mark_interrupted,
)
from app.graph.state import ComplaintState
from app.schemas.api import NODE_LABELS, NODE_SEQUENCE, NodeProgressEvent
from app.schemas.complaint import Complaint, ComplaintAnalysis, ComplaintExtraction

logger = logging.getLogger(__name__)

# Sentinel placed in the queue by the producer thread to signal "done"
_SENTINEL = object()


# ── SSE frame helpers ─────────────────────────────────────────────────────────

def _progress_frame(node: str, status: str, detail: str | None = None) -> dict:
    """
    Build an SSE frame dict for sse-starlette.

    sse-starlette expects {"event": str, "data": str}.
    We serialise the payload as JSON in `data` so the frontend can
    JSON.parse(event.data) in one line.
    """
    payload = NodeProgressEvent(
        event="node_progress",
        node=node,
        status=status,
        detail=detail or NODE_LABELS.get(node, node),
    )
    return {
        "event": "node_progress",
        "data": payload.model_dump_json(),
    }


def _interrupt_frame(question: str, missing_field: str, session_id: UUID) -> dict:
    data = json.dumps({
        "session_id": str(session_id),
        "clarifying_question": question,
        "missing_field": missing_field,
    })
    return {"event": "interrupt", "data": data}


def _done_frame(session_id: UUID, state: dict, clarifying_question: str | None = None) -> dict:
    """
    Build the final SSE frame carrying the complete ExtractResponse payload.
    The frontend uses this to hydrate the form in one atomic update.
    """
    from app.schemas.api import ExtractResponse

    analysis = ComplaintAnalysis(
        risk_justification=state.get("risk_justification"),
        duplicate_ids=state.get("duplicate_ids", []),
        duplicate_scores=state.get("duplicate_scores", []),
        root_cause_hypothesis=state.get("root_cause_hypothesis"),
        capa_recommendation=state.get("capa_text"),
        summary=state.get("summary_text"),
    )
    extraction = state.get("extraction") or ComplaintExtraction()
    complaint = Complaint(
        id=session_id,
        extraction=extraction,
        analysis=analysis,
        raw_text=state.get("raw_text"),
    )
    response = ExtractResponse(
        session_id=session_id,
        complaint=complaint,
        clarifying_question=clarifying_question,
    )
    return {"event": "done", "data": response.model_dump_json()}


def _error_frame(message: str) -> dict:
    data = json.dumps({"message": message})
    return {"event": "error", "data": data}


# ── Core streaming generator ──────────────────────────────────────────────────

async def stream_graph_events(
    initial_state: ComplaintState | Any,
    session_id: UUID,
) -> AsyncGenerator[dict, None]:
    """
    Async generator consumed by EventSourceResponse.

    Each `yield` produces one SSE frame sent to the browser.
    The generator runs the sync LangGraph loop in a thread-pool executor and
    bridges output into the async event loop via asyncio.Queue.

    Parameters
    ----------
    initial_state : the initial ComplaintState dict OR a LangGraph Command
                    (for resume-after-interrupt calls)
    session_id    : used to key the LangGraph checkpointer thread
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[dict | object] = asyncio.Queue()

    # ── Producer: runs the sync graph in a background thread ─────────────────
    def _run_graph() -> None:
        """
        Synchronous graph runner executed in a thread-pool worker.

        Converts each LangGraph chunk into an SSE frame dict and puts it on
        the queue.  Puts _SENTINEL when done (or on error) so the consumer
        knows to stop.

        Why loop.call_soon_threadsafe + queue.put_nowait?
          asyncio.Queue is NOT thread-safe for direct .put() from a non-event-
          loop thread.  call_soon_threadsafe schedules the put on the event
          loop thread, which IS safe.  This is the canonical cross-thread
          asyncio pattern.
        """
        config = get_thread_config(str(session_id))
        running_state: dict = {}

        def _put(item: dict | object) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        try:
            prev_nodes: set[str] = set()  # track which nodes we've seen start

            for chunk in complaint_graph.stream(
                initial_state, config, stream_mode="updates"
            ):
                # ── Interrupt event ───────────────────────────────────────────
                if "__interrupt__" in chunk:
                    interrupts = chunk["__interrupt__"]
                    payload: dict = {}
                    if interrupts:
                        first = interrupts[0]
                        payload = first.value if hasattr(first, "value") else {}

                    question = payload.get("clarifying_question", "Could you provide more details?")
                    missing_field = payload.get("missing_field", "")

                    mark_interrupted(session_id, question, missing_field)
                    logger.info("SSE: graph interrupted, missing_field=%s", missing_field)

                    _put(_progress_frame("ask_user", "started"))
                    _put(_interrupt_frame(question, missing_field, session_id))
                    _put(_SENTINEL)
                    return

                # ── Node update chunks ────────────────────────────────────────
                # stream_mode="updates" yields {"node_name": {state_delta}}.
                # Each key is a node that just completed.
                for node_name, node_deltas in chunk.items():
                    if node_name.startswith("__"):
                        continue  # skip LangGraph internal keys

                    # Emit "started" for this node if we haven't yet
                    # (LangGraph only tells us when a node completes, not when
                    # it starts — we emit started immediately before completed
                    # so the frontend doesn't wait until the node finishes to
                    # show any activity indicator).
                    if node_name not in prev_nodes:
                        _put(_progress_frame(node_name, "started"))
                        prev_nodes.add(node_name)

                    # Merge state delta
                    if isinstance(node_deltas, dict):
                        running_state.update(node_deltas)

                    # Emit "completed"
                    _put(_progress_frame(node_name, "completed"))

            # ── Graph finished normally ───────────────────────────────────────
            mark_complete(session_id)
            _put(_done_frame(session_id, running_state))

        except Exception as exc:
            logger.exception("SSE graph error for session %s", session_id)
            _put(_error_frame(str(exc)))
        finally:
            _put(_SENTINEL)

    # Start the graph in a thread-pool worker (non-blocking for the event loop)
    executor_future = loop.run_in_executor(None, _run_graph)

    # ── Consumer: yield frames from the queue to the SSE response ──────────────
    try:
        while True:
            # 120s per-item timeout: if the graph hasn't produced any output in
            # 2 minutes, something is stuck (OOM, Groq timeout, deadlock).
            # This surfaces the problem as a visible SSE error frame instead of
            # a silent hang that Render kills with no log output.
            try:
                item = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                logger.error(
                    "SSE stream timed out waiting for graph output for session %s "
                    "(no event in 120s — possible OOM or stuck LLM call)",
                    session_id,
                )
                yield _error_frame(
                    "Processing timed out. The server may be under memory pressure. "
                    "Please retry your request."
                )
                break
            if item is _SENTINEL:
                break
            yield item  # type: ignore[misc]
    finally:
        # Make sure the executor task is awaited even if the client disconnects
        # early — prevents "coroutine was never awaited" warnings.
        # Allow up to 120s for a running LLM call to finish before giving up.
        try:
            await asyncio.wait_for(executor_future, timeout=120.0)
        except (asyncio.TimeoutError, Exception):
            pass
