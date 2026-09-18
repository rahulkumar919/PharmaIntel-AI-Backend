"""
graph/state.py — the single shared state object threaded through every LangGraph node.

Why a TypedDict instead of a Pydantic model?
  LangGraph's StateGraph requires the state type to be a TypedDict (or a
  dataclass).  It uses the type annotations to know which keys are allowed and
  — crucially — how to merge partial updates returned by each node.  Each node
  returns only the keys it changed; LangGraph shallow-merges that dict back
  into the full state.  A Pydantic BaseModel can't be used directly here
  because LangGraph needs dict-style mutation, not Pydantic's immutable
  model semantics.

  We still get Pydantic's validation where it matters: the extraction and
  analysis fields are Pydantic models *stored inside* the TypedDict, so the
  LLM output is validated the moment it's assigned.

Key design decision — single state object:
  Every node reads from and writes to the same ComplaintState.  This means
  later nodes automatically have access to everything earlier nodes produced
  (e.g. capa_recommendation can see the extraction AND the risk justification)
  without any manual data-passing.  The graph is the data pipeline.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID
from typing_extensions import TypedDict   # TypedDict from typing_extensions works
                                           # across Python 3.11 and LangGraph's internals

from app.schemas.complaint import ComplaintExtraction, ComplaintAnalysis


class ComplaintState(TypedDict, total=False):
    """
    The complete state envelope for one complaint processing run.

    `total=False` means every key is optional at the TypedDict level — this is
    intentional because LangGraph nodes return *partial* dicts (only the keys
    they update), and Python's type checker would complain about missing keys
    otherwise.  The actual required-field enforcement lives in the
    completeness_check node logic.

    Field groups mirror the graph node that primarily writes them:
      parse_document      → raw_text, file_name, file_type
      extract_fields      → extraction, extraction_attempts
      completeness_check  → completeness_score, missing_fields
      ask_user            → clarifying_question, user_clarification
      risk_classification → (writes into extraction.initial_severity / priority)
                            risk_justification
      duplicate_detection → duplicate_ids, duplicate_scores
      capa_recommendation → root_cause_hypothesis, capa_recommendation
      summary_node        → summary
      (system)            → session_id, error
    """

    # ── Session metadata ──────────────────────────────────────────────────────
    session_id: UUID
    # Tracks which node is currently executing — useful for SSE progress events
    current_node: str

    # ── parse_document inputs / outputs ──────────────────────────────────────
    # file_bytes is written by the route handler before the graph starts;
    # parse_document reads it, extracts text, and writes raw_text.
    # We store bytes in state (not on disk) to keep the graph self-contained.
    # ASSUMPTION: complaints are small documents (<5 MB); storing bytes in
    # memory is acceptable.  For large files a temp-file path would be safer.
    file_bytes: Optional[bytes]  # raw uploaded file bytes — cleared after parsing
    raw_text: str                # full extracted text from the uploaded file
    file_name: Optional[str]     # original filename, kept for audit logging
    file_type: Optional[str]     # "pdf" | "docx" | "txt" | "eml" | "text"

    # ── extract_fields outputs ────────────────────────────────────────────────
    extraction: ComplaintExtraction
    # How many extraction attempts were made (normal = 1; after JSON repair = 2).
    # Stored so we can log repair-rate metrics over time.
    extraction_attempts: int

    # ── completeness_check outputs ────────────────────────────────────────────
    # Float 0.0–1.0: fraction of required fields that are non-null.
    completeness_score: float
    # List of field names that are still None after extraction.
    missing_fields: list[str]

    # ── ask_user outputs ──────────────────────────────────────────────────────
    # The single targeted question the LLM generated for the most critical gap.
    clarifying_question: Optional[str]
    # The user's reply, injected back into state when the graph resumes.
    user_clarification: Optional[str]

    # ── risk_classification outputs ───────────────────────────────────────────
    # Short prose justification — stored for the chat Q&A context but not shown
    # on the complaint form itself.
    risk_justification: Optional[str]

    # ── duplicate_detection outputs ───────────────────────────────────────────
    duplicate_ids: list[UUID]     # IDs of similar past complaints
    duplicate_scores: list[float] # corresponding cosine similarity values

    # ── capa_recommendation node outputs ─────────────────────────────────────
    # Named with _text suffix to avoid colliding with the LangGraph node name
    # "capa_node". LangGraph 0.3.x forbids a node name that matches a state key.
    root_cause_hypothesis: Optional[str]
    capa_text: Optional[str]          # was: capa_recommendation

    # ── summary_node output ───────────────────────────────────────────────────
    summary_text: Optional[str]       # was: summary (avoid collision with any node named "summary")

    # ── error handling ────────────────────────────────────────────────────────
    # If any node sets this, the graph can route to an error-reporting edge
    # instead of crashing the entire request.
    error: Optional[str]
