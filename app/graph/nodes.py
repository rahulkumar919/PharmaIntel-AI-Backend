"""
graph/nodes.py — LangGraph node functions for the complaint processing pipeline.

Phase status:
  parse_document      ✓ REAL  (Phase 2) — LangChain document loaders
  extract_fields      ✓ REAL  (Phase 2) — ChatGroq structured output + repair
  completeness_check  ✓ REAL  (Phase 3) — pure logic scoring, no LLM
  ask_user            ✓ REAL  (Phase 3) — LLM question + LangGraph interrupt()
  risk_classification ✓ REAL  (Phase 4) — ChatGroq extraction_model
  duplicate_detection ✓ REAL  (Phase 4) — sentence-transformers + pgvector
  capa_recommendation ✓ REAL  (Phase 4) — ChatGroq capa_model
  summary_node        ✓ REAL  (Phase 4) — ChatGroq extraction_model

Why each node returns a dict (not the full state):
  LangGraph merges the returned dict into the running state shallowly.
  Returning only what changed keeps each node focused and prevents accidental
  overwrites of another node's output.
"""

from __future__ import annotations

import logging

from app.graph.ask_user_logic import generate_clarifying_question
from app.graph.capa_generator import generate_capa
from app.graph.document_parser import extract_text
from app.graph.duplicate_detector import find_duplicates
from app.graph.extractor import ExtractionError, run_extraction
from app.graph.risk_classifier import classify_risk
from app.graph.state import ComplaintState
from app.graph.summariser import generate_summary
from app.schemas.complaint import ComplaintExtraction

logger = logging.getLogger(__name__)


# ── Node 1: parse_document ────────────────────────────────────────────────────

def parse_document(state: ComplaintState) -> dict:
    """
    Extract plain text from the uploaded file or pass through raw text.

    Routing logic (in priority order):
      1. If state["raw_text"] is already set (paste input) → pass it straight
         through; no file parsing needed.
      2. Otherwise read state["file_bytes"] + state["file_type"] and route to
         the correct LangChain loader via document_parser.extract_text().

    Why this is its own node (not merged with extract_fields):
      Parsing is I/O-bound and fails for different reasons than LLM calls
      (bad file format vs. API error).  Separating them gives the frontend
      a distinct "Parsing document…" SSE event before the longer LLM wait,
      and makes error messages unambiguous.

    State keys read  : raw_text, file_bytes, file_name, file_type
    State keys written: raw_text, file_type, current_node, error
    """
    logger.info("parse_document: starting")

    # Fast path — caller already extracted the text (paste input via route handler)
    if state.get("raw_text"):
        logger.info(
            "parse_document: raw_text already present (%d chars), skipping file parse",
            len(state["raw_text"]),
        )
        return {"current_node": "parse_document"}

    # File upload path — bytes were stored in state by the route handler
    file_bytes: bytes | None = state.get("file_bytes")  # type: ignore[assignment]
    file_name: str | None = state.get("file_name")
    file_type: str | None = state.get("file_type", "text")

    if not file_bytes:
        # Neither raw_text nor file_bytes were provided — this is a caller error
        error_msg = "parse_document: no raw_text and no file_bytes in state"
        logger.error(error_msg)
        return {"error": error_msg, "current_node": "parse_document"}

    try:
        text = extract_text(
            file_bytes=file_bytes,
            file_name=file_name,
            file_type=file_type,
        )
        logger.info("parse_document: extracted %d chars from '%s'", len(text), file_name)
        return {
            "raw_text": text,
            "current_node": "parse_document",
        }
    except ValueError as exc:
        # extract_text raises ValueError for unsupported types or empty results
        error_msg = f"parse_document failed: {exc}"
        logger.error(error_msg)
        return {"error": error_msg, "current_node": "parse_document"}


# ── Node 2: extract_fields ────────────────────────────────────────────────────

def extract_fields(state: ComplaintState) -> dict:
    """
    Call ChatGroq (gemma2-9b-it) to extract structured fields from raw_text.

    Delegates entirely to extractor.run_extraction() which owns:
      - The system + human prompt construction
      - The with_structured_output() call (JSON-mode function calling)
      - The repair retry if the first response fails Pydantic validation

    Why this node is thin (just calls run_extraction):
      The node's job is state management — reading from and writing to
      ComplaintState.  The LLM orchestration logic belongs in extractor.py
      where it can be unit-tested independently of LangGraph.

    If user_clarification is set in state (i.e. the graph resumed after
    ask_user), we append it to the raw_text so the LLM has the additional
    context the user provided when re-extracting.

    State keys read  : raw_text, user_clarification, extraction_attempts
    State keys written: extraction, extraction_attempts, current_node, error
    """
    logger.info("extract_fields: starting")

    raw_text: str = state.get("raw_text", "")
    if not raw_text:
        error_msg = "extract_fields: raw_text is empty — parse_document may have failed"
        logger.error(error_msg)
        return {"error": error_msg, "current_node": "extract_fields"}

    # If the user answered a clarifying question, append their answer to the
    # text so the LLM can use it to fill in the missing fields on this retry.
    clarification = state.get("user_clarification")
    if clarification:
        logger.info("extract_fields: appending user clarification to raw_text")
        raw_text = (
            raw_text
            + "\n\n--- USER CLARIFICATION ---\n"
            + clarification
        )

    # Track cumulative attempts across graph retries (normal run = 1 or 2;
    # after ask_user loop the counter accumulates so we can log repair rates).
    prior_attempts: int = state.get("extraction_attempts", 0)

    try:
        extraction, attempts_this_call = run_extraction(raw_text)
        logger.info(
            "extract_fields: succeeded in %d attempt(s) this call", attempts_this_call
        )
        return {
            "extraction": extraction,
            "extraction_attempts": prior_attempts + attempts_this_call,
            "current_node": "extract_fields",
        }
    except ExtractionError as exc:
        error_msg = f"extract_fields: {exc}"
        logger.error(error_msg)
        # Write a blank extraction so downstream nodes don't KeyError,
        # and set error so the route handler can surface a user-friendly message.
        return {
            "extraction": ComplaintExtraction(),
            "extraction_attempts": prior_attempts + 2,   # both attempts were consumed
            "error": error_msg,
            "current_node": "extract_fields",
        }


# ── Node 3: completeness_check ────────────────────────────────────────────────

# The fields a QA reviewer *must* have before a complaint can be processed.
# Deliberately minimal — we only block on truly critical missing fields.
# complaint_date and complaint_source are optional — many complaints don't
# explicitly state these and the graph should still proceed.
# The most important field is detailed_complaint_description: without it
# there is nothing to classify or generate CAPA from.
_REQUIRED_FIELDS: list[str] = [
    "detailed_complaint_description",   # without this, nothing can be done
    "product_name",                     # needed for risk classification
    "batch_lot_number",                 # needed for duplicate detection
]

# Fields ranked by importance — the first missing field in this order is the
# one ask_user will ask about.  Kept separate from _REQUIRED_FIELDS so the
# priority order can be adjusted independently of the completeness threshold.
_FIELD_PRIORITY: list[str] = [
    "detailed_complaint_description",   # most important — content of the complaint
    "product_name",
    "batch_lot_number",
    "complaint_type",
    "complaint_date",
    "customer_name",
    "complaint_source",
]


def completeness_check(state: ComplaintState) -> dict:
    """
    Score how many required fields the LLM successfully extracted.

    Pure Python logic — no LLM call.  Returns:
      completeness_score : float 0.0–1.0  (filled / total required)
      missing_fields     : list[str]  field names still None, ordered by
                           _FIELD_PRIORITY so ask_user always targets the
                           most important gap first.

    The conditional edge in graph.py reads completeness_score and routes to
    ask_user if it is below settings.completeness_threshold (default 0.75),
    or to risk_classification if it is above.

    Why deterministic (no LLM)?
      Completeness is binary per field: either the value is present or it isn't.
      An LLM would add latency, cost, and non-determinism to a check that is
      trivially expressed as `getattr(extraction, field) is None`.
    """
    extraction: ComplaintExtraction = state.get("extraction") or ComplaintExtraction()

    filled = [f for f in _REQUIRED_FIELDS if getattr(extraction, f, None) is not None]
    missing_required = [f for f in _REQUIRED_FIELDS if getattr(extraction, f, None) is None]

    score = len(filled) / len(_REQUIRED_FIELDS)

    # Re-order missing fields by _FIELD_PRIORITY so ask_user picks the most
    # important one first.  Fields not in the priority list go to the end.
    priority_index = {f: i for i, f in enumerate(_FIELD_PRIORITY)}
    missing_sorted = sorted(
        missing_required,
        key=lambda f: priority_index.get(f, len(_FIELD_PRIORITY)),
    )

    logger.info(
        "completeness_check: score=%.2f  filled=%d/%d  missing=%s",
        score, len(filled), len(_REQUIRED_FIELDS), missing_sorted,
    )

    return {
        "completeness_score": score,
        "missing_fields": missing_sorted,
        "current_node": "completeness_check",
    }


# ── Node 4: ask_user ──────────────────────────────────────────────────────────

def ask_user(state: ComplaintState) -> dict:
    """
    Generate a targeted clarifying question, pause the graph, and collect the
    user's answer — all in one node execution sequence.

    How LangGraph interrupt() works in 0.3.x (corrected pattern):
      First execution (no resume value yet):
        interrupt(value) raises GraphInterrupt internally.  LangGraph catches it,
        persists the checkpoint, and surfaces the interrupt value to the caller.
        The node does NOT return on this execution.

      Second execution (resume value available):
        interrupt(value) RETURNS the value passed by the caller via
        Command(resume=<answer>).  The node continues normally from that line,
        writes user_clarification into state, and returns.

      This means the node does NOT need to detect "am I resuming?" manually —
      interrupt() handles the branching internally.  The pattern is:
          answer = interrupt(question_payload)   # blocks first time, returns answer second time
          return {"user_clarification": answer}  # only reached on resume

    Why Command(resume=...) instead of update_state():
      update_state() is a direct checkpoint mutation that does not trigger the
      resume mechanism.  Command(resume=value) is the correct LangGraph API for
      supplying the answer to a pending interrupt — it injects the value into
      the scratchpad that interrupt() reads on the second node execution.

    State keys read  : missing_fields, extraction (for context in question generation)
    State keys written: clarifying_question, user_clarification, current_node
    """
    from langgraph.types import interrupt as lg_interrupt  # local import avoids
                                                           # circular import at load time

    missing: list[str] = state.get("missing_fields", [])

    if not missing:
        # Routed here but nothing is actually missing — shouldn't happen;
        # return cleanly so the graph continues to risk_classification.
        logger.warning("ask_user: no missing fields in state, skipping interrupt")
        return {"current_node": "ask_user"}

    target_field = missing[0]
    logger.info("ask_user: generating question for missing field '%s'", target_field)

    question = generate_clarifying_question(state, target_field)

    # First execution: interrupt() raises GraphInterrupt — node stops here.
    # Second execution (after Command(resume=answer)): interrupt() returns
    # the user's answer string and execution continues to the return below.
    user_answer = lg_interrupt({
        "clarifying_question": question,
        "missing_field":       target_field,
    })

    # Only reached on resume — user_answer is what the caller passed via Command(resume=...)
    logger.info("ask_user: received user answer: %s", str(user_answer)[:80])
    return {
        "clarifying_question": question,
        "user_clarification":  str(user_answer) if user_answer else "",
        "current_node":        "ask_user",
    }


# ── Node 5: risk_classification ───────────────────────────────────────────────

def risk_classification(state: ComplaintState) -> dict:
    """
    Assign initial_severity + priority and produce a short justification.

    Delegates to risk_classifier.classify_risk() which builds the prompt,
    calls ChatGroq (extraction_model), and returns a validated RiskOutput.

    Why we overwrite extraction.initial_severity / priority here (not in
    extract_fields)?
      Extract_fields uses a general extraction prompt that may return rough
      severity estimates.  This dedicated node uses a domain-specific risk
      prompt with ICH Q10 criteria for consistent, auditable classifications.
      Two separate prompts → two separate LLM calls → cleaner separation of
      concerns and better accuracy for each task.

    State keys read  : extraction
    State keys written: extraction (updated severity/priority), risk_justification
    """
    logger.info("risk_classification: starting")

    extraction = state.get("extraction") or ComplaintExtraction()
    risk = classify_risk(extraction)

    # Write severity + priority back into the extraction object so they appear
    # in the form.  model_copy(update=...) returns a new Pydantic instance —
    # Pydantic v2 models are immutable; this is the correct mutation pattern.
    updated_extraction = extraction.model_copy(update={
        "initial_severity": risk.initial_severity,
        "priority": risk.priority,
    })

    logger.info(
        "risk_classification: severity=%s priority=%s",
        risk.initial_severity.value,
        risk.priority.value,
    )

    return {
        "extraction": updated_extraction,
        "risk_justification": risk.justification,
        "current_node": "risk_classification",
    }


# ── Node 6: duplicate_detection ───────────────────────────────────────────────

def duplicate_detection(state: ComplaintState) -> dict:
    """
    Embed the complaint description and search pgvector for similar past complaints.

    Delegates to duplicate_detector.find_duplicates() which:
      1. Calls embedder.embed_text() to generate a 384-dim vector
      2. Runs a pgvector cosine-similarity query (<=>) against the DB
      3. Returns IDs and scores for matches above the threshold

    Graceful degradation:
      If the DB is not running (e.g. during development without Postgres),
      find_duplicates() catches the exception and returns ([], []).
      The graph continues — duplicate detection is informational, not blocking.

    Why sentence-transformers (not a cloud API)?
      Groq has no embeddings endpoint.  all-MiniLM-L6-v2 runs locally, is free,
      and produces 384-dim vectors matching our VECTOR(384) column.

    State keys read  : extraction (for description text)
    State keys written: duplicate_ids, duplicate_scores
    """
    logger.info("duplicate_detection: starting")

    extraction = state.get("extraction") or ComplaintExtraction()
    description = extraction.detailed_complaint_description or ""

    ids, scores = find_duplicates(description)

    if ids:
        logger.info(
            "duplicate_detection: found %d duplicate(s), top similarity=%.3f",
            len(ids), scores[0],
        )
    else:
        logger.info("duplicate_detection: no duplicates found above threshold")

    return {
        "duplicate_ids": ids,
        "duplicate_scores": scores,
        "current_node": "duplicate_detection",
    }


# ── Node 7: capa_recommendation ───────────────────────────────────────────────

def capa_recommendation(state: ComplaintState) -> dict:
    """
    Generate root-cause hypothesis and CAPA using the heavier capa_model.

    Delegates to capa_generator.generate_capa() which builds a rich prompt
    from the full state (extraction + risk_justification + duplicate context)
    and calls ChatGroq with the capa_model (openai/gpt-oss-120b).

    Why the heavier model?
      CAPA generation requires multi-step pharma domain reasoning: identifying
      the probable root cause (e.g. sealing machine temperature variance),
      proposing specific corrective actions (quarantine, inspection), and
      proposing preventive actions (SOP changes, in-process controls).
      The lightweight extraction model produces vague, generic CAPA text.
      The 120B model produces substantially more specific, actionable output.
      We deliberately pay the higher latency/cost only for this one node.

    State keys read  : extraction, risk_justification, duplicate_ids/scores
    State keys written: root_cause_hypothesis, capa_text
    """
    logger.info("capa_recommendation: starting (using capa_model)")

    capa = generate_capa(state)

    return {
        "root_cause_hypothesis": capa.root_cause_hypothesis,
        "capa_text": capa.capa_recommendation,
        "current_node": "capa_node",
    }


# ── Node 8: summary_node ──────────────────────────────────────────────────────

def summary_node(state: ComplaintState) -> dict:
    """
    Generate a 2-3 sentence human-readable summary for the chat sidebar.

    Delegates to summariser.generate_summary() which calls the lightweight
    extraction_model with the full complaint context (product, batch, severity,
    CAPA recommendation) to produce a polished digest sentence.

    Why generate this separately from CAPA?
      The summary is consumer-facing (shown in the chat sidebar to the reviewer).
      The CAPA text is operational (stored in the QMS and actioned by QA staff).
      Different audiences → different tone → separate prompt → separate node.

    State keys read  : extraction, risk_justification, root_cause_hypothesis, capa_text
    State keys written: summary_text
    """
    logger.info("summary_node: starting")

    summary = generate_summary(state)
    logger.info("summary_node: generated %d-char summary", len(summary))

    return {
        "summary_text": summary,
        "current_node": "summary_node",
    }
