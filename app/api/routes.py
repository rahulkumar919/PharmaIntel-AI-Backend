"""
api/routes.py — FastAPI route handlers, Phase 5.

What changed from Phase 4:
  /extract/stream (NEW GET endpoint) — streams LangGraph node progress over
    Server-Sent Events.  The frontend opens an EventSource to this URL; each
    node emits a "node_progress" frame, an "interrupt" or "done" frame closes
    the stream.  This is the primary endpoint for the frontend progress bar.

  /extract (POST, kept) — synchronous fallback that waits for the full graph
    to complete before returning.  Used by non-browser clients (curl, tests)
    that don't support EventSource.  Internally calls the same graph; just
    doesn't stream intermediate events.

SSE frame vocabulary (event: field):
  node_progress  — {"node": str, "status": "started"|"completed", "detail": str}
  interrupt      — {"session_id": str, "clarifying_question": str, "missing_field": str}
  done           — ExtractResponse JSON (same schema as POST /extract response)
  error          — {"message": str}
"""

from __future__ import annotations

import json
import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.db.engine import get_db
from app.db.models import ComplaintORM
from app.graph.embedder import embed_text
from app.graph.graph import complaint_graph, get_thread_config
from app.graph.session_store import (
    create_session,
    get_session,
    mark_complete,
    mark_interrupted,
    mark_resumed,
)
from app.graph.state import ComplaintState
from app.graph.streamer import stream_graph_events
from app.schemas.api import (
    ChatRequest,
    ChatResponse,
    ComplaintDetailResponse,
    ConfirmComplaintRequest,
    ConfirmComplaintResponse,
    ExtractResponse,
)
from app.schemas.complaint import Complaint, ComplaintAnalysis, ComplaintExtraction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/complaints", tags=["complaints"])


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _state_to_complaint(state: ComplaintState, session_id: UUID) -> Complaint:
    """
    Map LangGraph state keys → Complaint response model.

    Single place that knows about the capa_text / summary_text rename that was
    required to avoid LangGraph node-name collisions (see state.py comments).
    """
    analysis = ComplaintAnalysis(
        risk_justification=state.get("risk_justification"),
        duplicate_ids=state.get("duplicate_ids", []),
        duplicate_scores=state.get("duplicate_scores", []),
        root_cause_hypothesis=state.get("root_cause_hypothesis"),
        capa_recommendation=state.get("capa_text"),
        summary=state.get("summary_text"),
    )
    extraction = state.get("extraction") or ComplaintExtraction()
    return Complaint(
        id=session_id,
        extraction=extraction,
        analysis=analysis,
        raw_text=state.get("raw_text"),
    )


def _run_graph_to_completion_or_interrupt(
    initial_state,
    session_id: UUID,
) -> tuple[ComplaintState | None, str | None]:
    """
    Stream the graph and handle two possible outcomes:

    1. Graph runs to END (no interrupt):
       Returns (final_state, None).

    2. Graph hits interrupt() in ask_user:
       In LangGraph 0.3.x, interrupt() is only visible via stream_mode="updates".
       stream_mode="values" never emits __interrupt__ — it just silently skips it.
       We stream with "updates" and watch for the '__interrupt__' key in each chunk.
       Returns (None, question_text).

    stream_mode="updates" yields per-node delta dicts (only changed keys).
    We accumulate them into running_state to reconstruct the full state when done.

    Phase 5 wraps this same loop to emit SSE events — the logic here is unchanged;
    only an outer async generator is added around it.
    """
    config = get_thread_config(str(session_id))
    running_state: dict = {}

    try:
        for chunk in complaint_graph.stream(
            initial_state, config, stream_mode="updates"
        ):
            if "__interrupt__" in chunk:
                interrupts = chunk["__interrupt__"]
                payload = {}
                if interrupts:
                    first = interrupts[0]
                    payload = first.value if hasattr(first, "value") else {}

                question = payload.get("clarifying_question", "Could you provide more details?")
                missing_field = payload.get("missing_field", "")
                mark_interrupted(session_id, question, missing_field)
                logger.info(
                    "Graph interrupted for session %s — missing field: %s",
                    session_id, missing_field,
                )
                return None, question

            # Regular update chunk: {"node_name": {state_delta}}
            # Merge all node deltas into running_state
            for node_updates in chunk.values():
                if isinstance(node_updates, dict):
                    running_state.update(node_updates)

        mark_complete(session_id)
        # Cast accumulated dict to ComplaintState — it has all the same keys
        return running_state, None  # type: ignore[return-value]

    except Exception as exc:
        logger.exception("Graph execution error for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Graph error: {exc}") from exc


# ── GET /api/complaints/extract/stream (SSE) ──────────────────────────────────

@router.get("/extract/stream")
async def extract_complaint_stream(
    session_id: UUID = Query(..., description="Session UUID returned by a prior POST /extract call, OR a new UUID generated by the client before opening the stream."),
    raw_text: str | None = Query(None, description="Complaint text (URL-encoded). Supply this OR use the multipart POST endpoint first."),
):
    """
    Stream LangGraph node progress as Server-Sent Events.

    Usage pattern (two options):

    Option A — text input, single round-trip:
      1. Client generates a UUID (session_id).
      2. Client opens EventSource to:
           GET /api/complaints/extract/stream?session_id=<uuid>&raw_text=<encoded>
      3. Server streams node_progress events then a final "done" event.

    Option B — file upload then stream:
      1. Client POSTs the file to POST /extract (gets session_id back).
      2. Client opens EventSource to:
           GET /api/complaints/extract/stream?session_id=<uuid>
         (raw_text omitted — graph re-uses the checkpoint already created by POST)
      NOTE: Phase 6 frontend uses Option A for paste and Option B for file uploads.

    SSE event types:
      node_progress : {"node": str, "status": "started"|"completed", "detail": str}
      interrupt     : {"session_id": str, "clarifying_question": str, "missing_field": str}
      done          : ExtractResponse JSON — hydrate the form with this
      error         : {"message": str}

    Why GET (not POST) for SSE?
      EventSource (the browser API) only supports GET requests.  We work around
      the inability to send a body by accepting raw_text as a query parameter for
      small inputs.  Large file uploads use the two-step pattern (Option B).

    Why EventSourceResponse (sse-starlette)?
      It handles the text/event-stream Content-Type header, keep-alive pings,
      client-disconnect detection, and proper chunked transfer encoding — all
      the boilerplate you'd otherwise have to write manually with StreamingResponse.
    """
    create_session(session_id)

    # Build the initial state for this streaming run.
    # If raw_text is supplied, run the full graph from scratch.
    # If not, check whether a checkpoint already exists for this session_id
    # (Option B — the POST /extract already stored state in the MemorySaver).
    config = get_thread_config(str(session_id))
    existing_checkpoint = complaint_graph.get_state(config)

    if raw_text:
        initial_state: ComplaintState = {
            "session_id": session_id,
            "current_node": "start",
            "raw_text": raw_text,
            "file_type": "text",
            "extraction_attempts": 0,
            "duplicate_ids": [],
            "duplicate_scores": [],
            "missing_fields": [],
            "completeness_score": 0.0,
        }
    elif existing_checkpoint and existing_checkpoint.values:
        # Resume from existing checkpoint (e.g. after file upload via POST)
        # LangGraph re-runs from the last completed node.
        # Passing None means "use the checkpoint as-is, continue from there."
        initial_state = None   # type: ignore[assignment]
    else:
        # Neither raw_text nor an existing checkpoint — can't proceed
        async def _error_gen():
            import json as _json
            yield {"event": "error", "data": _json.dumps(
                {"message": "Provide raw_text or upload a file via POST /extract first."}
            )}
        return EventSourceResponse(_error_gen())

    return EventSourceResponse(
        stream_graph_events(initial_state, session_id),
        # ping=20 sends a comment (": ping") every 20s to keep the connection
        # alive through load balancers that close idle HTTP connections.
        ping=20,
    )


# ── POST /api/complaints/extract ───────────────────────────────────────────────

@router.post("/extract", response_model=ExtractResponse)
async def extract_complaint(
    file: UploadFile | None = File(None),
    raw_text: str | None = Form(None),
):
    """
    Synchronous fallback — accept file OR text, run the full graph, return JSON.

    Primary use cases:
      - File uploads (browser can't send a body via EventSource)
      - Non-browser clients (curl, pytest, integration tests)
      - Clients that don't support EventSource

    For browser file uploads the recommended flow is:
      1. POST /extract with the file → get session_id back
      2. Open EventSource to GET /extract/stream?session_id=<id>
         to show the progress bar while the graph runs

    For paste input, the browser can use GET /extract/stream directly with
    raw_text as a query parameter (avoids the two-step round-trip).

    Response is identical to the "done" SSE event payload.
    """
    if file is None and not raw_text:
        raise HTTPException(
            status_code=422,
            detail="Provide either a file upload (multipart) or raw_text (form field).",
        )

    session_id = uuid4()
    create_session(session_id)

    # Build initial state
    initial_state: ComplaintState = {
        "session_id": session_id,
        "current_node": "start",
        "extraction_attempts": 0,
        "duplicate_ids": [],
        "duplicate_scores": [],
        "missing_fields": [],
        "completeness_score": 0.0,
    }

    if file is not None:
        file_bytes = await file.read()
        suffix = (file.filename or "").rsplit(".", 1)[-1].lower()
        initial_state["file_name"] = file.filename
        initial_state["file_type"] = suffix if suffix in {"pdf", "docx", "txt", "eml"} else "text"
        initial_state["file_bytes"] = file_bytes
    else:
        initial_state["raw_text"] = raw_text
        initial_state["file_type"] = "text"

    final_state, clarifying_question = _run_graph_to_completion_or_interrupt(
        initial_state, session_id
    )

    # If interrupted, final_state is None — read the partial state from the
    # checkpoint so we can still return whatever was extracted before the pause.
    if final_state is None:
        config = get_thread_config(str(session_id))
        checkpoint = complaint_graph.get_state(config)
        # checkpoint.values is a dict of all state keys saved so far
        partial_state: ComplaintState = checkpoint.values if checkpoint else {}
        complaint = _state_to_complaint(partial_state, session_id)
    else:
        complaint = _state_to_complaint(final_state, session_id)

    return ExtractResponse(
        session_id=session_id,
        complaint=complaint,
        clarifying_question=clarifying_question,
    )


# ── POST /api/complaints/transcribe ───────────────────────────────────────────

@router.post("/transcribe", summary="Transcribe audio to text via Groq Whisper")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio recording file (webm, wav, m4a, mp3)"),
):
    """
    Transcribe speech to text using Groq's high-speed Whisper Large v3 Turbo model.
    Accepts browser MediaRecorder audio (e.g. audio/webm or audio/wav).
    """
    from app.config import get_settings
    from groq import Groq

    settings = get_settings()
    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload received.")

    filename = file.filename or "recording.webm"
    content_type = file.content_type or "audio/webm"

    try:
        client = Groq(api_key=settings.groq_api_key)
        try:
            transcription = client.audio.transcriptions.create(
                file=(filename, audio_bytes, content_type),
                model="whisper-large-v3-turbo",
                language="en",
                response_format="json",
            )
            text = transcription.text if hasattr(transcription, "text") else str(transcription)
        except Exception as turbo_err:
            logger.warning("Whisper turbo failed (%s), falling back to whisper-large-v3", turbo_err)
            transcription = client.audio.transcriptions.create(
                file=(filename, audio_bytes, content_type),
                model="whisper-large-v3",
                language="en",
                response_format="json",
            )
            text = transcription.text if hasattr(transcription, "text") else str(transcription)

        logger.info("Transcribed audio successfully (%d bytes -> %d chars: '%s')", len(audio_bytes), len(text), text[:60])
        return {
            "text": text.strip(),
            "model": "whisper-large-v3-turbo",
            "bytes_received": len(audio_bytes),
        }
    except Exception as exc:
        logger.error("Audio transcription failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")


# ── POST /api/complaints/chat ──────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat_with_complaint(body: ChatRequest):
    """
    Handle a follow-up message from the user.

    MODE A — RESUME (session is interrupted):
      The user has answered the clarifying question.  We:
        1. Read the current checkpoint state.
        2. Inject body.message as user_clarification into the state.
        3. Resume the graph (invoke with None as new input — LangGraph
           reads the checkpoint and continues from the interrupt point).
        4. If the graph completes: return the final extraction.
        5. If the graph interrupts again (another missing field): return
           the next clarifying question.

    MODE B — Q&A (session is complete or no interrupt pending):
      The user is asking a free-form question about the loaded complaint.
      We build a context prompt from the saved state and call the LLM
      to answer the question.  No graph resumption needed.

    Why pass None as the input to resume?
      LangGraph's checkpoint mechanism means the graph already has all its
      state saved.  Passing None tells .invoke() "don't add new state, just
      resume from where you left off."  The user's answer is injected via
      update_state() before resuming, which merges it into the checkpoint.
    """
    session = get_session(body.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {body.session_id} not found. Run /extract first.",
        )

    config = get_thread_config(str(body.session_id))

    # ── MODE A: resume from interrupt ─────────────────────────────────────────
    if session.interrupted:
        logger.info(
            "chat: resuming session %s with clarification: %s",
            body.session_id, body.message[:80],
        )

        mark_resumed(body.session_id)

        # Resume using Command(resume=value) — the correct LangGraph 0.3.x API.
        # Command(resume=value) injects `value` into the interrupt() scratchpad
        # so that when ask_user re-executes, interrupt() returns the value
        # instead of raising GraphInterrupt again.
        # Passing Command as the first argument to .invoke() signals "resume mode".
        from langgraph.types import Command

        final_state, next_question = _run_graph_to_completion_or_interrupt(
            Command(resume=body.message), body.session_id
        )

        if next_question:
            # Another field is still missing — surface the next question
            return ChatResponse(
                reply=next_question,
                updated_extraction=None,
            )

        # Graph completed — return the updated extraction so the frontend
        # can re-hydrate the form with the now-complete data.
        checkpoint = complaint_graph.get_state(config)
        state: ComplaintState = checkpoint.values if checkpoint else {}
        extraction = state.get("extraction") or ComplaintExtraction()

        return ChatResponse(
            reply=(
                "Thank you — I've updated the complaint record with the information "
                "you provided.  The form has been refreshed."
            ),
            updated_extraction=extraction,
        )

    # ── MODE B: free-form Q&A + field correction ─────────────────────────────
    logger.info("chat: Q&A mode for session %s: %s", body.session_id, body.message[:80])

    checkpoint = complaint_graph.get_state(config)
    state: ComplaintState = checkpoint.values if checkpoint else {}

    reply, updated_fields = _answer_question(body.message, state)

    # If the LLM detected field corrections, persist them into the checkpoint
    # so future chat turns have the corrected values as context.
    if updated_fields:
        current_extraction = state.get("extraction") or ComplaintExtraction()
        patched = current_extraction.model_copy(update=updated_fields)
        complaint_graph.update_state(config, {"extraction": patched})
        logger.info("chat: field correction applied: %s", list(updated_fields.keys()))

    return ChatResponse(reply=reply, updated_extraction=None, updated_fields=updated_fields or None)


def _answer_question(question: str, state: ComplaintState) -> tuple[str, dict]:
    """
    Answer a free-form question OR detect a field correction and return a patch.

    Returns (reply_text, updated_fields_dict).
    updated_fields_dict is empty {} when no correction was detected.

    How field-correction detection works:
      We ask the LLM to return a JSON object with two keys:
        "reply"          — the conversational response to show the user
        "updated_fields" — a dict of {field_name: new_value} for any fields
                           the user is explicitly correcting (empty dict if none)

      The LLM sees the current form values as context, so it knows which fields
      exist and what their current values are.  If the user says "sorry the
      batch number is BMX240602", the LLM returns:
        {"reply": "Got it. I have updated the Batch / Lot Number...",
         "updated_fields": {"batch_lot_number": "BMX240602"}}

      This is better than regex because natural language corrections can be
      phrased many ways ("actually it's", "correction:", "the batch is", etc.)
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq
    from app.config import get_settings
    import json as _json

    settings = get_settings()
    extraction = state.get("extraction") or ComplaintExtraction()

    # Build context from current form values
    context_lines = []
    for field_name, value in extraction.model_dump().items():
        if value is not None:
            context_lines.append(f"  {field_name}: {value}")
    if state.get("risk_justification"):
        context_lines.append(f"  risk_justification: {state['risk_justification']}")
    if state.get("root_cause_hypothesis"):
        context_lines.append(f"  root_cause_hypothesis: {state['root_cause_hypothesis']}")
    if state.get("capa_text"):
        context_lines.append(f"  capa_recommendation: {state['capa_text']}")
    context = "\n".join(context_lines) or "  (no complaint data loaded yet)"

    system = """\
You are PharmaIntel Copilot, an intelligent pharmaceutical QA assistant helping a user fill in a
complaint intake form.

You have two jobs:
1. Answer questions about the complaint data.
2. Detect when the user is correcting a specific form field value.

FIELD NAMES in the form (use these exact names in updated_fields):
  complaint_source, customer_name, product_name, product_strength_grade,
  batch_lot_number, manufacturing_date, expiry_date, quantity_affected,
  complaint_type, complaint_date, detailed_complaint_description,
  initial_severity, priority

RESPONSE FORMAT — always return a single JSON object:
{
  "reply": "<conversational response, 1-2 sentences>",
  "updated_fields": {}
}

If the user is correcting one or more fields, include them:
{
  "reply": "Got it. I have updated the Batch / Lot Number to \\"BMX240602\\" and the Affected Quantity to \\"48 capsules\\" in the form.",
  "updated_fields": {"batch_lot_number": "BMX240602", "quantity_affected": "48 capsules"}
}

Rules:
- Only include a field in updated_fields if the user is EXPLICITLY providing
  a new value for it.  Do not add fields for general questions.
- For the reply, confirm what you changed in plain English.
- For pure questions (no correction), return updated_fields as {}.
- Return ONLY the JSON — no markdown, no extra text.
"""

    human = f"Current form values:\n{context}\n\nUser message: {question}"

    try:
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.extraction_model,
            temperature=0.1,
            max_tokens=400,
        )
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        raw = response.content.strip()

        # Parse the JSON response
        try:
            parsed = _json.loads(raw)
            reply = parsed.get("reply", raw)
            updated_fields = parsed.get("updated_fields", {})
            if not isinstance(updated_fields, dict):
                updated_fields = {}
            return reply, updated_fields
        except _json.JSONDecodeError:
            # LLM returned prose instead of JSON — return it as-is, no field updates
            return raw, {}

    except Exception as exc:
        logger.warning("Q&A LLM call failed: %s", exc)
        return (
            "I'm having trouble reaching the AI service right now. Please try again.",
            {},
        )


# ── POST /api/complaints (persist confirmed record) ────────────────────────────

@router.post("", response_model=ConfirmComplaintResponse, status_code=201)
async def confirm_complaint(
    body: ConfirmComplaintRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Persist the user-confirmed (and possibly user-edited) complaint to Postgres.

    Flow:
      1. Read the graph checkpoint for body.session_id to get the AI analysis
         fields (risk_justification, root_cause_hypothesis, capa_text, etc.)
         that were generated during extraction but aren't part of the user-
         editable form payload.
      2. Merge body.extraction (user-confirmed field values) with those analysis
         fields to build the full ComplaintORM record.
      3. Generate the description embedding from the confirmed description text
         so this record is immediately searchable for duplicate detection.
      4. Insert and commit.  Return the real DB UUID.

    Why merge checkpoint + user payload (not just save the checkpoint)?
      The user may have edited fields on the form before clicking Submit —
      e.g. corrected the batch number or changed the severity.  body.extraction
      is the authoritative source of truth for the extractable fields.
      The checkpoint is the authoritative source for AI-generated analysis fields
      (CAPA, root cause, summary) that the user doesn't edit directly.
    """
    logger.info(
        "confirm_complaint: session=%s product=%s",
        body.session_id, body.extraction.product_name,
    )

    # ── 1. Read checkpoint for AI analysis fields ─────────────────────────────
    config = get_thread_config(str(body.session_id))
    checkpoint = complaint_graph.get_state(config)
    saved_state: dict = checkpoint.values if checkpoint and checkpoint.values else {}

    # ── 2. Generate description embedding ────────────────────────────────────
    # Run embed_text in a thread-pool executor so it doesn't block the event
    # loop — sentence-transformers is a synchronous CPU-bound operation.
    import asyncio
    description = body.extraction.detailed_complaint_description or ""
    loop = asyncio.get_event_loop()
    embedding: list[float] = await loop.run_in_executor(None, embed_text, description)

    # ── 3. Build the ORM record ───────────────────────────────────────────────
    extraction_dict = body.extraction.model_dump(mode="json")

    record = ComplaintORM(
        session_id=body.session_id,

        # Typed columns — duplicated from extraction_data for fast queries
        product_name=body.extraction.product_name,
        batch_lot_number=body.extraction.batch_lot_number,
        customer_name=body.extraction.customer_name,
        complaint_date=(
            str(body.extraction.complaint_date)
            if body.extraction.complaint_date else None
        ),
        initial_severity=(
            body.extraction.initial_severity.value
            if body.extraction.initial_severity else None
        ),
        priority=(
            body.extraction.priority.value
            if body.extraction.priority else None
        ),

        # Full extraction payload as JSONB
        extraction_data=extraction_dict,

        # AI analysis — from graph checkpoint (not user-editable)
        risk_justification=saved_state.get("risk_justification"),
        root_cause_hypothesis=saved_state.get("root_cause_hypothesis"),
        capa_text=saved_state.get("capa_text"),
        summary_text=saved_state.get("summary_text"),
        duplicate_ids=json.dumps(
            [str(uid) for uid in saved_state.get("duplicate_ids", [])]
        ),

        # Embedding for future duplicate detection
        description_embedding=embedding,

        # Audit
        raw_text=saved_state.get("raw_text"),
        is_confirmed=True,
    )

    # ── 4. Persist to Postgres & Fallback Local Ledger ───────────────────────
    complaint_id = uuid4()
    saved_to_db = False

    if db is not None:
        try:
            db.add(record)
            await asyncio.wait_for(db.flush(), timeout=20.0)
            complaint_id = record.id
            saved_to_db = True
            logger.info("confirm_complaint: successfully persisted to PostgreSQL as complaint_id=%s", complaint_id)
        except Exception as db_exc:
            logger.warning("PostgreSQL commit failed (%s), will rely on QMS ledger backup", db_exc)

    # Always write to persistent local QMS ledger as audit backup
    _save_to_local_ledger(complaint_id, body, saved_state, extraction_dict)
    logger.info("confirm_complaint: persisted to local QMS ledger as complaint_id=%s (db_saved=%s)", complaint_id, saved_to_db)
    return ConfirmComplaintResponse(complaint_id=complaint_id)


def _save_to_local_ledger(complaint_id: UUID, body: ConfirmComplaintRequest, saved_state: dict, extraction_dict: dict):
    """Write confirmed complaint to local JSON ledger when Postgres is unavailable."""
    import json as _json
    from pathlib import Path
    from datetime import datetime, timezone

    ledger_path = Path(__file__).resolve().parent.parent.parent / "data" / "qms_ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:
            data = {}

    data[str(complaint_id)] = {
        "id": str(complaint_id),
        "session_id": str(body.session_id),
        "product_name": body.extraction.product_name,
        "batch_lot_number": body.extraction.batch_lot_number,
        "customer_name": body.extraction.customer_name,
        "complaint_date": str(body.extraction.complaint_date) if body.extraction.complaint_date else None,
        "initial_severity": body.extraction.initial_severity.value if body.extraction.initial_severity else None,
        "priority": body.extraction.priority.value if body.extraction.priority else None,
        "extraction_data": extraction_dict,
        "risk_justification": saved_state.get("risk_justification"),
        "root_cause_hypothesis": saved_state.get("root_cause_hypothesis"),
        "capa_text": saved_state.get("capa_text"),
        "summary_text": saved_state.get("summary_text"),
        "raw_text": saved_state.get("raw_text"),
        "is_confirmed": True,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(ledger_path, "w", encoding="utf-8") as f:
        _json.dump(data, f, indent=2)


def _get_from_local_ledger(complaint_id: UUID):
    """Read complaint from local JSON ledger if Postgres is unavailable."""
    import json as _json
    from pathlib import Path

    ledger_path = Path(__file__).resolve().parent.parent.parent / "data" / "qms_ledger.json"
    if not ledger_path.exists():
        return None
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
            return data.get(str(complaint_id))
    except Exception:
        return None


# ── GET /api/complaints/{complaint_id} ────────────────────────────────────────

@router.get("/{complaint_id}", response_model=ComplaintDetailResponse)
async def get_complaint(
    complaint_id: UUID,
    db: AsyncSession | None = Depends(get_db),
):
    """
    Retrieve a previously saved complaint by its database UUID.
    """
    if db is not None:
        try:
            result = await db.execute(
                select(ComplaintORM).where(ComplaintORM.id == complaint_id)
            )
            row: ComplaintORM | None = result.scalar_one_or_none()
            if row is not None:
                complaint = Complaint(
                    id=row.id,
                    extraction=ComplaintExtraction.model_validate(row.extraction_data or {}),
                    analysis=ComplaintAnalysis(
                        risk_justification=row.risk_justification,
                        root_cause_hypothesis=row.root_cause_hypothesis,
                        capa_recommendation=row.capa_text,
                        summary=row.summary_text,
                    ),
                    raw_text=row.raw_text,
                )
                return ComplaintDetailResponse(complaint=complaint)
        except Exception:
            pass

    # Check local ledger
    entry = _get_from_local_ledger(complaint_id)
    if entry:
        complaint = Complaint(
            id=UUID(entry["id"]),
            extraction=ComplaintExtraction.model_validate(entry.get("extraction_data") or {}),
            analysis=ComplaintAnalysis(
                risk_justification=entry.get("risk_justification"),
                root_cause_hypothesis=entry.get("root_cause_hypothesis"),
                capa_recommendation=entry.get("capa_text"),
                summary=entry.get("summary_text"),
            ),
            raw_text=entry.get("raw_text"),
        )
        return ComplaintDetailResponse(complaint=complaint)

    raise HTTPException(
        status_code=404,
        detail=f"Complaint {complaint_id} not found.",
    )


# ── GET /api/complaints/ledger/all ──────────────────────────────────────────

@router.get("/ledger/all", summary="List all saved complaints from PostgreSQL Database")
async def list_all_ledger_complaints(
    db: AsyncSession | None = Depends(get_db),
):
    """Retrieve all committed complaints from the PostgreSQL database (complaints table)."""
    records = []
    source = "PostgreSQL Database (Neon Cloud - Table: complaints)"

    if db is not None:
        try:
            result = await db.execute(select(ComplaintORM).order_by(ComplaintORM.created_at.desc()))
            rows = result.scalars().all()
            for r in rows:
                records.append({
                    "id": str(r.id),
                    "session_id": str(r.session_id) if r.session_id else None,
                    "product_name": r.product_name,
                    "batch_lot_number": r.batch_lot_number,
                    "customer_name": r.customer_name,
                    "complaint_date": r.complaint_date,
                    "initial_severity": r.initial_severity,
                    "priority": r.priority,
                    "summary_text": r.summary_text,
                    "root_cause_hypothesis": r.root_cause_hypothesis,
                    "capa_text": r.capa_text,
                    "is_confirmed": r.is_confirmed,
                    "created_at": str(r.created_at),
                    "storage_backend": "PostgreSQL Database (Neon)",
                })
        except Exception as e:
            logger.warning("Could not fetch from PostgreSQL: %s", e)

    # Fallback if DB empty or offline
    if not records:
        source = "Fallback QMS Ledger (qms_ledger.json)"
        import json as _json
        from pathlib import Path
        ledger_path = Path(__file__).resolve().parent.parent.parent / "data" / "qms_ledger.json"
        if ledger_path.exists():
            try:
                with open(ledger_path, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                    records = list(data.values())
            except Exception:
                pass

    return {
        "total": len(records),
        "database_backend": source,
        "database_table": "complaints",
        "records": records,
    }



