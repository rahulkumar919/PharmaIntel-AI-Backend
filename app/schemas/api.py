"""
schemas/api.py — HTTP-layer request and response models.

Why separate from complaint.py?
  Domain models (complaint.py) describe the business data; API models describe
  what the HTTP boundary accepts and returns.  Keeping them separate means we
  can evolve the internal domain without breaking the API contract, and we can
  add HTTP-specific fields (e.g. session_id, status) without polluting the
  domain model.  This is the classic "DTO / anti-corruption layer" pattern.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.complaint import Complaint, ComplaintExtraction


# ── POST /api/complaints/extract ──────────────────────────────────────────────

class ExtractTextRequest(BaseModel):
    """
    Used when the caller pastes raw text instead of uploading a file.
    The endpoint also accepts multipart/form-data for file uploads;
    that path uses FastAPI's UploadFile directly, not this model.
    """
    raw_text: str = Field(..., min_length=10, description="Raw complaint text pasted by the user.")


class NodeProgressEvent(BaseModel):
    """
    Shape of each Server-Sent Event chunk emitted while the graph runs.

    event values (the SSE `event:` field):
      node_progress  — a node just started or completed
      interrupt      — graph paused at ask_user; clarifying_question is set
      done           — graph finished; payload contains the full ExtractResponse
      error          — unrecoverable graph error; detail contains the message

    The frontend state machine:
      node_progress → update progress bar
      interrupt     → show clarifying_question in chat box, wait for user reply
      done          → hydrate the form with the complaint payload
      error         → show error banner
    """
    event: str = "node_progress"     # SSE event: field
    node: str = ""                   # node name, e.g. "extract_fields"
    status: str = "completed"        # "started" | "completed" | "error"
    detail: Optional[str] = None     # human-readable detail or JSON payload


# Node display metadata — used by the frontend to show readable labels
# and calculate progress percentage.
# Order matches the graph topology so the progress bar advances linearly.
NODE_SEQUENCE: list[str] = [
    "parse_document",
    "extract_fields",
    "completeness_check",
    "risk_classification",
    "duplicate_detection",
    "capa_node",
    "summary_node",
]

NODE_LABELS: dict[str, str] = {
    "parse_document":      "Parsing document",
    "extract_fields":      "Extracting fields",
    "completeness_check":  "Checking completeness",
    "ask_user":            "Waiting for clarification",
    "risk_classification": "Classifying risk",
    "duplicate_detection": "Checking for duplicates",
    "capa_node":           "Generating CAPA",
    "summary_node":        "Summarising",
}


class ExtractResponse(BaseModel):
    """
    Final response returned at the end of the SSE stream (event: "done").
    Contains the fully populated complaint so the frontend can hydrate the form.
    """
    session_id: UUID = Field(..., description="Opaque ID that links this extraction to the chat endpoint.")
    complaint: Complaint
    clarifying_question: Optional[str] = Field(
        None,
        description="Set when the graph hit the ask_user node and needs more info. "
                    "The frontend should display this in the chat box.",
    )


# ── POST /api/complaints/chat ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Payload for the conversational follow-up endpoint.
    session_id ties the message to a previously extracted complaint.
    """
    session_id: UUID
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    # If the graph produced an updated extraction after the user answered
    # the clarifying question, the frontend should re-hydrate the form.
    updated_extraction: Optional[ComplaintExtraction] = None
    # Sparse patch dict of field_name → new_value when the user corrects
    # specific fields via chat (e.g. "sorry the batch number is BMX240602").
    # The frontend applies only these keys, leaving other fields untouched.
    updated_fields: Optional[dict] = None


# ── POST /api/complaints (persist confirmed record) ───────────────────────────

class ConfirmComplaintRequest(BaseModel):
    """
    The user has reviewed and (optionally edited) the auto-populated form.
    We receive the full extraction payload — whatever the user confirmed —
    plus the session_id so we can merge in the AI analysis fields from state.
    """
    session_id: UUID
    extraction: ComplaintExtraction       # possibly user-edited values


class ConfirmComplaintResponse(BaseModel):
    complaint_id: UUID
    message: str = "Complaint successfully recorded."


# ── GET /api/complaints/{id} ──────────────────────────────────────────────────

class ComplaintDetailResponse(BaseModel):
    """Full complaint record returned for viewing a previously saved complaint."""
    complaint: Complaint
