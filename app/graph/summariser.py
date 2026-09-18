"""
graph/summariser.py — 2-3 sentence summary for the chat sidebar.

Why summarise at all?
  The chat sidebar needs a human-readable digest so a reviewer can understand
  the complaint at a glance without reading the entire form.  A well-written
  summary also makes the chat Q&A more useful — the LLM in _answer_question()
  can reference the summary to give contextual answers.

Model: extraction_model (lightweight).
  Summarisation is a short-form generation task, not deep reasoning.
  The lightweight model is perfectly capable and much faster here.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import get_settings
from app.graph.state import ComplaintState
from app.schemas.complaint import ComplaintExtraction

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a pharmaceutical QA assistant writing a concise complaint summary for a
case management dashboard.  Write exactly 2-3 sentences covering:
  1. What the complaint is (product, batch, issue type)
  2. The assessed severity/priority and key risk
  3. The primary recommended action

Write in third person, past tense.  Be specific — include product name, batch
number, severity level, and at least one concrete recommended action.
Do NOT start with "This complaint" or "The complaint".
Output only the summary text — no labels, no JSON, no markdown.
"""

_HUMAN_TEMPLATE = """\
Product       : {product_name} {product_strength_grade}
Batch         : {batch_lot_number}
Type          : {complaint_type}
Severity      : {initial_severity} / Priority: {priority}
Description   : {description}
Root cause    : {root_cause}
CAPA          : {capa}

Write the 2-3 sentence summary now:
"""


def generate_summary(state: ComplaintState) -> str:
    """
    Generate a 2-3 sentence human-readable summary of the complaint.

    Falls back to a template string if the LLM call fails — the summary is
    cosmetic (for the sidebar) and should never block the pipeline.
    """
    settings = get_settings()
    extraction: ComplaintExtraction = state.get("extraction") or ComplaintExtraction()

    def _val(v, default="Not specified") -> str:
        if v is None:
            return default
        return v.value if hasattr(v, "value") else str(v)

    human_content = _HUMAN_TEMPLATE.format(
        product_name=_val(extraction.product_name),
        product_strength_grade=_val(extraction.product_strength_grade, ""),
        batch_lot_number=_val(extraction.batch_lot_number),
        complaint_type=_val(extraction.complaint_type),
        initial_severity=_val(extraction.initial_severity),
        priority=_val(extraction.priority),
        description=(extraction.detailed_complaint_description or "Not provided.")[:400],
        root_cause=(state.get("root_cause_hypothesis") or "Pending investigation.")[:300],
        capa=(state.get("capa_text") or "Pending.")[:300],
    )

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.extraction_model,
        temperature=0.2,    # slight warmth produces more natural-sounding prose
        max_tokens=200,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ])
        summary = response.content.strip()
        if summary and len(summary) > 20:
            logger.info("generate_summary: summary generated (%d chars)", len(summary))
            return summary
        logger.warning("generate_summary: LLM returned suspiciously short summary")
    except Exception as exc:
        logger.warning("generate_summary: LLM call failed (%s), using fallback", exc)

    # ── Template fallback ─────────────────────────────────────────────────────
    return (
        f"A {_val(extraction.complaint_type, 'quality')} complaint was received "
        f"regarding {_val(extraction.product_name, 'an unspecified product')} "
        f"(Batch {_val(extraction.batch_lot_number, 'unknown')}). "
        f"Severity assessed as {_val(extraction.initial_severity, 'pending')} / "
        f"Priority {_val(extraction.priority, 'pending')}. "
        f"CAPA initiated per standard QMS procedure."
    )
