"""
graph/risk_classifier.py — LLM helper for severity + priority scoring.

Why its own module?
  Same pattern as extractor.py and ask_user_logic.py: the node function in
  nodes.py stays a thin state-reader/writer; all LLM logic lives here and
  can be unit-tested with a mock LLM independently of the graph.

Model choice: extraction_model (openai/gpt-oss-20b)
  Risk classification is a structured labelling task — read the description,
  assign one of three severity levels and one of three priority levels, and
  write a short justification.  This is the same kind of pattern-matching
  that extraction does, so the lightweight model handles it well.
  We deliberately reserve the capa_model for multi-step reasoning tasks.

Output format:
  We ask the LLM to return a small JSON object with three keys:
    initial_severity, priority, justification
  We parse it with Pydantic and fall back to safe defaults on failure.
  Using json_mode here keeps it consistent with extract_fields.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.schemas.complaint import ComplaintExtraction, Priority, Severity

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

class RiskOutput(BaseModel):
    """
    The structured JSON the LLM must return.
    Kept separate from ComplaintExtraction so we can validate it independently
    and write only severity/priority back to the extraction object.
    """
    initial_severity: Severity
    priority: Priority
    justification: str


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a pharmaceutical QA risk assessor applying ICH Q10 and GMP severity
criteria to customer complaints.

Given the complaint details, assess:
  initial_severity : Critical | Major | Minor
  priority         : High | Medium | Low

Severity guidelines:
  Critical — direct patient safety risk, sterility failure, wrong API, adverse event,
             or regulatory action required. Immediate escalation.
  Major    — significant quality defect (packaging, labeling, potency deviation) with
             no confirmed patient harm but potential risk if not contained.
  Minor    — cosmetic, documentation, or administrative issue with no quality or
             safety impact.

Priority guidelines:
  High     — batch still in distribution / patient exposure possible / regulatory deadline.
  Medium   — batch contained or recalled, investigation ongoing.
  Low      — issue resolved or no distribution risk.

Return ONLY a JSON object with exactly these three keys — no prose, no markdown:
{
  "initial_severity": "<Critical|Major|Minor>",
  "priority": "<High|Medium|Low>",
  "justification": "<1-2 sentence explanation>"
}
"""

_HUMAN_TEMPLATE = """\
Complaint details:
  Product      : {product_name}
  Batch        : {batch_lot_number}
  Complaint type: {complaint_type}
  Quantity affected: {quantity_affected}
  Customer     : {customer_name}

Description:
{description}

Return the risk assessment JSON:
"""


# ── Public function ───────────────────────────────────────────────────────────

def classify_risk(extraction: ComplaintExtraction) -> RiskOutput:
    """
    Call the LLM to assign severity + priority and produce a justification.

    Falls back to Major/High on any LLM or parse error — deliberately
    conservative (erring toward over-reaction in a pharma safety context).

    Returns
    -------
    RiskOutput with initial_severity, priority, justification
    """
    settings = get_settings()

    human_content = _HUMAN_TEMPLATE.format(
        product_name=extraction.product_name or "Unknown",
        batch_lot_number=extraction.batch_lot_number or "Unknown",
        complaint_type=(extraction.complaint_type.value
                        if extraction.complaint_type else "Unknown"),
        quantity_affected=extraction.quantity_affected or "Unknown",
        customer_name=extraction.customer_name or "Unknown",
        description=(extraction.detailed_complaint_description or
                     "No description provided."),
    )

    # If the LLM already gave us severity/priority during extraction, we still
    # re-classify here because the extraction model was making a quick guess
    # during field extraction.  The dedicated risk prompt produces better-
    # reasoned, more consistent results.

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.extraction_model,
        temperature=0,          # deterministic — same input must yield same severity
        max_tokens=256,
    )

    structured_llm = llm.with_structured_output(
        RiskOutput,
        method="json_mode",
        include_raw=True,
    )

    try:
        result = structured_llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ])

        if result.get("parsed") is not None:
            logger.info(
                "classify_risk: severity=%s priority=%s",
                result["parsed"].initial_severity,
                result["parsed"].priority,
            )
            return result["parsed"]

        # Parsing failed — log and fall back
        logger.warning(
            "classify_risk: structured output parse failed, raw=%r",
            _get_content(result),
        )

    except Exception as exc:
        logger.warning("classify_risk: LLM call failed (%s), using fallback", exc)

    # ── Conservative fallback ─────────────────────────────────────────────────
    # In pharma, under-reporting severity is more dangerous than over-reporting.
    # Default to Major/High so nothing slips through unreviewed.
    logger.warning("classify_risk: using conservative fallback Major/High")
    return RiskOutput(
        initial_severity=Severity.MAJOR,
        priority=Priority.HIGH,
        justification=(
            "Risk assessment could not be completed by the AI. "
            "Defaulted to Major/High — please review manually."
        ),
    )


def _get_content(result: dict) -> str:
    """Extract the raw content string from an include_raw result dict."""
    raw = result.get("raw")
    if raw is None:
        return "<no raw>"
    content = raw.content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content)[:300]
