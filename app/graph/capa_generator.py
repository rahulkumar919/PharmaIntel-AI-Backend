"""
graph/capa_generator.py — root-cause hypothesis + CAPA recommendation via LLM.

Why the heavy model here?
  CAPA generation requires genuine multi-step domain reasoning:
    1. Interpret the complaint type, severity, and description
    2. Hypothesise a root cause from GMP/ICH knowledge
    3. Propose specific corrective actions (stop the immediate problem)
    4. Propose preventive actions (stop recurrence)
  The lightweight extraction model handles pattern-matching well but produces
  vague CAPA text.  The heavier capa_model (openai/gpt-oss-120b) produces
  substantially more specific, actionable recommendations.

  This is the deliberate model-selection trade-off from the project spec:
  "pick [the heavier model] deliberately per node, don't use one model everywhere."

Output structure:
  We ask for a JSON object with two keys: root_cause_hypothesis and
  capa_recommendation.  Structured output keeps the node's state-writing
  logic simple — no regex parsing of freeform prose.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.config import get_settings
from app.graph.state import ComplaintState
from app.schemas.complaint import ComplaintExtraction

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

class CapaOutput(BaseModel):
    root_cause_hypothesis: str
    capa_recommendation: str


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior pharmaceutical quality assurance specialist with expertise in
GMP compliance, ICH Q10 quality systems, and CAPA management.

Given a complaint record, provide:
  1. root_cause_hypothesis — the most probable technical or process root cause
     based on GMP knowledge. Be specific: name the likely process step, equipment,
     or control that failed. Do NOT say "further investigation required" — give
     your best professional hypothesis.
  2. capa_recommendation — structured corrective AND preventive actions:
     Corrective: immediate actions to contain and resolve the current issue.
     Preventive: systemic changes to prevent recurrence (SOPs, controls, training).

Use pharma-specific language (reference ICH guidelines, GMP requirements, or
regulatory standards where relevant).

Return ONLY a JSON object with these exact two keys — no markdown, no extra text:
{
  "root_cause_hypothesis": "...",
  "capa_recommendation": "..."
}
"""

_HUMAN_TEMPLATE = """\
COMPLAINT RECORD
================
Product       : {product_name}
Strength/Grade: {product_strength_grade}
Batch/Lot     : {batch_lot_number}
Mfg Date      : {manufacturing_date}
Expiry        : {expiry_date}
Qty Affected  : {quantity_affected}
Customer      : {customer_name}
Source        : {complaint_source}
Type          : {complaint_type}
Severity      : {initial_severity}
Priority      : {priority}

Description:
{description}

Risk Justification (from risk_classification node):
{risk_justification}

Potential Duplicates (same issue reported before):
{duplicate_context}

Generate the root cause hypothesis and CAPA recommendation JSON:
"""


# ── Public function ───────────────────────────────────────────────────────────

def generate_capa(state: ComplaintState) -> CapaOutput:
    """
    Build context from the full state and call the capa_model for CAPA output.

    We pass the entire complaint record including risk_justification and
    duplicate context so the model can reason about the full picture — not
    just the description in isolation.

    Falls back to templated text on any error so the graph never halts here.
    """
    settings = get_settings()
    extraction: ComplaintExtraction = state.get("extraction") or ComplaintExtraction()

    def _val(v) -> str:
        """Render enum values and None cleanly."""
        if v is None:
            return "Not specified"
        return v.value if hasattr(v, "value") else str(v)

    # Build duplicate context string
    dup_ids: list = state.get("duplicate_ids") or []
    dup_scores: list = state.get("duplicate_scores") or []
    if dup_ids:
        dup_lines = [
            f"  - Complaint ID {uid} (similarity {score:.2f})"
            for uid, score in zip(dup_ids, dup_scores)
        ]
        duplicate_context = "\n".join(dup_lines)
    else:
        duplicate_context = "No similar complaints found in the database."

    human_content = _HUMAN_TEMPLATE.format(
        product_name=_val(extraction.product_name),
        product_strength_grade=_val(extraction.product_strength_grade),
        batch_lot_number=_val(extraction.batch_lot_number),
        manufacturing_date=_val(extraction.manufacturing_date),
        expiry_date=_val(extraction.expiry_date),
        quantity_affected=_val(extraction.quantity_affected),
        customer_name=_val(extraction.customer_name),
        complaint_source=_val(extraction.complaint_source),
        complaint_type=_val(extraction.complaint_type),
        initial_severity=_val(extraction.initial_severity),
        priority=_val(extraction.priority),
        description=extraction.detailed_complaint_description or "Not provided.",
        risk_justification=state.get("risk_justification") or "Not available.",
        duplicate_context=duplicate_context,
    )

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.capa_model,   # heavier model — deliberate choice
        temperature=0.1,  # slight warmth for creative but grounded recommendations
        max_tokens=1024,
    )

    structured_llm = llm.with_structured_output(
        CapaOutput,
        method="json_mode",
        include_raw=True,
    )

    try:
        result = structured_llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ])

        if result.get("parsed") is not None:
            logger.info("generate_capa: CAPA generated successfully")
            return result["parsed"]

        logger.warning("generate_capa: structured output parse failed")

    except Exception as exc:
        logger.warning("generate_capa: LLM call failed (%s), using fallback", exc)

    # ── Fallback ──────────────────────────────────────────────────────────────
    product = _val(extraction.product_name)
    batch = _val(extraction.batch_lot_number)
    ctype = _val(extraction.complaint_type)
    return CapaOutput(
        root_cause_hypothesis=(
            f"Root cause analysis pending for {ctype} complaint regarding "
            f"{product} batch {batch}. AI CAPA generation unavailable — "
            "manual investigation required."
        ),
        capa_recommendation=(
            f"Corrective: Quarantine batch {batch} pending investigation. "
            "Review relevant batch records and in-process data. "
            "Preventive: Conduct root cause analysis per SOP-QA-001 and "
            "implement identified preventive measures within 30 days."
        ),
    )
