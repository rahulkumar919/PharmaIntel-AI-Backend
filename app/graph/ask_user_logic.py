"""
graph/ask_user_logic.py — LLM helper that generates one targeted clarifying question.

Why a separate module (not inline in the node)?
  The same pattern as extractor.py: the node stays a thin state-manager while
  the LLM orchestration lives here where it can be unit-tested independently.

Design decision — ONE question, not a list:
  Asking the user for multiple missing fields at once causes form fatigue and
  makes the conversation feel like a bureaucratic checklist.  We ask for the
  single most important missing field (already ranked by _FIELD_PRIORITY in
  nodes.py).  If the graph loops back after the answer, completeness_check
  re-scores and ask_user picks the next gap — so multi-turn clarification
  is naturally incremental.

Why use the LLM here at all (instead of a template string)?
  A template like "Please provide the batch_lot_number" is robotic and
  doesn't carry domain context.  The LLM produces a question that references
  the product name and complaint type already in state, making it feel like
  a knowledgeable QA reviewer is asking — not a form validator.

Model choice: extraction_model (openai/gpt-oss-20b).
  Question generation is a short structured task, not heavy reasoning.
  Using the same lightweight model as extraction keeps latency low.
  We deliberately do NOT use the capa_model here — that's reserved for the
  multi-step reasoning in the CAPA node.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import get_settings
from app.graph.state import ComplaintState

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a pharmaceutical QA specialist conducting a structured complaint intake.
Your task is to ask ONE precise clarifying question to collect a single missing
piece of information needed to process the complaint.

Rules:
- Ask about ONLY the one field specified by the caller.
- Reference the product name, batch number, or complaint type if already known,
  to make the question feel contextual and specific rather than generic.
- Write in a professional but conversational tone — this is a chat message,
  not a form label.
- The question must be a single sentence ending in a question mark.
- Do NOT ask about multiple fields.
- Do NOT explain why you need the information.
- Do NOT add greetings, sign-offs, or filler phrases.
"""

_HUMAN_PROMPT_TEMPLATE = """\
Complaint context known so far:
  Product     : {product_name}
  Batch       : {batch_lot_number}
  Type        : {complaint_type}
  Customer    : {customer_name}
  Description : {description_snippet}

Missing field to ask about: {missing_field}
Field description          : {field_description}

Write the clarifying question now:
"""

# Human-readable descriptions for each field name — injected into the prompt
# so the LLM understands what "batch_lot_number" means without guessing.
_FIELD_DESCRIPTIONS: dict[str, str] = {
    "complaint_source":               "the channel through which the complaint arrived (Email, Phone, Portal, Distributor, or Regulatory)",
    "customer_name":                  "the full name of the customer or organisation raising the complaint",
    "product_name":                   "the commercial or INN name of the pharmaceutical product involved",
    "product_strength_grade":         "the strength (e.g. 500 mg) or grade (e.g. USP) of the product",
    "batch_lot_number":               "the manufacturer's batch or lot number printed on the product label",
    "manufacturing_date":             "the date of manufacture as printed on the label (YYYY-MM-DD)",
    "expiry_date":                    "the expiry or best-before date of the batch (YYYY-MM-DD)",
    "quantity_affected":              "the quantity of product involved in the complaint, including units (e.g. 50 tablets, 2 kg)",
    "complaint_type":                 "the category of the complaint: Quality Defect, Packaging, Labeling, Adverse Event, or Delivery/Documentation",
    "complaint_date":                 "the date the complaint was raised or first received (YYYY-MM-DD)",
    "detailed_complaint_description": "a clear description of the issue as reported by the customer",
}


# ── Public function ───────────────────────────────────────────────────────────

def generate_clarifying_question(state: ComplaintState, missing_field: str) -> str:
    """
    Call the LLM to produce one targeted question for the given missing field.

    Falls back to a deterministic template if the LLM call fails — we never
    want the graph to crash just because question generation had an API hiccup.
    The fallback is polished enough to be presented to the user.

    Parameters
    ----------
    state         : current ComplaintState (used to inject known context)
    missing_field : the single field name to ask about (from _FIELD_PRIORITY)

    Returns
    -------
    str — a single sentence question ready for display in the chat UI
    """
    settings = get_settings()
    extraction = state.get("extraction")

    # Pull known values from extraction for context injection
    product_name    = getattr(extraction, "product_name", None)    or "unknown product"
    batch           = getattr(extraction, "batch_lot_number", None) or "unknown batch"
    complaint_type  = getattr(extraction, "complaint_type", None)
    customer_name   = getattr(extraction, "customer_name", None)   or "the customer"
    description     = getattr(extraction, "detailed_complaint_description", None) or ""

    # Truncate description to 120 chars so the prompt stays compact
    description_snippet = (description[:120] + "…") if len(description) > 120 else description

    field_description = _FIELD_DESCRIPTIONS.get(
        missing_field,
        missing_field.replace("_", " "),  # graceful fallback for unexpected keys
    )

    human_prompt = _HUMAN_PROMPT_TEMPLATE.format(
        product_name=product_name,
        batch_lot_number=batch,
        complaint_type=complaint_type.value if complaint_type else "unknown",
        customer_name=customer_name,
        description_snippet=description_snippet or "(not yet provided)",
        missing_field=missing_field.replace("_", " "),
        field_description=field_description,
    )

    try:
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.extraction_model,   # fast model — short generation task
            temperature=0.3,   # slight warmth for natural phrasing vs. robotic template
            max_tokens=128,    # questions are short; cap tokens to avoid padding
        )
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ])
        question = response.content.strip()

        # Sanity-check: if the model returned something very long or empty,
        # fall back to the template rather than showing garbage to the user.
        if not question or len(question) > 400:
            raise ValueError(f"LLM returned unexpected question length: {len(question)}")

        logger.info("generate_clarifying_question: LLM question for '%s': %s", missing_field, question)
        return question

    except Exception as exc:
        logger.warning(
            "generate_clarifying_question: LLM failed (%s), using template fallback",
            exc,
        )
        return _template_fallback(missing_field, product_name, batch, field_description)


def _template_fallback(
    missing_field: str,
    product_name: str,
    batch: str,
    field_description: str,
) -> str:
    """
    Deterministic fallback question used when the LLM call fails.

    Written to be coherent and specific enough to display in production —
    not just a placeholder.  We reference product and batch so the user
    knows exactly which complaint we're asking about.
    """
    return (
        f"Regarding the complaint for {product_name} (batch {batch}), "
        f"could you please provide {field_description}?"
    )
