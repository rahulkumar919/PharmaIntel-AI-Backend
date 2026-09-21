"""
graph/extractor.py — ChatGroq structured-output extraction + JSON-repair retry.

Why this lives in its own module:
  The extraction logic has two distinct concerns — (1) calling the LLM and
  (2) validating / repairing its output.  Keeping both here, rather than
  inline in nodes.py, means the node stays a thin orchestrator and this
  module can be unit-tested with a mock LLM without touching the graph.

High-level flow
───────────────
  1. Build a system + human prompt pair from the raw complaint text.
  2. Call ChatGroq(gemma2-9b-it).with_structured_output(ComplaintExtraction,
       method="json_mode") — this tells Groq to return strict JSON matching
       the Pydantic schema; LangChain validates the response automatically.
  3. If validation succeeds  → return (extraction, attempts=1).
  4. If Pydantic raises ValidationError or the LLM returns malformed JSON:
       a. Log the bad output verbatim so we can audit repair-rate later.
       b. Send a SECOND prompt that includes the broken JSON and explicit
          fix instructions (the "repair sub-step").
       c. If attempt 2 also fails → raise ExtractionError with a clear message
          so the caller can surface it to the user.

Why with_structured_output (not a plain string call + manual json.loads)?
  with_structured_output() passes the Pydantic schema as a JSON Schema to
  Groq's function-calling / tool-use API.  The model is constrained at the
  token level to produce valid JSON matching our schema — it's not just
  prompting; the API enforces the structure.  This dramatically reduces
  parse failures compared to asking the model to "return JSON" in a
  freeform string.

  ASSUMPTION: langchain-groq 0.1.x exposes with_structured_output via
  ChatGroq.  If a future version changes this API, the safest fallback is
  ChatGroq(...) | PydanticOutputParser(pydantic_object=ComplaintExtraction).

Why gemma2-9b-it and not llama-3.3-70b here?
  Extraction is a pattern-matching task: find field X in the text, map it
  to enum Y.  gemma2-9b-it handles this well and is ~3× faster and cheaper
  than the 70B model.  We reserve llama-3.3-70b for CAPA where multi-step
  pharma domain reasoning matters more than speed.
"""

from __future__ import annotations

import json
import logging
from typing import Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.config import get_settings
from app.schemas.complaint import ComplaintExtraction

logger = logging.getLogger(__name__)


# ── Custom exception ──────────────────────────────────────────────────────────

class ExtractionError(RuntimeError):
    """
    Raised when both the initial extraction AND the repair attempt fail.
    Wrapping in a domain-specific exception (rather than letting a raw
    ValidationError bubble up) gives the route handler a clean signal to
    return a user-friendly 422 response instead of an opaque 500.
    """


# ── Prompt templates ──────────────────────────────────────────────────────────

# System prompt — sets the LLM's role and output contract once.
# Written as a module-level constant so it is easy to tune without hunting
# through function bodies.
_SYSTEM_PROMPT = """\
You are a pharmaceutical quality-management specialist extracting structured \
data from customer complaint documents (API & FDF manufacturing context).

TASK
Read the complaint text and extract exactly the fields in the schema below. \
Return ONLY a single valid JSON object matching that schema — no prose, no \
markdown code fences, no commentary before or after it.

OUTPUT SCHEMA (return every key, even when the value is null)
{
  "complaint_source": string | null,        // enum, see below
  "customer_name": string | null,
  "product_name": string | null,
  "product_strength_grade": string | null,
  "batch_lot_number": string | null,
  "manufacturing_date": string | null,      // ISO 8601: YYYY-MM-DD
  "expiry_date": string | null,             // ISO 8601: YYYY-MM-DD
  "quantity_affected": string | null,       // number + unit, e.g. "50 units", "2 kg"
  "complaint_type": string | null,          // enum, see below
  "complaint_date": string | null,          // ISO 8601: YYYY-MM-DD
  "detailed_complaint_description": string | null,
  "initial_severity": string | null,        // enum, see below
  "priority": string | null                 // enum, see below
}

ENUM CONSTRAINTS — use one of these exact strings, or null. Never invent a
new value, and never guess a synonym into an enum slot.
  complaint_source : "Email" | "Phone" | "Portal" | "Distributor" | "Regulatory"
  complaint_type   : "Quality Defect" | "Packaging" | "Labeling" | "Adverse Event" | "Delivery/Documentation"
  initial_severity : "Critical" | "Major" | "Minor"
  priority         : "High" | "Medium" | "Low"

EXTRACTION RULES
1. Extract only what is explicitly stated or unambiguously implied. If a
   field is not mentioned anywhere, set it to null — do not infer or
   estimate it.
2. Dates: convert any format you see (e.g. "12th March 2025", "03/12/25")
   into YYYY-MM-DD. If the year is ambiguous or missing, set the field to
   null rather than guessing the year.
3. quantity_affected must always include a unit. If a number is given with
   no unit and none can be reasonably inferred from context (e.g. tablets,
   kg, units, vials), set it to null rather than assuming a unit.
4. initial_severity and priority are your professional QA judgment based on
   the complaint content (e.g. an adverse event or sterility failure is
   Critical/High; a labeling typo with no safety impact is Minor/Low) —
   these two fields should rarely be null unless the text gives you nothing
   at all to assess severity from.
5. detailed_complaint_description should be a faithful, condensed
   restatement of the customer's issue in 1-3 sentences — do not copy the
   entire source text verbatim, and do not add interpretation that isn't in
   the source.
6. If the text contains multiple products, batches, or complaints, extract
   only the primary/first complaint described and ignore the rest.
7. Never fabricate a customer name, batch number, or date. A plausible-
   sounding value is not the same as a stated value.

EXAMPLE

Input complaint text:
"Hi team, this is Raj Malhotra from MedSupply Distributors. We received a
shipment of Amoxicillin 500mg capsules, batch AMX-2024-0517, mfg date
14/05/2024, and noticed 3 blister packs (about 30 tablets total) with
discoloration on the capsules. This was flagged during routine incoming
inspection on 2nd June 2024, not from a patient. No adverse events."

Expected output:
{
  "complaint_source": "Distributor",
  "customer_name": "Raj Malhotra",
  "product_name": "Amoxicillin",
  "product_strength_grade": "500mg",
  "batch_lot_number": "AMX-2024-0517",
  "manufacturing_date": "2024-05-14",
  "expiry_date": null,
  "quantity_affected": "30 tablets",
  "complaint_type": "Quality Defect",
  "complaint_date": "2024-06-02",
  "detailed_complaint_description": "Discoloration observed on capsules in 3 blister packs during routine incoming inspection; no patient exposure or adverse event reported.",
  "initial_severity": "Major",
  "priority": "Medium"
}

Now extract from the complaint text provided by the user, following the
same schema and rules exactly.
"""

# Human prompt — the complaint text is injected here at call time.
# Keeping this as a separate template (rather than appending to the system
# prompt) matches the recommended ChatGroq message structure: system sets
# the role/contract, human provides the input data.
_HUMAN_PROMPT_TEMPLATE = """\
Extract structured fields from the following pharmaceutical customer complaint.
Return ONLY the JSON object — nothing before or after it.

--- COMPLAINT TEXT BEGIN ---
{complaint_text}
--- COMPLAINT TEXT END ---
"""

# Repair prompt — sent as a second Human message when attempt 1 produces
# malformed JSON.  Including the broken output lets the model see exactly
# what went wrong and correct it rather than starting from scratch.
_REPAIR_PROMPT_TEMPLATE = """\
Your previous response could not be parsed as valid JSON matching the required schema.

Broken output you returned:
---
{broken_output}
---

Validation / parse error:
{validation_error}

Fix ONLY what is wrong.  Return the corrected JSON object and NOTHING ELSE:
- No markdown code fences (no ```json)
- No explanatory text before or after the JSON
- All field names must be exactly as in the schema (snake_case)
- Enum values must match exactly (case-sensitive, see schema in your system prompt)
- Dates must be YYYY-MM-DD or null
- Do not add or remove any keys

Corrected JSON:
"""


# ── Main public function ──────────────────────────────────────────────────────

def run_extraction(raw_text: str) -> Tuple[ComplaintExtraction, int]:
    """
    Call the LLM to extract structured fields from complaint text.

    Returns
    -------
    (ComplaintExtraction, attempts)
        attempts = 1 if the first call succeeded, 2 if the repair step was needed.

    Raises
    ------
    ExtractionError
        If both attempts fail.  The caller (the node) should write
        state["error"] and allow the graph to surface a friendly message.
    """
    settings = get_settings()

    # Build the LLM client.
    # temperature=0 is deliberate: extraction is deterministic pattern-matching,
    # not creative generation.  Any temperature > 0 introduces unnecessary
    # variance in field values.
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.extraction_model,   # "gemma2-9b-it"
        temperature=0,
        # max_tokens capped at 1024 — extraction output is a compact JSON object;
        # we don't want the model to pad it with prose that would break parsing.
        max_tokens=1024,
    )

    # Bind the Pydantic schema to the LLM.
    # We use method="json_mode" here rather than the default tool-calling approach
    # because the openai/gpt-oss-20b model on Groq returns JSON directly rather
    # than via a tool-call response — using tool-calling with this model causes a
    # 400 "Tool choice required but model did not call a tool" error.
    # json_mode tells LangChain to parse the raw string response as JSON and
    # validate it against the Pydantic schema, which is equivalent for our purposes.
    # include_raw=True gives us both the parsed object AND the raw string so we
    # can build the repair prompt if parsing fails.
    structured_llm = llm.with_structured_output(
        ComplaintExtraction,
        method="json_mode",
        include_raw=True,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_HUMAN_PROMPT_TEMPLATE.format(complaint_text=raw_text)),
    ]

    # ── Attempt 1 ─────────────────────────────────────────────────────────────
    result = _invoke_structured(structured_llm, messages, attempt=1)

    if result["parsed"] is not None:
        logger.info("extract_fields: attempt 1 succeeded")
        return result["parsed"], 1

    # ── Repair attempt ────────────────────────────────────────────────────────
    # Attempt 1 produced malformed JSON or a schema mismatch.
    # We capture the broken output and validation error, then ask the model
    # to fix its own response.  This "self-repair" pattern works well with
    # instruction-tuned models that can reason about their own outputs.
    broken_output = _extract_raw_content(result)
    parsing_error = str(result.get("parsing_error", "Unknown parsing error"))

    logger.warning(
        "extract_fields: attempt 1 failed — broken_output=%r error=%s",
        broken_output[:200],   # truncate for log readability
        parsing_error,
    )

    # Append the broken output + repair instructions as a new Human message.
    # We keep the original system + human messages so the model still has
    # the complaint text in context.
    repair_messages = messages + [
        HumanMessage(
            content=_REPAIR_PROMPT_TEMPLATE.format(
                broken_output=broken_output,
                validation_error=parsing_error,
            )
        )
    ]

    result2 = _invoke_structured(structured_llm, repair_messages, attempt=2)

    if result2["parsed"] is not None:
        logger.info("extract_fields: repair attempt 2 succeeded")
        return result2["parsed"], 2

    # Both attempts failed — surface a clear error to the caller
    final_error = str(result2.get("parsing_error", "Unknown parsing error"))
    raise ExtractionError(
        f"Field extraction failed after 2 attempts. "
        f"Last validation error: {final_error}. "
        f"Last raw output: {_extract_raw_content(result2)[:300]}"
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _invoke_structured(structured_llm, messages: list, attempt: int) -> dict:
    """
    Call the structured LLM and return the raw include_raw dict.

    Retries up to 3 times on rate-limit or transient network errors using
    exponential back-off (1s, 2s, 4s).  Each retry is logged so the
    production log drain shows exactly why a call was retried.

    Catches network/API errors and re-raises them as ExtractionError so the
    node doesn't need to handle Groq-specific exception types.
    """
    import httpx  # local import — only needed when network errors occur

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_with_retry():
        return structured_llm.invoke(messages)

    try:
        return _call_with_retry()
    except Exception as exc:
        # Re-raise with context so the caller can distinguish LLM errors from
        # parse errors.
        raise ExtractionError(
            f"LLM call failed on attempt {attempt} after retries: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _extract_raw_content(result: dict) -> str:
    """
    Pull the raw string content out of the include_raw result dict.

    with_structured_output(include_raw=True) returns:
      {"raw": AIMessage, "parsed": <model> | None, "parsing_error": <str> | None}

    The AIMessage.content may itself be a JSON string or a list of content
    blocks (for models that return tool-call responses).  We normalise both
    to a plain string for logging and the repair prompt.
    """
    raw_msg = result.get("raw")
    if raw_msg is None:
        return "<no raw output>"

    content = raw_msg.content
    if isinstance(content, list):
        # Tool-call style: content is a list of dicts with "text" or "json" keys
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or json.dumps(block.get("json", {})))
            else:
                parts.append(str(block))
        return "\n".join(parts)

    return str(content)
