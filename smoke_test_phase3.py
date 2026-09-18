"""
smoke_test_phase3.py — verifies Phase 3: completeness_check, interrupt, resume, Q&A.

Run from backend/:
    python smoke_test_phase3.py

Three test suites:
  1. completeness_check unit test   — pure logic, no LLM
  2. Interrupt detection            — graph pauses when fields are missing
  3. Resume + Q&A round-trip        — inject answer, graph completes, Q&A works

Exit 0 = all passed.  Exit 1 = at least one failure.
"""
from __future__ import annotations
import sys, logging, os
from pathlib import Path

# ── Fix Windows cp1252 console encoding ──────────────────────────────────────
# Box-drawing characters (─, ═) used in section headers crash on Windows
# consoles that default to cp1252.  Setting UTF-8 here before any print()
# call fixes it without touching the system locale.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(level=logging.WARNING)   # suppress info noise during tests

# ── helpers ───────────────────────────────────────────────────────────────────
PASS, FAIL = "✓", "✗"
_results: list[tuple[str, bool, str]] = []

def check(label: str, ok: bool, detail: str = "") -> None:
    symbol = PASS if ok else FAIL
    line = f"  {symbol}  {label}"
    if detail:
        line += f"\n       {detail}"
    print(line)
    _results.append((label, ok, detail))

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 1 — completeness_check unit test (no LLM, no graph)
# ─────────────────────────────────────────────────────────────────────────────
def test_completeness_check() -> None:
    print("\n── Suite 1: completeness_check logic ───────────────────────────────")
    from datetime import date
    from app.graph.nodes import completeness_check, _REQUIRED_FIELDS
    from app.schemas.complaint import (
        ComplaintExtraction, ComplaintSource, ComplaintType
    )
    from uuid import uuid4

    # Case A: all required fields filled → score = 1.0, missing = []
    full = ComplaintExtraction(
        complaint_source=ComplaintSource.EMAIL,
        customer_name="Test Corp",
        product_name="Amoxicillin",
        batch_lot_number="BX-001",
        complaint_type=ComplaintType.PACKAGING,
        complaint_date=date(2024, 3, 1),
        detailed_complaint_description="Broken seal.",
    )
    state_a = {"session_id": uuid4(), "extraction": full,
               "duplicate_ids": [], "duplicate_scores": [], "missing_fields": [],
               "completeness_score": 0.0, "extraction_attempts": 0}
    result_a = completeness_check(state_a)
    check("Full extraction → score = 1.0",
          result_a["completeness_score"] == 1.0,
          f"score={result_a['completeness_score']}")
    check("Full extraction → missing_fields = []",
          result_a["missing_fields"] == [],
          str(result_a["missing_fields"]))

    # Case B: only product_name filled → score = 1/7
    sparse = ComplaintExtraction(product_name="Amoxicillin")
    state_b = dict(state_a, extraction=sparse)
    result_b = completeness_check(state_b)
    expected_score = round(1 / len(_REQUIRED_FIELDS), 10)
    check(f"Sparse extraction → score = 1/{len(_REQUIRED_FIELDS)}",
          abs(result_b["completeness_score"] - expected_score) < 1e-9,
          f"score={result_b['completeness_score']:.4f}")
    check("Sparse extraction → 6 missing fields",
          len(result_b["missing_fields"]) == len(_REQUIRED_FIELDS) - 1,
          str(result_b["missing_fields"]))
    check("Most important missing field is first in list",
          result_b["missing_fields"][0] == "detailed_complaint_description",
          f"first={result_b['missing_fields'][0]}")

    # Case C: empty extraction → score = 0.0
    empty = ComplaintExtraction()
    state_c = dict(state_a, extraction=empty)
    result_c = completeness_check(state_c)
    check("Empty extraction → score = 0.0",
          result_c["completeness_score"] == 0.0,
          f"score={result_c['completeness_score']}")

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 2 — interrupt detection
# Tests that a complaint missing required fields causes the graph to pause.
# ─────────────────────────────────────────────────────────────────────────────
def test_interrupt_detection() -> tuple:
    print("\n── Suite 2: interrupt detection (sparse complaint) ─────────────────")
    from uuid import uuid4
    # All fields null — score will be 0.0, guaranteed to trigger ask_user
    sparse_text = "We have a problem with our product shipment. Please investigate."

    from app.graph.graph import complaint_graph, get_thread_config
    from app.graph.session_store import create_session, mark_interrupted

    session_id = uuid4()
    create_session(session_id)
    config = get_thread_config(str(session_id))

    initial_state = {
        "session_id": session_id,
        "current_node": "start",
        "raw_text": sparse_text,
        "file_type": "text",
        "extraction_attempts": 0,
        "duplicate_ids": [],
        "duplicate_scores": [],
        "missing_fields": [],
        "completeness_score": 0.0,
    }

    # Use .stream(stream_mode="updates") — the only mode that surfaces __interrupt__
    interrupted = False
    question_text = None
    for chunk in complaint_graph.stream(initial_state, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupted = True
            interrupts = chunk["__interrupt__"]
            if interrupts:
                payload = interrupts[0].value if hasattr(interrupts[0], "value") else {}
                question_text = payload.get("clarifying_question", "")
                missing_field = payload.get("missing_field", "")
                mark_interrupted(session_id, question_text, missing_field)
            break   # stop streaming after first interrupt

    check("Graph interrupted on sparse complaint", interrupted,
          "score=0.0 should always trigger ask_user")
    if interrupted:
        check("Interrupt payload contains clarifying_question",
              bool(question_text), f"question={question_text!r}")
        check("Clarifying question ends with '?'",
              bool(question_text) and question_text.strip().endswith("?"),
              question_text or "")

    # Verify checkpoint was written
    checkpoint = complaint_graph.get_state(config)
    check("Checkpoint persisted after interrupt",
          checkpoint is not None and bool(checkpoint.values),
          f"keys: {list(checkpoint.values.keys()) if checkpoint and checkpoint.values else 'none'}")

    return session_id, question_text

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 3 — resume round-trip + Q&A
# ─────────────────────────────────────────────────────────────────────────────
def test_resume_and_qa(session_id, question_text) -> None:
    print("\n── Suite 3: resume round-trip + Q&A ───────────────────────────────")
    if session_id is None:
        print("  ⚠  Skipping — Suite 2 did not produce an interrupted session")
        return

    from app.graph.graph import complaint_graph, get_thread_config
    from app.graph.session_store import get_session, mark_resumed
    from app.schemas.complaint import ComplaintExtraction
    from langgraph.types import Command

    config = get_thread_config(str(session_id))

    # Inject a rich clarification that should fill most missing fields
    clarification = (
        "The complaint is from MedSupply GmbH (email). "
        "Product: Amoxicillin 500mg, batch AMX-2024-001, received 2024-03-15. "
        "50 units had broken seals — this is a Quality Defect complaint."
    )

    mark_resumed(session_id)
    check("mark_resumed() succeeds without exception", True)

    # Resume the graph using Command(resume=clarification)
    resumed_ok = False
    final_state = None
    try:
        # stream the resume — collect all chunks until done or next interrupt
        next_question = None
        for chunk in complaint_graph.stream(Command(resume=clarification), config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupts = chunk["__interrupt__"]
                q2 = interrupts[0].value.get("clarifying_question", "") if interrupts and hasattr(interrupts[0], "value") else ""
                check("Graph resumed and asked a follow-up question",
                      bool(q2), f"next question: {q2!r}")
                next_question = q2
                resumed_ok = True
                break
            else:
                # accumulate deltas into final_state
                if final_state is None:
                    final_state = {}
                for node_updates in chunk.values():
                    if isinstance(node_updates, dict):
                        final_state.update(node_updates)
        if next_question is None:
            resumed_ok = True  # completed fully
    except Exception as exc:
        check("Graph resumed without unexpected error",
              False, f"{type(exc).__name__}: {exc}")

    check("Graph resumed after update_state()", resumed_ok)

    if final_state:
        extraction = final_state.get("extraction") or ComplaintExtraction()
        check("Resumed graph produced an extraction",
              extraction is not None)
        check("Extraction has product_name after resume",
              extraction.product_name is not None,
              f"product_name={extraction.product_name}")

    # ── Q&A sub-test ─────────────────────────────────────────────────────────
    print("\n  Q&A sub-test:")
    from app.api.routes import _answer_question

    checkpoint = complaint_graph.get_state(config)
    state = checkpoint.values if checkpoint else {}

    answer = _answer_question("What is the batch number for this complaint?", state)
    check("Q&A returns a non-empty answer",
          bool(answer) and len(answer) > 10,
          f"answer={answer[:120]!r}")
    check("Q&A answer is a string",
          isinstance(answer, str))

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    test_completeness_check()

    result = test_interrupt_detection()
    session_id = result[0] if isinstance(result, tuple) else None
    question_text = result[1] if isinstance(result, tuple) else None

    test_resume_and_qa(session_id, question_text)

    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{'═'*60}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        for label, ok, detail in _results:
            if not ok:
                print(f"    {FAIL} {label}")
                if detail:
                    print(f"       {detail[:200]}")
        sys.exit(1)
    else:
        print("  — all checks passed ✓")
        sys.exit(0)

if __name__ == "__main__":
    main()
