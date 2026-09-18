"""
smoke_test_phase4.py — verifies Phase 4 nodes without requiring a live Postgres.

Run from backend/:
    python smoke_test_phase4.py

Test suites:
  1. embedder unit test         — model loads, embed_text() returns 384-dim vector
  2. risk_classifier unit test  — real LLM call, returns valid Severity + Priority
  3. capa_generator unit test   — real LLM call (capa_model), returns non-empty strings
  4. summariser unit test       — real LLM call, returns 2-3 sentence summary
  5. Full graph integration     — parse→extract→completeness→risk→dup(mocked)→capa→summary
     duplicate_detection is mocked (no DB needed) — returns [] gracefully
  6. DB model instantiation     — ComplaintORM can be created from extraction dict

Exit 0 = all passed.  Exit 1 = at least one failure.
"""
from __future__ import annotations
import sys, logging, traceback
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(level=logging.WARNING)

# ── helpers ───────────────────────────────────────────────────────────────────
PASS, FAIL = "V", "X"
_results: list[tuple[str, bool, str]] = []

def check(label: str, ok: bool, detail: str = "") -> None:
    symbol = PASS if ok else FAIL
    print(f"  {symbol}  {label}" + (f"\n       {detail}" if detail else ""))
    _results.append((label, ok, detail))

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 1 — embedder
# ─────────────────────────────────────────────────────────────────────────────
def test_embedder() -> None:
    print("\n-- Suite 1: embedder --")
    try:
        from app.graph.embedder import embed_text, EMBEDDING_DIM
        vec = embed_text("Amoxicillin capsules with broken seals, batch BX-001.")
        check("embed_text() returns a list", isinstance(vec, list))
        check(f"vector has {EMBEDDING_DIM} dimensions", len(vec) == EMBEDDING_DIM,
              f"got {len(vec)}")
        check("vector values are floats", all(isinstance(v, float) for v in vec[:5]))
        # Unit-normalised: magnitude should be ~1.0
        magnitude = sum(v**2 for v in vec) ** 0.5
        check("vector is unit-normalised (magnitude ~1.0)",
              abs(magnitude - 1.0) < 0.01, f"magnitude={magnitude:.4f}")
        # Two different texts should produce different vectors
        vec2 = embed_text("Metformin tablets adverse event report.")
        different = any(abs(a - b) > 1e-6 for a, b in zip(vec, vec2))
        check("different texts produce different vectors", different)
        # Empty text returns zero vector
        zero = embed_text("")
        check("empty text returns zero vector", all(v == 0.0 for v in zero))
    except Exception:
        check("embedder suite raised no exception", False, traceback.format_exc(limit=3))

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 2 — risk_classifier (real LLM call)
# ─────────────────────────────────────────────────────────────────────────────
def test_risk_classifier() -> None:
    print("\n-- Suite 2: risk_classifier (LLM) --")
    try:
        from datetime import date
        from app.graph.risk_classifier import classify_risk
        from app.schemas.complaint import (
            ComplaintExtraction, ComplaintSource, ComplaintType, Severity, Priority
        )

        # Adverse event should come back Critical/High
        extraction = ComplaintExtraction(
            product_name="Metformin 1000mg",
            batch_lot_number="MF-2024-0089",
            complaint_type=ComplaintType.ADVERSE_EVENT,
            quantity_affected="90 tablets",
            customer_name="Dr. Priya Nair",
            detailed_complaint_description=(
                "Three patients hospitalised with severe GI distress and elevated lactate "
                "after taking their regular Metformin dose from this batch."
            ),
        )
        result = classify_risk(extraction)
        check("classify_risk() returns RiskOutput", result is not None)
        check("initial_severity is a Severity enum",
              isinstance(result.initial_severity, Severity),
              str(result.initial_severity))
        check("priority is a Priority enum",
              isinstance(result.priority, Priority),
              str(result.priority))
        check("justification is non-empty",
              bool(result.justification) and len(result.justification) > 20,
              result.justification[:100])
        # An adverse event should be Critical or at minimum Major
        check("adverse event severity is Critical or Major",
              result.initial_severity in (Severity.CRITICAL, Severity.MAJOR),
              f"got {result.initial_severity}")

        # Minor issue (documentation) should come back Minor or Major (not Critical)
        minor_extraction = ComplaintExtraction(
            product_name="Ibuprofen 400mg",
            batch_lot_number="IBU-001",
            complaint_type=ComplaintType.DELIVERY_DOCUMENTATION,
            detailed_complaint_description="Missing Certificate of Analysis document. Product intact.",
        )
        minor_result = classify_risk(minor_extraction)
        check("documentation complaint is not Critical",
              minor_result.initial_severity != Severity.CRITICAL,
              f"got {minor_result.initial_severity}")
    except Exception:
        check("risk_classifier suite raised no exception", False, traceback.format_exc(limit=3))

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 3 — capa_generator (real LLM call, capa_model)
# ─────────────────────────────────────────────────────────────────────────────
def test_capa_generator() -> None:
    print("\n-- Suite 3: capa_generator (LLM capa_model) --")
    try:
        from uuid import uuid4
        from datetime import date
        from app.graph.capa_generator import generate_capa
        from app.schemas.complaint import (
            ComplaintExtraction, ComplaintSource, ComplaintType, Severity, Priority
        )

        extraction = ComplaintExtraction(
            product_name="Amoxicillin Trihydrate Capsules",
            product_strength_grade="500 mg",
            batch_lot_number="BX-20240315-002",
            complaint_type=ComplaintType.PACKAGING,
            initial_severity=Severity.MAJOR,
            priority=Priority.HIGH,
            detailed_complaint_description=(
                "50 capsules from batch BX-20240315-002 had broken induction seals "
                "on HDPE bottle closures. Found during incoming inspection."
            ),
        )
        state = {
            "session_id": uuid4(),
            "extraction": extraction,
            "risk_justification": "Packaging seal failure, Major severity, batch in distribution.",
            "duplicate_ids": [],
            "duplicate_scores": [],
        }
        result = generate_capa(state)
        check("generate_capa() returns CapaOutput", result is not None)
        check("root_cause_hypothesis is non-empty",
              bool(result.root_cause_hypothesis) and len(result.root_cause_hypothesis) > 30,
              result.root_cause_hypothesis[:120])
        check("capa_recommendation is non-empty",
              bool(result.capa_recommendation) and len(result.capa_recommendation) > 30,
              result.capa_recommendation[:120])
        # CAPA should mention the batch or quarantine for a packaging issue
        capa_lower = result.capa_recommendation.lower()
        check("CAPA mentions a concrete action (quarantine/recall/inspect/seal/corrective)",
              any(w in capa_lower for w in
                  ["quarantine", "recall", "inspect", "seal", "corrective", "batch"]),
              result.capa_recommendation[:200])
    except Exception:
        check("capa_generator suite raised no exception", False, traceback.format_exc(limit=3))

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 4 — summariser (real LLM call)
# ─────────────────────────────────────────────────────────────────────────────
def test_summariser() -> None:
    print("\n-- Suite 4: summariser (LLM) --")
    try:
        from uuid import uuid4
        from app.graph.summariser import generate_summary
        from app.schemas.complaint import (
            ComplaintExtraction, ComplaintType, Severity, Priority
        )

        extraction = ComplaintExtraction(
            product_name="Atorvastatin Calcium Tablets",
            product_strength_grade="40 mg",
            batch_lot_number="AT-2024-0441-B",
            complaint_type=ComplaintType.LABELING,
            initial_severity=Severity.CRITICAL,
            priority=Priority.HIGH,
            detailed_complaint_description=(
                "PIL instructs twice-daily dosing instead of approved once-daily. "
                "Overdose risk if patients follow leaflet."
            ),
        )
        state = {
            "session_id": uuid4(),
            "extraction": extraction,
            "risk_justification": "Critical — direct overdose risk.",
            "root_cause_hypothesis": "Wrong master document version used for PIL print run.",
            "capa_text": "Class II recall; corrected PIL to be issued.",
        }
        summary = generate_summary(state)
        check("generate_summary() returns a non-empty string",
              bool(summary) and len(summary) > 30, summary[:120])
        # Should mention product and batch
        lower = summary.lower()
        check("summary mentions product or batch",
              any(w in lower for w in ["atorvastatin", "at-2024", "labeling", "label", "recall"]),
              summary[:200])
    except Exception:
        check("summariser suite raised no exception", False, traceback.format_exc(limit=3))

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 5 — full graph integration (DB mocked)
# ─────────────────────────────────────────────────────────────────────────────
def test_full_graph_phase4() -> None:
    print("\n-- Suite 5: full graph integration (duplicate_detection mocked) --")
    try:
        from uuid import uuid4
        from unittest.mock import patch
        from app.graph.graph import complaint_graph, get_thread_config
        from app.graph.session_store import create_session

        # Mock find_duplicates so we don't need a live DB
        with patch("app.graph.nodes.find_duplicates", return_value=([], [])):
            session_id = uuid4()
            create_session(session_id)
            config = get_thread_config(str(session_id))

            # A well-formed complaint — should flow straight through without interrupt
            text = (
                "From: MedSupply Europe GmbH\n"
                "Date: 15 March 2024\n"
                "Product: Amoxicillin 500mg Capsules, Batch BX-20240315-002\n"
                "Issue: 50 capsule bottles with broken induction seals found during "
                "incoming inspection. Complaint received via email."
            )
            initial_state = {
                "session_id": session_id,
                "current_node": "start",
                "raw_text": text,
                "file_type": "text",
                "extraction_attempts": 0,
                "duplicate_ids": [],
                "duplicate_scores": [],
                "missing_fields": [],
                "completeness_score": 0.0,
            }

            final_state = {}
            interrupted = False
            for chunk in complaint_graph.stream(initial_state, config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    interrupted = True
                    break
                for node_deltas in chunk.values():
                    if isinstance(node_deltas, dict):
                        final_state.update(node_deltas)

        check("graph completed without interrupt", not interrupted)

        extraction = final_state.get("extraction")
        check("extraction is present in final state", extraction is not None)
        check("product_name extracted",
              extraction and extraction.product_name is not None,
              str(getattr(extraction, "product_name", None)))
        check("initial_severity filled by risk_classification",
              extraction and extraction.initial_severity is not None,
              str(getattr(extraction, "initial_severity", None)))
        check("priority filled by risk_classification",
              extraction and extraction.priority is not None,
              str(getattr(extraction, "priority", None)))
        check("risk_justification in state",
              bool(final_state.get("risk_justification")),
              (final_state.get("risk_justification") or "")[:80])
        check("root_cause_hypothesis in state",
              bool(final_state.get("root_cause_hypothesis")),
              (final_state.get("root_cause_hypothesis") or "")[:80])
        check("capa_text in state",
              bool(final_state.get("capa_text")),
              (final_state.get("capa_text") or "")[:80])
        check("summary_text in state",
              bool(final_state.get("summary_text")),
              (final_state.get("summary_text") or "")[:80])

        print(f"\n  Summary: {final_state.get('summary_text', '')[:160]}")
        print(f"  Severity: {getattr(extraction, 'initial_severity', 'N/A')}")
        print(f"  Priority: {getattr(extraction, 'priority', 'N/A')}")

    except Exception:
        check("full graph suite raised no exception", False, traceback.format_exc(limit=4))

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 6 — DB model instantiation (no live DB needed)
# ─────────────────────────────────────────────────────────────────────────────
def test_db_model() -> None:
    print("\n-- Suite 6: DB model instantiation --")
    try:
        import uuid, json
        from app.db.models import ComplaintORM
        from app.graph.embedder import embed_text, EMBEDDING_DIM

        vec = embed_text("Test complaint description for sealing defect.")
        record = ComplaintORM(
            id=uuid.uuid4(),
            product_name="Amoxicillin 500mg",
            batch_lot_number="BX-001",
            customer_name="Test Customer",
            complaint_date="2024-03-15",
            initial_severity="Major",
            priority="High",
            extraction_data={"product_name": "Amoxicillin 500mg"},
            risk_justification="Test justification.",
            root_cause_hypothesis="Test root cause.",
            capa_text="Test CAPA.",
            summary_text="Test summary.",
            duplicate_ids=json.dumps([]),
            description_embedding=vec,
            raw_text="Raw complaint text.",
            is_confirmed=True,
        )
        check("ComplaintORM instantiates without error", True)
        check("description_embedding has correct dimension",
              len(record.description_embedding) == EMBEDDING_DIM,
              f"got {len(record.description_embedding)}")
        check("__repr__ works", "Amoxicillin" in repr(record))
    except Exception:
        check("DB model suite raised no exception", False, traceback.format_exc(limit=3))

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    test_embedder()
    test_risk_classifier()
    test_capa_generator()
    test_summariser()
    test_full_graph_phase4()
    test_db_model()

    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{'='*60}")
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
        print("  -- all checks passed")
        sys.exit(0)

if __name__ == "__main__":
    main()
