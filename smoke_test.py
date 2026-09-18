"""
smoke_test.py — manual end-to-end test for Phase 2 nodes.

Run from the backend/ directory AFTER copying .env.example → .env and filling
in your GROQ_API_KEY:

    cd backend
    python smoke_test.py                          # tests all three sample files
    python smoke_test.py --file samples/complaint_packaging_defect.txt
    python smoke_test.py --file samples/complaint_adverse_event.eml
    python smoke_test.py --text "Product Ibuprofen 400mg batch IBU-001 received
                                 damaged, 20 tablets broken, shipped via portal."

What this script tests:
  1. document_parser.extract_text()  — file loading produces non-empty text
  2. extractor.run_extraction()      — LLM call returns a valid ComplaintExtraction
  3. The two nodes wired together via a minimal fake state dict
  4. The full complaint_graph.invoke() on real text (exercises all stub nodes too)

This is NOT a pytest suite (that comes in a later phase) — it's a quick
"does it actually work end-to-end" check you can run before committing.

Exit codes:
  0 — all checks passed
  1 — one or more checks failed (details printed to stderr)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from uuid import uuid4

# ── Make sure the app package is importable when running from backend/ ────────
# Without this, `python smoke_test.py` fails with ModuleNotFoundError
# because Python doesn't automatically add the current directory to sys.path
# when running a script (it adds the script's *parent*, which IS backend/).
# The app package lives at backend/app/, so backend/ must be on the path.
sys.path.insert(0, str(Path(__file__).parent))

# Load .env before importing app modules (pydantic-settings reads os.environ)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("smoke_test")

# ── Now safe to import app modules ────────────────────────────────────────────
from app.graph.document_parser import extract_text
from app.graph.extractor import run_extraction, ExtractionError
from app.graph.graph import complaint_graph
from app.graph.state import ComplaintState


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
_results: list[tuple[str, bool, str]] = []   # (label, passed, detail)


def check(label: str, passed: bool, detail: str = "") -> None:
    """Record a check result and print it immediately."""
    symbol = PASS if passed else FAIL
    line = f"  {symbol}  {label}"
    if detail:
        line += f"\n       {detail}"
    print(line)
    _results.append((label, passed, detail))


# ─────────────────────────────────────────────────────────────────────────────
# Individual test suites
# ─────────────────────────────────────────────────────────────────────────────

def test_parser(file_path: Path) -> str | None:
    """
    Test 1 — document_parser.extract_text().

    Reads file bytes, calls extract_text, checks the result is non-empty text.
    Returns the extracted text so the caller can pipe it into test_extraction.
    """
    print(f"\n── Parser test: {file_path.name} ──────────────────────────────")
    try:
        file_bytes = file_path.read_bytes()
        suffix = file_path.suffix.lstrip(".").lower()
        text = extract_text(
            file_bytes=file_bytes,
            file_name=file_path.name,
            file_type=suffix,
        )
        check("extract_text() returns non-empty string", bool(text and len(text) > 50),
              f"length={len(text)} chars")
        check("extract_text() contains expected keywords",
              any(kw in text.lower() for kw in ["batch", "product", "complaint", "date"]),
              text[:120].replace("\n", " "))
        return text
    except Exception:
        check("extract_text() raised no exception", False, traceback.format_exc(limit=3))
        return None


def test_raw_text_parser(raw_text: str) -> str:
    """Test 1b — parser fast-path for pre-supplied raw text."""
    print("\n── Parser test: raw text ────────────────────────────────────────")
    text = extract_text(raw_text=raw_text)
    check("extract_text() raw-text fast-path works", bool(text), f"length={len(text)}")
    return text


def test_extraction(raw_text: str, label: str) -> None:
    """
    Test 2 — extractor.run_extraction().

    Calls the real Groq API and validates the returned ComplaintExtraction.
    We check:
      - No exception raised
      - Returns a ComplaintExtraction (not None)
      - At least one of the key fields was extracted (not all None)
      - attempts is 1 or 2 (never 0 or >2)
    """
    print(f"\n── Extraction test: {label} ─────────────────────────────────────")
    try:
        extraction, attempts = run_extraction(raw_text)

        check("run_extraction() returns ComplaintExtraction",
              extraction is not None)

        check("attempts is 1 or 2 (no infinite retry)",
              attempts in (1, 2),
              f"attempts={attempts}")

        # At least 3 of the 11 extractable fields should be non-None for a
        # well-formed complaint — if fewer than 3 were extracted, the prompt
        # or the model response is likely broken.
        filled = [
            f for f in [
                extraction.customer_name, extraction.product_name,
                extraction.batch_lot_number, extraction.complaint_type,
                extraction.complaint_date, extraction.detailed_complaint_description,
                extraction.complaint_source,
            ]
            if f is not None
        ]
        check(f"≥3 key fields extracted (got {len(filled)}/7 sampled)",
              len(filled) >= 3,
              f"filled fields: {filled}")

        # Print the full extraction as pretty JSON so you can inspect it
        print("\n  Extracted fields:")
        # model_dump() serialises dates to ISO strings, enums to their .value
        for k, v in extraction.model_dump().items():
            if v is not None:
                print(f"    {k:35s}: {v}")

    except ExtractionError as exc:
        check("run_extraction() raised no ExtractionError", False, str(exc))
    except Exception:
        check("run_extraction() raised no unexpected exception", False,
              traceback.format_exc(limit=3))


def test_full_graph(raw_text: str, label: str) -> None:
    """
    Test 3 — full complaint_graph.invoke() end-to-end.

    Exercises the complete node chain (including still-stub nodes like
    risk_classification, duplicate_detection, etc.) with real text.
    Verifies that:
      - The graph completes without exception
      - Final state contains expected keys
      - extraction in final state is a ComplaintExtraction with data
    """
    print(f"\n── Full graph test: {label} ─────────────────────────────────────")
    try:
        initial_state: ComplaintState = {
            "session_id": uuid4(),
            "current_node": "start",
            "raw_text": raw_text,
            "file_type": "text",
            "extraction_attempts": 0,
            "duplicate_ids": [],
            "duplicate_scores": [],
            "missing_fields": [],
            "completeness_score": 0.0,
        }

        final_state = complaint_graph.invoke(initial_state)

        check("graph.invoke() completes without exception", True)

        check("final state contains 'summary_text' key",
              "summary_text" in final_state)

        check("final state contains 'extraction' key",
              "extraction" in final_state)

        check("final state extraction has product_name set",
              final_state.get("extraction") is not None and
              final_state["extraction"].product_name is not None,
              f"product_name={getattr(final_state.get('extraction'), 'product_name', None)}")

        print(f"\n  Summary: {final_state.get('summary_text', '(none)')}")

    except Exception:
        check("graph.invoke() raised no exception", False,
              traceback.format_exc(limit=4))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Phase 2 nodes")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=Path,
                       help="Path to a single complaint file to test")
    group.add_argument("--text", type=str,
                       help="Raw complaint text to test directly")
    args = parser.parse_args()

    samples_dir = Path(__file__).parent / "samples"

    if args.text:
        # Single raw-text test
        text = test_raw_text_parser(args.text)
        if text:
            test_extraction(text, "raw text input")
            test_full_graph(text, "raw text input")

    elif args.file:
        # Single file test
        file_path = args.file if args.file.is_absolute() else Path(__file__).parent / args.file
        text = test_parser(file_path)
        if text:
            test_extraction(text, file_path.name)
            test_full_graph(text, file_path.name)

    else:
        # Default: run all three sample files
        sample_files = [
            samples_dir / "complaint_packaging_defect.txt",
            samples_dir / "complaint_adverse_event.eml",
            samples_dir / "complaint_labeling_error.txt",
        ]
        for fp in sample_files:
            if not fp.exists():
                print(f"  ⚠  Sample file not found, skipping: {fp}")
                continue
            text = test_parser(fp)
            if text:
                test_extraction(text, fp.name)
                test_full_graph(text, fp.name)

    # ── Summary ───────────────────────────────────────────────────────────────
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
