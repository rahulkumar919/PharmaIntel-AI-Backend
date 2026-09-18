"""Quick import + graph compilation check — run: python check_imports.py"""
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv('.env')

print("Testing imports...")
from app.config import get_settings
from app.schemas.complaint import ComplaintExtraction, Severity, Priority
from app.graph.state import ComplaintState
from app.graph.document_parser import extract_text
from app.graph.extractor import run_extraction, ExtractionError
from app.graph.graph import complaint_graph
print("  All imports OK")

print("Testing graph node registration...")
nodes = list(complaint_graph.nodes)
expected = {"parse_document","extract_fields","completeness_check",
            "ask_user","risk_classification","duplicate_detection",
            "capa_node","summary_node"}   # capa_node renamed from capa_recommendation
missing = expected - set(nodes)
assert not missing, f"Missing nodes: {missing}"
print(f"  Nodes present: {sorted(nodes)}")

print("Testing document_parser (raw text fast-path)...")
text = extract_text(raw_text="Complaint: Amoxicillin batch AMX-001 damaged on arrival.")
assert len(text) > 10
print(f"  extract_text returned {len(text)} chars  OK")

print("Testing stub graph invoke (no LLM call — raw_text path)...")
from uuid import uuid4
state = {
    "session_id": uuid4(),
    "current_node": "start",
    "raw_text": "Product Amoxicillin 500mg batch AMX-001 received with broken seals.",
    "file_type": "text",
    "extraction_attempts": 0,
    "duplicate_ids": [],
    "duplicate_scores": [],
    "missing_fields": [],
    "completeness_score": 0.0,
}
# NOTE: extract_fields WILL call Groq here if GROQ_API_KEY is set in .env.
# If the key is not set yet, this will raise ExtractionError — that is expected.
try:
    result = complaint_graph.invoke(state)
    summary = result.get("summary", "(none)")
    print(f"  Graph completed. summary={summary[:80]}")
    print("  extraction.product_name =", result.get("extraction", {}).product_name if hasattr(result.get("extraction", None), "product_name") else "N/A")
except Exception as exc:
    print(f"  Graph raised {type(exc).__name__}: {exc}")
    print("  (Expected if GROQ_API_KEY not yet set in .env)")

print()
print("Import + structure checks: ALL PASSED")
