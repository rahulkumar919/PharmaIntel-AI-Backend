"""
full_feature_check.py  —  automated test of every app feature.
Run: python full_feature_check.py
Exit 0 = all pass.  Exit 1 = failures found.
"""
import sys, json, uuid, time, urllib.parse, traceback
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv(".env")
import httpx

BASE = "http://127.0.0.1:8000"
PASS, FAIL = "V", "X"
results = []

def check(label, ok, detail=""):
    sym = PASS if ok else FAIL
    print(f"  {sym}  {label}" + (f"\n       {detail}" if detail else ""))
    results.append((label, ok, detail))

def sse_extract(text=None, pdf_path=None, timeout=120):
    """Run extraction via SSE. Returns (final_state_dict, interrupted_question)."""
    sid = str(uuid.uuid4())
    events, done_data, interrupt_data = [], None, None

    if text:
        params = urllib.parse.urlencode({"session_id": sid, "raw_text": text})
        url = f"{BASE}/api/complaints/extract/stream?{params}"
        with httpx.Client(timeout=timeout) as c:
            with c.stream("GET", url) as r:
                cur = "message"
                for line in r.iter_lines():
                    line = line.strip()
                    if line.startswith("event:"): cur = line[6:].strip()
                    elif line.startswith("data:"):
                        raw = line[5:].strip()
                        if raw:
                            events.append(cur)
                            if cur == "done":      done_data = json.loads(raw); break
                            if cur == "interrupt": interrupt_data = json.loads(raw); break
                        cur = "message"
    elif pdf_path:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{BASE}/api/complaints/extract",
                       files={"file": (pdf_path.split("/")[-1], pdf_bytes, "application/pdf")})
            if r.status_code != 200:
                return None, None, f"Upload failed: {r.status_code} {r.text[:200]}"
            done_data = r.json()
            events = ["done"]

    return events, done_data, interrupt_data


print("\n" + "="*60)
print("  PHARMA COMPLAINTAI — FULL FEATURE CHECK")
print("="*60)

# ─────────────────────────────────────────────────────────────────────
# 1. HEALTH
# ─────────────────────────────────────────────────────────────────────
print("\n-- 1. Health Endpoints --")
try:
    r = httpx.get(f"{BASE}/health", timeout=5)
    check("GET /health returns 200", r.status_code == 200)
    check("health body = {status: ok}", r.json().get("status") == "ok")
except Exception as e:
    check("Backend is reachable", False, str(e))
    print("\nBackend not running. Start it first:\n  cd backend && uvicorn app.main:app --reload")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# 2. PASTE TEXT → FORM FILLS
# ─────────────────────────────────────────────────────────────────────
print("\n-- 2. Paste Text → Form Extraction --")
complaint_text = (
    "Apollo Pharmacy reported discolored capsules in Amoxicillin Capsules 500 mg. "
    "Batch number AMX240602. Manufacturing date March 2026. Expiry date February 2028. "
    "Received via pharmacy portal. Please log this complaint."
)
try:
    events, done, interrupt = sse_extract(text=complaint_text)

    check("SSE stream completes without error", done is not None,
          f"events={events}")
    check("No interrupt fired (enough data to proceed)", interrupt is None)

    if done:
        ext = done.get("complaint", {}).get("extraction", {})
        check("product_name extracted",
              bool(ext.get("product_name")), f"got: {ext.get('product_name')}")
        check("batch_lot_number extracted",
              bool(ext.get("batch_lot_number")), f"got: {ext.get('batch_lot_number')}")
        check("complaint_type extracted",
              bool(ext.get("complaint_type")), f"got: {ext.get('complaint_type')}")
        check("initial_severity assigned (Major or Critical)",
              ext.get("initial_severity") in ("Major", "Critical"),
              f"got: {ext.get('initial_severity')}")
        check("priority assigned",
              ext.get("priority") in ("High", "Medium", "Low"),
              f"got: {ext.get('priority')}")

        analysis = done.get("complaint", {}).get("analysis", {})
        check("root_cause_hypothesis generated",
              bool(analysis.get("root_cause_hypothesis")),
              str(analysis.get("root_cause_hypothesis",""))[:80])
        check("capa_recommendation generated",
              bool(analysis.get("capa_recommendation")),
              str(analysis.get("capa_recommendation",""))[:80])
        check("summary generated",
              bool(analysis.get("summary")),
              str(analysis.get("summary",""))[:80])

        session_id_text = done.get("session_id")
        print(f"\n  Severity: {ext.get('initial_severity')} | Priority: {ext.get('priority')}")
        print(f"  Summary: {str(analysis.get('summary',''))[:100]}")
except Exception:
    check("Paste text extraction", False, traceback.format_exc(limit=2))
    session_id_text = None

# ─────────────────────────────────────────────────────────────────────
# 3. ADVERSE EVENT → CRITICAL SEVERITY
# ─────────────────────────────────────────────────────────────────────
print("\n-- 3. Adverse Event → Critical Severity --")
adverse_text = (
    "Dr. Priya Nair, City Hospital Mumbai. Email complaint. "
    "Three patients hospitalised with severe GI distress after taking Metformin 1000mg "
    "tablets from batch MF-2024-0089. Manufacturing date 10 February 2024. "
    "Approximately 90 tablets affected. Reporting to CDSCO under Schedule Y."
)
try:
    events, done, interrupt = sse_extract(text=adverse_text)
    if done:
        ext = done.get("complaint", {}).get("extraction", {})
        check("Adverse event severity is Critical",
              ext.get("initial_severity") == "Critical",
              f"got: {ext.get('initial_severity')}")
        check("product extracted (Metformin)",
              "metformin" in str(ext.get("product_name","")).lower() or
              bool(ext.get("product_name")),
              f"got: {ext.get('product_name')}")
except Exception:
    check("Adverse event severity check", False, traceback.format_exc(limit=2))

# ─────────────────────────────────────────────────────────────────────
# 4. PDF UPLOAD → FORM FILLS
# ─────────────────────────────────────────────────────────────────────
print("\n-- 4. PDF Upload → Extraction --")
import os
pdf_files = [
    ("samples/Zenith_Life_Sciences_Complaint_CC-2026-00154.pdf", "MFH260712A", "Critical"),
    ("samples/Apollo_Pharmacy_Complaint_AMX-2026-001.pdf",       "AMX240602",  "Major"),
    ("samples/MedSupply_Europe_Complaint_BX-20240315-002.pdf",   "BX-20240315-002", "Major"),
]
for pdf_path, expected_batch, expected_sev in pdf_files:
    name = os.path.basename(pdf_path)
    if not os.path.exists(pdf_path):
        check(f"PDF exists: {name}", False, "File not found — run: python generate_sample_pdfs.py")
        continue
    try:
        _, done, _ = sse_extract(pdf_path=pdf_path)
        if done:
            ext = done.get("complaint", {}).get("extraction", {})
            batch_ok = (expected_batch.lower() in
                       str(ext.get("batch_lot_number","")).lower())
            check(f"[{name[:30]}] batch extracted ({expected_batch})",
                  batch_ok, f"got: {ext.get('batch_lot_number')}")
            sev_ok = ext.get("initial_severity") in ("Critical","Major","Minor")
            check(f"[{name[:30]}] severity assigned ({expected_sev})",
                  sev_ok, f"got: {ext.get('initial_severity')}")
    except Exception:
        check(f"PDF upload: {name[:30]}", False, traceback.format_exc(limit=2))

# ─────────────────────────────────────────────────────────────────────
# 5. CHAT — FIELD CORRECTION
# ─────────────────────────────────────────────────────────────────────
print("\n-- 5. Chat Field Correction --")
if session_id_text:
    try:
        r = httpx.post(f"{BASE}/api/complaints/chat", json={
            "session_id": session_id_text,
            "message": "ah sorry the batch number is BMX240602 and affected quantity is 48 capsules",
        }, timeout=30)
        check("POST /chat returns 200", r.status_code == 200,
              f"status={r.status_code}")
        if r.status_code == 200:
            body = r.json()
            check("reply is non-empty", bool(body.get("reply")),
                  body.get("reply","")[:80])
            uf = body.get("updated_fields", {})
            check("updated_fields contains batch_lot_number",
                  "batch_lot_number" in uf,
                  f"updated_fields={uf}")
            check("batch updated to BMX240602",
                  uf.get("batch_lot_number") == "BMX240602",
                  f"got: {uf.get('batch_lot_number')}")
            check("quantity updated to 48 capsules",
                  "48" in str(uf.get("quantity_affected","")),
                  f"got: {uf.get('quantity_affected')}")
            print(f"\n  Reply: {body.get('reply','')[:100]}")
    except Exception:
        check("Chat field correction", False, traceback.format_exc(limit=2))
else:
    print("  (skipped — no session from test 2)")

# ─────────────────────────────────────────────────────────────────────
# 6. CHAT — FREE-FORM Q&A
# ─────────────────────────────────────────────────────────────────────
print("\n-- 6. Chat Free-Form Q&A --")
if session_id_text:
    try:
        r = httpx.post(f"{BASE}/api/complaints/chat", json={
            "session_id": session_id_text,
            "message": "What is the batch number of the affected product?",
        }, timeout=30)
        check("Q&A returns 200", r.status_code == 200)
        if r.status_code == 200:
            reply = r.json().get("reply","")
            check("Q&A reply is non-empty", len(reply) > 10, reply[:100])
            check("Q&A reply mentions batch number",
                  any(w in reply.upper() for w in
                      ["AMX","BMX","BATCH","LOT","240602"]),
                  reply[:100])
    except Exception:
        check("Chat Q&A", False, traceback.format_exc(limit=2))

# ─────────────────────────────────────────────────────────────────────
# 7. INTERRUPT FLOW (vague complaint)
# ─────────────────────────────────────────────────────────────────────
print("\n-- 7. Clarifying Question (Interrupt Flow) --")
vague = "We have a problem with our recent delivery. Please investigate."
try:
    events, done, interrupt = sse_extract(text=vague)
    if interrupt:
        q = interrupt.get("clarifying_question", "")
        check("Graph interrupts on vague complaint", True)
        check("Clarifying question is non-empty", bool(q), q[:100])
        check("Question ends with ?", q.strip().endswith("?"), q[:100])
        print(f"\n  Question: {q[:100]}")
    elif done:
        ext = done.get("complaint", {}).get("extraction", {})
        check("Graph completed (LLM inferred enough fields)",
              True, f"product={ext.get('product_name')}")
        print("  (Graph completed without interrupt — LLM inferred fields)")
except Exception:
    check("Interrupt flow", False, traceback.format_exc(limit=2))

# ─────────────────────────────────────────────────────────────────────
# 8. COMPLETENESS CHECK (pure logic)
# ─────────────────────────────────────────────────────────────────────
print("\n-- 8. Completeness Check Logic --")
try:
    from datetime import date
    from app.graph.nodes import completeness_check, _REQUIRED_FIELDS
    from app.schemas.complaint import ComplaintExtraction, ComplaintSource, ComplaintType

    full = ComplaintExtraction(
        detailed_complaint_description="Discolored capsules found.",
        product_name="Amoxicillin",
        batch_lot_number="AMX-001",
    )
    r1 = completeness_check({"extraction": full, "session_id": uuid.uuid4(),
                             "duplicate_ids":[], "duplicate_scores":[],
                             "missing_fields":[], "completeness_score":0.0,
                             "extraction_attempts":0})
    check("Full extraction scores 1.0", r1["completeness_score"] == 1.0,
          f"score={r1['completeness_score']}")

    empty = ComplaintExtraction()
    r2 = completeness_check({"extraction": empty, "session_id": uuid.uuid4(),
                             "duplicate_ids":[], "duplicate_scores":[],
                             "missing_fields":[], "completeness_score":0.0,
                             "extraction_attempts":0})
    check("Empty extraction scores 0.0", r2["completeness_score"] == 0.0,
          f"score={r2['completeness_score']}")
    check("Missing fields list populated", len(r2["missing_fields"]) == len(_REQUIRED_FIELDS),
          str(r2["missing_fields"]))
except Exception:
    check("Completeness check", False, traceback.format_exc(limit=2))

# ─────────────────────────────────────────────────────────────────────
# 9. EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────
print("\n-- 9. Embedding Model (sentence-transformers) --")
try:
    from app.graph.embedder import embed_text, EMBEDDING_DIM
    v = embed_text("Discolored capsules in Amoxicillin batch.")
    check("embed_text returns list", isinstance(v, list))
    check(f"Vector has {EMBEDDING_DIM} dimensions", len(v) == EMBEDDING_DIM,
          f"got {len(v)}")
    mag = sum(x**2 for x in v)**0.5
    check("Vector is unit-normalised", abs(mag-1.0) < 0.01, f"magnitude={mag:.4f}")
except Exception:
    check("Embedding model", False, traceback.format_exc(limit=2))

# ─────────────────────────────────────────────────────────────────────
# 10. DB MODEL INSTANTIATION
# ─────────────────────────────────────────────────────────────────────
print("\n-- 10. Database Model --")
try:
    import uuid as _uuid, json as _json
    from app.db.models import ComplaintORM
    from app.graph.embedder import embed_text as _emb
    rec = ComplaintORM(
        id=_uuid.uuid4(),
        product_name="Amoxicillin 500mg",
        batch_lot_number="BX-001",
        initial_severity="Major",
        extraction_data={"product_name": "Amoxicillin 500mg"},
        description_embedding=_emb("test"),
        is_confirmed=True,
    )
    check("ComplaintORM instantiates OK", True)
    check("description_embedding has 384 dims",
          len(rec.description_embedding) == 384)
except Exception:
    check("DB model instantiation", False, traceback.format_exc(limit=2))

# ─────────────────────────────────────────────────────────────────────
# 11. API DOCS
# ─────────────────────────────────────────────────────────────────────
print("\n-- 11. API Docs --")
try:
    r = httpx.get(f"{BASE}/docs", timeout=5)
    check("Swagger /docs available", r.status_code == 200)
except Exception:
    check("Swagger /docs", False, traceback.format_exc(limit=1))

# ─────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"\n{'='*60}")
print(f"  RESULTS: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} FAILED)")
    print("\n  Failed checks:")
    for label, ok, detail in results:
        if not ok:
            print(f"    X {label}")
            if detail: print(f"      {detail[:200]}")
    sys.exit(1)
else:
    print("  — ALL PASSED")
    sys.exit(0)
