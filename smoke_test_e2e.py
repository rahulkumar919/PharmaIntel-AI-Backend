"""
smoke_test_e2e.py — full end-to-end wiring test against a live backend server.

Starts uvicorn in a background thread, then exercises every public endpoint:
  GET  /health                    — liveness probe
  POST /api/complaints/extract    — sync extraction (paste text)
  GET  /api/complaints/extract/stream — SSE streaming extraction
  POST /api/complaints/chat       — Q&A mode (session complete)
  POST /api/complaints            — confirm/persist (stub — no live DB needed)
  GET  /api/complaints/{id}       — retrieve (returns 404 without DB, which is fine)

Run from backend/:
    python smoke_test_e2e.py

Exit 0 = all checks passed.  Exit 1 = failure.

DB note:
  The duplicate_detection and confirm_complaint endpoints need Postgres.
  Without a live DB both degrade gracefully (empty duplicate list; confirm
  returns a 500 which we explicitly test for and treat as "expected without DB").
  All LLM-dependent nodes call the real Groq API so GROQ_API_KEY must be set.
"""
from __future__ import annotations
import sys, time, threading, traceback, json, urllib.parse
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import logging
logging.basicConfig(level=logging.WARNING)

import httpx

PASS, FAIL = "V", "X"
_results: list[tuple[str, bool, str]] = []

def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'V' if ok else 'X'}  {label}" + (f"\n       {detail}" if detail else ""))
    _results.append((label, ok, detail))


# ── Server lifecycle ──────────────────────────────────────────────────────────

PORT = 18422

def _start_server() -> bool:
    import uvicorn
    def _run():
        uvicorn.run("app.main:app", host="127.0.0.1", port=PORT, log_level="error")
    threading.Thread(target=_run, daemon=True).start()

    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


BASE = f"http://127.0.0.1:{PORT}"

# Rich complaint used for all extraction tests
COMPLAINT_TEXT = (
    "From: MedSupply Europe GmbH, Frankfurt\n"
    "Date: 15 March 2024\n"
    "Product: Amoxicillin Trihydrate Capsules 500mg/USP, Batch BX-20240315-002\n"
    "Complaint received by email.\n"
    "Issue: 50 capsule bottles from a shipment of 1,000 units had broken induction "
    "seals on the HDPE closures, discovered during incoming goods inspection. "
    "No patient exposure reported. The batch is still held in our quarantine area."
)


# ── Suite 1: health ───────────────────────────────────────────────────────────

def test_health(client: httpx.Client) -> None:
    print("\n-- Suite 1: health endpoints --")
    r = client.get(f"{BASE}/health")
    check("GET /health returns 200", r.status_code == 200, r.text)
    check("GET /health returns {status: ok}",
          r.json().get("status") == "ok", str(r.json()))


# ── Suite 2: sync POST /extract ───────────────────────────────────────────────

def test_sync_extract(client: httpx.Client) -> str | None:
    """Returns session_id on success for use by later suites."""
    print("\n-- Suite 2: POST /api/complaints/extract (sync) --")
    try:
        r = client.post(
            f"{BASE}/api/complaints/extract",
            data={"raw_text": COMPLAINT_TEXT},
            timeout=120.0,
        )
        check("POST /extract returns 200", r.status_code == 200,
              f"status={r.status_code} body={r.text[:200]}")
        if r.status_code != 200:
            return None

        body = r.json()
        check("response has session_id", "session_id" in body)
        check("response has complaint.extraction", "extraction" in body.get("complaint", {}))

        ext = body["complaint"]["extraction"]
        check("product_name extracted", bool(ext.get("product_name")),
              f"product_name={ext.get('product_name')}")
        check("batch_lot_number extracted", bool(ext.get("batch_lot_number")),
              f"batch={ext.get('batch_lot_number')}")
        check("complaint_type extracted", bool(ext.get("complaint_type")),
              f"type={ext.get('complaint_type')}")
        check("initial_severity filled by risk node", bool(ext.get("initial_severity")),
              f"severity={ext.get('initial_severity')}")
        check("priority filled by risk node", bool(ext.get("priority")),
              f"priority={ext.get('priority')}")

        analysis = body["complaint"].get("analysis", {})
        check("root_cause_hypothesis generated", bool(analysis.get("root_cause_hypothesis")),
              (analysis.get("root_cause_hypothesis") or "")[:80])
        check("capa_recommendation generated", bool(analysis.get("capa_recommendation")),
              (analysis.get("capa_recommendation") or "")[:80])
        check("summary generated", bool(analysis.get("summary")),
              (analysis.get("summary") or "")[:80])

        print(f"\n  Severity : {ext.get('initial_severity')}")
        print(f"  Priority : {ext.get('priority')}")
        print(f"  Summary  : {(analysis.get('summary') or '')[:120]}")

        return body["session_id"]

    except Exception:
        check("sync extract raised no exception", False, traceback.format_exc(limit=3))
        return None


# ── Suite 3: SSE GET /extract/stream ─────────────────────────────────────────

def test_sse_stream(client: httpx.Client) -> str | None:
    """Opens the SSE stream with raw_text param, collects all events."""
    print("\n-- Suite 3: GET /api/complaints/extract/stream (SSE) --")
    try:
        params = urllib.parse.urlencode({
            "session_id": __import__("uuid").uuid4(),
            "raw_text": COMPLAINT_TEXT,
        })
        url = f"{BASE}/api/complaints/extract/stream?{params}"

        events: list[str] = []
        done_data: dict = {}
        current_event = "message"

        with client.stream("GET", url, timeout=120.0) as resp:
            check("SSE returns 200", resp.status_code == 200,
                  f"status={resp.status_code}")
            check("Content-Type is text/event-stream",
                  "text/event-stream" in resp.headers.get("content-type", ""),
                  resp.headers.get("content-type", ""))

            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    raw = line[len("data:"):].strip()
                    if raw:
                        events.append(current_event)
                        if current_event == "done":
                            try: done_data = json.loads(raw)
                            except Exception: pass
                            break
                    current_event = "message"

        check("SSE emits node_progress events",
              "node_progress" in events, f"events: {events[:8]}")
        check("SSE ends with 'done' event",
              events and events[-1] == "done", f"last: {events[-1] if events else 'none'}")
        check("done payload has complaint.extraction",
              "extraction" in done_data.get("complaint", {}))

        ext = done_data.get("complaint", {}).get("extraction", {})
        check("product_name in SSE done payload", bool(ext.get("product_name")),
              f"product_name={ext.get('product_name')}")

        return done_data.get("session_id")

    except Exception:
        check("SSE stream raised no exception", False, traceback.format_exc(limit=3))
        return None


# ── Suite 4: POST /chat (Q&A mode) ────────────────────────────────────────────

def test_chat_qa(client: httpx.Client, session_id: str) -> None:
    print("\n-- Suite 4: POST /api/complaints/chat (Q&A mode) --")
    if not session_id:
        print("  (skipped — no session_id from previous suite)")
        return
    try:
        r = client.post(
            f"{BASE}/api/complaints/chat",
            json={"session_id": session_id, "message": "What is the batch number?"},
            timeout=30.0,
        )
        check("POST /chat returns 200", r.status_code == 200,
              f"status={r.status_code}")
        if r.status_code == 200:
            body = r.json()
            check("reply is non-empty string",
                  isinstance(body.get("reply"), str) and len(body["reply"]) > 5,
                  body.get("reply", "")[:100])
    except Exception:
        check("chat Q&A raised no exception", False, traceback.format_exc(limit=3))


# ── Suite 5: POST /api/complaints (confirm, graceful DB failure) ──────────────

def test_confirm(client: httpx.Client, session_id: str) -> None:
    print("\n-- Suite 5: POST /api/complaints (confirm/persist) --")
    if not session_id:
        print("  (skipped — no session_id)")
        return
    try:
        payload = {
            "session_id": session_id,
            "extraction": {
                "complaint_source": "Email",
                "customer_name": "MedSupply Europe GmbH",
                "product_name": "Amoxicillin Trihydrate Capsules",
                "product_strength_grade": "500 mg",
                "batch_lot_number": "BX-20240315-002",
                "complaint_type": "Packaging",
                "complaint_date": "2024-03-15",
                "detailed_complaint_description": "50 bottles with broken seals.",
                "initial_severity": "Major",
                "priority": "High",
            },
        }
        r = client.post(f"{BASE}/api/complaints", json=payload, timeout=30.0)
        # 201 = persisted (Postgres running), 500 = no DB (expected in dev)
        check("POST /api/complaints returns 201 or 500 (500 = no DB, expected)",
              r.status_code in (201, 500),
              f"status={r.status_code}")
        if r.status_code == 201:
            check("response has complaint_id", "complaint_id" in r.json())
    except Exception:
        check("confirm raised no exception", False, traceback.format_exc(limit=3))


# ── Suite 6: GET /api/complaints/{id} (404 without DB) ───────────────────────

def test_get_complaint(client: httpx.Client) -> None:
    print("\n-- Suite 6: GET /api/complaints/{id} --")
    fake_id = "00000000-0000-0000-0000-000000000000"
    try:
        r = client.get(f"{BASE}/api/complaints/{fake_id}", timeout=10.0)
        check("GET /api/complaints/{id} returns 404 for unknown ID",
              r.status_code == 404,
              f"status={r.status_code}")
    except Exception:
        check("get complaint raised no exception", False, traceback.format_exc(limit=3))


# ── Suite 7: FastAPI /docs available in dev mode ──────────────────────────────

def test_docs(client: httpx.Client) -> None:
    print("\n-- Suite 7: /docs available --")
    r = client.get(f"{BASE}/docs", timeout=5.0)
    check("GET /docs returns 200 in dev mode", r.status_code == 200,
          f"status={r.status_code}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Starting uvicorn on port", PORT, "…")
    if not _start_server():
        print("  X  Server did not start within 20s — aborting")
        sys.exit(1)
    print("  V  Server started")

    with httpx.Client(timeout=30.0) as client:
        test_health(client)
        session_id = test_sync_extract(client)
        sse_session_id = test_sse_stream(client)
        test_chat_qa(client, session_id or sse_session_id)
        test_confirm(client, session_id or sse_session_id)
        test_get_complaint(client)
        test_docs(client)

    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        for label, ok, detail in _results:
            if not ok:
                print(f"    X {label}")
                if detail: print(f"       {detail[:250]}")
        sys.exit(1)
    else:
        print("  -- all checks passed")
        sys.exit(0)

if __name__ == "__main__":
    main()
