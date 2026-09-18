"""
smoke_test_phase5.py — verifies Phase 5 SSE streaming end-to-end.

Tests:
  1. streamer unit test  — async generator yields correctly shaped SSE frames
                           for a known complaint text (graph mocked for speed)
  2. Live SSE integration — starts uvicorn in a background thread, opens the
                            SSE endpoint with httpx, verifies event sequence
  3. Sync POST /extract  — still works as a non-SSE fallback

Run from backend/:
    python smoke_test_phase5.py

Exit 0 = all passed.  Exit 1 = failure.
"""
from __future__ import annotations
import asyncio, json, sys, logging, time, threading, traceback
from pathlib import Path
from uuid import uuid4

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
logging.basicConfig(level=logging.WARNING)

PASS, FAIL = "V", "X"
_results: list[tuple[str, bool, str]] = []

def check(label: str, ok: bool, detail: str = "") -> None:
    symbol = PASS if ok else FAIL
    print(f"  {symbol}  {label}" + (f"\n       {detail}" if detail else ""))
    _results.append((label, ok, detail))

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 1 — streamer unit test (graph mocked, no LLM/DB needed)
# ─────────────────────────────────────────────────────────────────────────────

async def _collect_sse_frames(initial_state, session_id) -> list[dict]:
    """Run stream_graph_events and collect all frames into a list."""
    from app.graph.streamer import stream_graph_events
    frames = []
    async for frame in stream_graph_events(initial_state, session_id):
        frames.append(frame)
    return frames


def test_streamer_unit() -> None:
    print("\n-- Suite 1: streamer unit test (graph mocked) --")
    from unittest.mock import patch, MagicMock
    from uuid import uuid4
    from app.graph.session_store import create_session
    from app.schemas.complaint import (
        ComplaintExtraction, ComplaintSource, ComplaintType, Severity, Priority
    )
    from datetime import date

    session_id = uuid4()
    create_session(session_id)

    # Build a plausible final extraction so _done_frame has real data
    extraction = ComplaintExtraction(
        complaint_source=ComplaintSource.EMAIL,
        customer_name="Test Corp",
        product_name="Amoxicillin 500mg",
        batch_lot_number="BX-001",
        complaint_type=ComplaintType.PACKAGING,
        complaint_date=date(2024, 3, 15),
        detailed_complaint_description="Broken seals on 50 bottles.",
        initial_severity=Severity.MAJOR,
        priority=Priority.HIGH,
    )

    # Simulate what complaint_graph.stream() yields in stream_mode="updates"
    # Two normal node chunks, then END (loop exits naturally)
    mock_chunks = [
        {"parse_document":      {"raw_text": "test text", "current_node": "parse_document"}},
        {"extract_fields":      {"extraction": extraction, "extraction_attempts": 1,
                                  "current_node": "extract_fields"}},
        {"completeness_check":  {"completeness_score": 1.0, "missing_fields": [],
                                  "current_node": "completeness_check"}},
        {"risk_classification": {"extraction": extraction, "risk_justification": "Major/High.",
                                  "current_node": "risk_classification"}},
        {"duplicate_detection": {"duplicate_ids": [], "duplicate_scores": [],
                                  "current_node": "duplicate_detection"}},
        {"capa_node":           {"root_cause_hypothesis": "Sealing machine issue.",
                                  "capa_text": "Quarantine batch.", "current_node": "capa_node"}},
        {"summary_node":        {"summary_text": "A packaging complaint.",
                                  "current_node": "summary_node"}},
    ]

    try:
        with patch("app.graph.streamer.complaint_graph") as mock_graph:
            mock_graph.stream.return_value = iter(mock_chunks)
            mock_graph.get_state.return_value = MagicMock(values={})

            initial_state = {
                "session_id": session_id, "current_node": "start",
                "raw_text": "Broken seals on Amoxicillin batch BX-001.",
                "file_type": "text", "extraction_attempts": 0,
                "duplicate_ids": [], "duplicate_scores": [],
                "missing_fields": [], "completeness_score": 0.0,
            }

            frames = asyncio.run(_collect_sse_frames(initial_state, session_id))

        check("stream_graph_events() yields at least one frame", len(frames) > 0,
              f"got {len(frames)} frames")

        event_types = [f.get("event") for f in frames]
        check("stream includes node_progress events",
              "node_progress" in event_types, str(event_types))
        check("stream ends with 'done' event",
              event_types[-1] == "done", f"last event={event_types[-1]}")
        check("no 'error' events",
              "error" not in event_types, str(event_types))

        # Verify node_progress frames have correct structure
        progress_frames = [f for f in frames if f.get("event") == "node_progress"]
        check("node_progress frames have 'data' field",
              all("data" in f for f in progress_frames))

        # Parse the data JSON of a progress frame
        sample = json.loads(progress_frames[0]["data"])
        check("node_progress data has 'node' field", "node" in sample, str(sample))
        check("node_progress data has 'status' field", "status" in sample, str(sample))
        check("node_progress data has 'detail' field", "detail" in sample, str(sample))

        # Verify started + completed pairs exist for at least one node
        statuses = [(json.loads(f["data"])["node"], json.loads(f["data"])["status"])
                    for f in progress_frames]
        nodes_started   = {n for n, s in statuses if s == "started"}
        nodes_completed = {n for n, s in statuses if s == "completed"}
        check("each node emits both 'started' and 'completed'",
              nodes_started == nodes_completed,
              f"started={nodes_started} completed={nodes_completed}")

        # Verify 'done' frame contains valid ExtractResponse JSON
        done_frame = next(f for f in frames if f.get("event") == "done")
        done_data = json.loads(done_frame["data"])
        check("done event has session_id", "session_id" in done_data, str(list(done_data.keys())))
        check("done event has complaint", "complaint" in done_data)
        check("done complaint has extraction",
              "extraction" in done_data.get("complaint", {}))

    except Exception:
        check("streamer unit test raised no exception", False, traceback.format_exc(limit=4))

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 2 — live SSE integration test (real uvicorn, real EventSource via httpx)
# ─────────────────────────────────────────────────────────────────────────────

def _start_server(port: int) -> threading.Event:
    """
    Start uvicorn in a daemon thread.  Returns a threading.Event that is set
    once the server is ready (detected by a successful GET /health).
    """
    import uvicorn
    ready = threading.Event()

    def _run():
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=port,
            log_level="error",   # suppress uvicorn access logs during test
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Poll /health until the server responds (max 15s)
    import httpx
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200:
                ready.set()
                return ready
        except Exception:
            pass
        time.sleep(0.3)

    return ready


def test_live_sse() -> None:
    print("\n-- Suite 2: live SSE integration (real server) --")
    try:
        import httpx
    except ImportError:
        check("httpx available for SSE test", False, "pip install httpx")
        return

    PORT = 18421
    ready = _start_server(PORT)
    base = f"http://127.0.0.1:{PORT}"

    if not ready.is_set():
        check("uvicorn started within 15s", False, "server did not start in time")
        return

    check("uvicorn server started", True)

    # A well-formed complaint that should complete without interrupt
    complaint_text = (
        "From: MedSupply Europe GmbH. "
        "Product: Amoxicillin Trihydrate Capsules 500mg, batch BX-20240315-002. "
        "Complaint received by email on 15 March 2024. "
        "50 capsule bottles had broken induction seals. Quality Defect complaint."
    )
    session_id = str(uuid4())
    url = (
        f"{base}/api/complaints/extract/stream"
        f"?session_id={session_id}"
        f"&raw_text={httpx.URL('', params={'t': complaint_text}).params['t']}"
    )

    # Re-encode properly
    import urllib.parse
    url = (
        f"{base}/api/complaints/extract/stream"
        f"?session_id={session_id}"
        f"&raw_text={urllib.parse.quote(complaint_text)}"
    )

    received_events: list[str] = []
    received_data: list[dict] = []

    # httpx streaming to consume SSE
    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("GET", url) as response:
                check("SSE endpoint returns 200",
                      response.status_code == 200,
                      f"status={response.status_code}")
                ct = response.headers.get("content-type", "")
                check("Content-Type is text/event-stream",
                      "text/event-stream" in ct, f"content-type={ct}")

                current_event = "message"
                for line in response.iter_lines():
                    line = line.strip()
                    if line.startswith("event:"):
                        current_event = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        raw = line[len("data:"):].strip()
                        if raw:
                            received_events.append(current_event)
                            try:
                                received_data.append(json.loads(raw))
                            except Exception:
                                received_data.append({"_raw": raw})
                        if current_event == "done":
                            break   # stop reading after done
                        current_event = "message"

    except Exception as exc:
        check("SSE stream consumed without connection error", False, str(exc)[:200])
        return

    check("received at least one node_progress event",
          "node_progress" in received_events,
          f"events received: {received_events[:10]}")
    check("stream ended with 'done' event",
          received_events and received_events[-1] == "done",
          f"last event: {received_events[-1] if received_events else 'none'}")

    if "done" in received_events:
        done_idx = received_events.index("done")
        done_payload = received_data[done_idx]
        check("done payload has session_id", "session_id" in done_payload,
              str(list(done_payload.keys())))
        check("done payload has complaint.extraction",
              "complaint" in done_payload and "extraction" in done_payload.get("complaint", {}))
        ext = done_payload.get("complaint", {}).get("extraction", {})
        check("product_name extracted via SSE stream",
              bool(ext.get("product_name")), f"product_name={ext.get('product_name')}")


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 3 — sync POST /extract still works (non-SSE fallback)
# ─────────────────────────────────────────────────────────────────────────────

def test_sync_extract_fallback() -> None:
    print("\n-- Suite 3: sync POST /extract fallback --")
    try:
        import httpx
        PORT = 18421   # reuse server from suite 2
        base = f"http://127.0.0.1:{PORT}"

        complaint_text = (
            "MedSupply GmbH. Amoxicillin 500mg, batch BX-20240315-002. "
            "Email complaint on 2024-03-15. 50 units broken seals. Packaging defect."
        )

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{base}/api/complaints/extract",
                data={"raw_text": complaint_text},
            )

        check("POST /extract returns 200", response.status_code == 200,
              f"status={response.status_code} body={response.text[:200]}")

        if response.status_code == 200:
            body = response.json()
            check("response has session_id", "session_id" in body)
            check("response has complaint", "complaint" in body)
            ext = body.get("complaint", {}).get("extraction", {})
            check("product_name in sync response",
                  bool(ext.get("product_name")), f"product_name={ext.get('product_name')}")

    except Exception:
        check("sync POST /extract raised no exception", False, traceback.format_exc(limit=3))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    test_streamer_unit()
    test_live_sse()
    test_sync_extract_fallback()

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
                    print(f"       {detail[:300]}")
        sys.exit(1)
    else:
        print("  -- all checks passed")
        sys.exit(0)

if __name__ == "__main__":
    main()
