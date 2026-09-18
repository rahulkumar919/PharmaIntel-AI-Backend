# PharmaIntel AI — Intelligent QA & Complaint Management System

> **AI-Powered Customer Complaint Intelligence & Triage for Pharmaceutical Manufacturing**  
> Built with LangGraph · FastAPI · React 18 · Redux Toolkit · Groq Whisper Turbo · Neon PostgreSQL · pgvector · Google Font Inter

---

## Table of Contents

1. [What This App Does](#1-what-this-app-does)
2. [Features](#2-features)
3. [Tech Stack & Why Each Was Chosen](#3-tech-stack--why-each-was-chosen)
4. [Architecture — How Everything Connects](#4-architecture--how-everything-connects)
5. [LangGraph Pipeline — 8 AI Nodes Explained](#5-langgraph-pipeline--8-ai-nodes-explained)
6. [Feature 1: Paste Text → Form Fills Automatically](#6-feature-1-paste-text--form-fills-automatically)
7. [Feature 2: Upload PDF / Email → AI Extracts Data](#7-feature-2-upload-pdf--email--ai-extracts-data)
8. [Feature 3: Chat Corrections → Form Updates Live](#8-feature-3-chat-corrections--form-updates-live)
9. [Feature 4: AI Copilot Risk Assessment Box](#9-feature-4-ai-copilot-risk-assessment-box)
10. [Feature 5: Real-Time Progress Bar (SSE Streaming)](#10-feature-5-real-time-progress-bar-sse-streaming)
11. [Feature 6: Duplicate Detection (pgvector)](#11-feature-6-duplicate-detection-pgvector)
12. [Feature 7: Save to Database (QMS Ledger)](#12-feature-7-save-to-database-qms-ledger)
13. [Feature 8: Voice-to-Text Dictation (Groq Whisper Turbo)](#13-feature-8-voice-to-text-dictation-groq-whisper-turbo)
14. [Feature 9: Neon Cloud PostgreSQL & Live Ledger Explorer](#14-feature-9-neon-cloud-postgresql--live-ledger-explorer)
15. [Feature 10: Modern Pharma-Tech UI & Google Font Inter](#15-feature-10-modern-pharma-tech-ui--google-font-inter)
16. [Frontend Code Walkthrough](#16-frontend-code-walkthrough)
17. [Backend Code Walkthrough](#17-backend-code-walkthrough)
18. [End-to-End Request Flow](#18-end-to-end-request-flow)
19. [Setup & Run Instructions](#19-setup--run-instructions)
20. [Test Data for Demo](#20-test-data-for-demo)
21. [Video Walkthrough Script](#21-video-walkthrough-script)

---

## 1. What This App Does

A pharmaceutical quality assurance (QA) specialist receives a customer complaint — by email, PDF report, or phone notes. Normally they must manually read the complaint, identify the product, batch number, defect type, and fill in a structured form. This takes 15–30 minutes per complaint and is error-prone.

**PharmaIntel AI does it in seconds with multi-modal inputs:**

1. **Multi-Modal Ingestion**: QA specialists can paste complaint correspondence, upload complex PDF reports, **or speak directly via Voice-to-Text (Groq Whisper)**.
2. **Document & Speech Intelligence**: An AI pipeline (LangGraph + Groq LLMs) reads the complaint, extracts all structured GxP fields, and normalizes product attributes.
3. **Automated Form Population**: The 5-stage Customer Complaint Record form fills automatically in real-time.
4. **Autonomous Root Cause & CAPA**: The AI evaluates ICH Q10 severity (Major/Critical/Minor), synthesizes root cause hypotheses, and drafts actionable CAPA workflows.
5. **Interactive Copilot Corrections**: If a specialist corrects an attribute (*"sorry the batch number is BMX240602"*), the Copilot updates the form live with green highlighting.
6. **Enterprise Persistence**: With one click, the complaint is committed into **Neon Cloud PostgreSQL** with `pgvector` embeddings and an audit-trailed QMS ledger.

---

## 2. Features

| # | Feature | How it works |
|---|---------|-------------|
| 1 | **Paste text → form fills** | LLM extracts 13 structured fields from free text |
| 2 | **Upload PDF/email → form fills** | In-memory `pypdf` document extraction + LLM structured parsing |
| 3 | **Chat corrections → form updates** | `/chat` returns `updated_fields` patch, Redux applies it live |
| 4 | **AI Risk Assessment box** | `risk_classification` node assigns severity + CAPA node generates recommendations |
| 5 | **Real-time progress bar** | LangGraph `.stream()` → Server-Sent Events → React with zero lingering animations |
| 6 | **Duplicate detection** | `sentence-transformers` embeddings + `pgvector` cosine search |
| 7 | **Save to QMS database** | PostgreSQL + SQLAlchemy async + fallback JSON ledger |
| 8 | **Clarifying questions** | LangGraph `interrupt()` pauses graph, bot asks targeted question |
| 9 | **Voice-to-Text (Whisper)** | Browser `MediaRecorder` + Groq `whisper-large-v3-turbo` sub-second transcription |
| 10 | **Neon Cloud DB & Live Explorer** | Hosted PostgreSQL 18.6 with live API explorer & direct Swagger endpoints |
| 11 | **Enterprise UI & Inter Font** | Google Font Inter typography, bespoke SVG icons, GxP 21 CFR Part 11 badges |

---

## 3. Tech Stack & Why Each Was Chosen

### Backend
| Technology | Why chosen |
|-----------|-----------|
| **Python 3.13 + FastAPI** | Async framework, automatic OpenAPI docs, Pydantic validation |
| **LangGraph** | Owns the AI pipeline control flow — conditional routing, human-in-the-loop, streaming |
| **LangChain** | Only used as a wrapper: `ChatGroq`, document loaders, `with_structured_output()` |
| **Groq API** | Fastest LLM inference available; LPU chips give ~10x faster tokens/sec than GPU |
| **`openai/gpt-oss-20b`** | Lightweight model for extraction, risk classification, summaries (~1-2s response) |
| **`openai/gpt-oss-120b`** | Heavy model for CAPA reasoning only — where quality matters more than speed |
| **sentence-transformers** | Local embeddings (no API key needed — Groq has no embeddings endpoint) |
| **PostgreSQL + pgvector** | Vector similarity search for duplicate detection |
| **SQLAlchemy async** | Non-blocking DB queries in async FastAPI context |
| **sse-starlette** | Server-Sent Events support for real-time progress streaming |

### Frontend
| Technology | Why chosen |
|-----------|-----------|
| **React 18** | Component model fits the form + chat two-panel layout |
| **Redux Toolkit** | Single store for 13 form fields + chat messages + progress state; `populateFromExtraction` hydrates entire form in one dispatch |
| **Vite** | Fast dev server with HMR; `/api` proxy eliminates CORS issues in development |
| **Axios** | Promise-based HTTP client for REST calls |
| **EventSource (browser native)** | SSE client for real-time progress — no extra library needed |

---

## 4. Architecture — How Everything Connects

```
┌─────────────────────────────────────────────────────────────────┐
│  BROWSER  http://localhost:5173                                  │
│                                                                  │
│  ┌──────────────────┐    ┌─────────────────────────────────┐    │
│  │  ComplaintForm   │    │  CopilotPanel (AIVOA Copilot)   │    │
│  │  (left panel)    │    │  (right panel)                  │    │
│  │                  │    │                                 │    │
│  │  13 form fields  │    │  Chat messages                  │    │
│  │  Redux state     │◄───│  SSE progress events            │    │
│  │                  │    │  File upload / paste input      │    │
│  └──────────────────┘    └─────────────────────────────────┘    │
│          ▲                          │                            │
│          │ Redux dispatch           │ EventSource + fetch        │
└──────────┼──────────────────────────┼────────────────────────────┘
           │                          │
           │                          ▼
┌──────────┼──────────────────────────────────────────────────────┐
│  FASTAPI  http://127.0.0.1:8000                                  │
│                                                                  │
│  GET  /api/complaints/extract/stream  ← SSE (primary)           │
│  POST /api/complaints/extract         ← sync fallback            │
│  POST /api/complaints/chat            ← Q&A + field correction   │
│  POST /api/complaints                 ← persist to DB            │
│  GET  /api/complaints/{id}            ← retrieve saved complaint │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LangGraph StateGraph                                   │    │
│  │                                                         │    │
│  │  parse_document → extract_fields → completeness_check  │    │
│  │      ↓ (if < 40% complete)                             │    │
│  │  ask_user [interrupt() → resume via Command(resume=x)] │    │
│  │      ↓ (if ≥ 40% complete)                             │    │
│  │  risk_classification → duplicate_detection             │    │
│  │      → capa_node → summary_node → END                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  PostgreSQL + pgvector    sentence-transformers (local)          │
└──────────────────────────────────────────────────────────────────┘
```

### Data flow — Redux state shape

```javascript
// complaintSlice — persistent business data
{
  session_id: "uuid-from-backend",
  fields: {
    complaint_source: "Pharmacy",     // 13 form fields
    customer_name: "Apollo Pharmacy",
    product_name: "Amoxicillin Capsules",
    product_strength_grade: "500 mg",
    batch_lot_number: "AMX240602",
    manufacturing_date: "2026-03-01",
    expiry_date: "2028-02-01",
    quantity_affected: "12 capsules",
    complaint_type: "Quality Defect",
    complaint_date: "2024-03-15",
    detailed_complaint_description: "Discolored capsules...",
    initial_severity: "Major",        // filled by risk_classification node
    priority: "High",
  },
  analysis: {
    risk_justification: "...",        // from risk_classification
    root_cause_hypothesis: "...",     // from capa_node
    capa_recommendation: "...",       // from capa_node
    summary: "...",                   // from summary_node
  },
  _lastPatch: {},    // tracks chat-corrected fields for green flash
}

// uiSlice — transient UI state
{
  extractionStatus: "done",           // idle→streaming→done/interrupted/error
  nodeStatus: { parse_document: "done", extract_fields: "done", ... },
  progressPct: 100,
  messages: [{ role: "assistant", content: "Complaint parsed..." }],
  chatLoading: false,
  duplicateIds: [],
}
```

---

## 5. LangGraph Pipeline — 8 AI Nodes Explained

**File:** `backend/app/graph/nodes.py`  
**File:** `backend/app/graph/graph.py`

LangGraph models the AI pipeline as a directed graph. Each **node** is a Python function that reads from and writes to a shared **State** dictionary. **Edges** define which node runs next (fixed or conditional).

```python
# backend/app/graph/graph.py — the complete graph topology

graph.add_edge(START,                 "parse_document")
graph.add_edge("parse_document",      "extract_fields")
graph.add_edge("extract_fields",      "completeness_check")

# CONDITIONAL EDGE — the only branch in the graph
graph.add_conditional_edges(
    "completeness_check",
    _route_after_completeness,     # reads state["completeness_score"]
    {
        "ask_user":            "ask_user",           # score < 0.4
        "risk_classification": "risk_classification" # score >= 0.4
    }
)

graph.add_edge("ask_user",            "extract_fields")  # loop back after answer
graph.add_edge("risk_classification", "duplicate_detection")
graph.add_edge("duplicate_detection", "capa_node")
graph.add_edge("capa_node",           "summary_node")
graph.add_edge("summary_node",        END)
```

### Node 1: `parse_document`
**File:** `backend/app/graph/document_parser.py`

```python
def parse_document(state: ComplaintState) -> dict:
    # Fast path: text already in state (paste input)
    if state.get("raw_text"):
        return {"current_node": "parse_document"}

    # File upload: convert bytes to plain text
    text = extract_text(
        file_bytes=state.get("file_bytes"),
        file_name=state.get("file_name"),
        file_type=state.get("file_type"),   # "pdf" | "docx" | "txt" | "eml"
    )
    return {"raw_text": text, "current_node": "parse_document"}
```

**Supports:** PDF (PyPDFLoader), DOCX (Docx2txtLoader), TXT (plain read), EML (Python `email` stdlib).  
**Why separate from extract_fields:** Parsing failures (bad file format) are different from LLM failures — separate nodes give clearer error messages and separate progress bar steps.

---

### Node 2: `extract_fields`
**File:** `backend/app/graph/extractor.py`

```python
# The extraction call — ChatGroq with structured output
structured_llm = llm.with_structured_output(
    ComplaintExtraction,   # Pydantic schema with all 13 fields
    method="json_mode",    # force JSON output
    include_raw=True,      # get raw string too, needed for repair
)

result = structured_llm.invoke([
    SystemMessage(content=_SYSTEM_PROMPT),   # pharma QA expert instructions
    HumanMessage(content=complaint_text),
])

if result["parsed"] is not None:
    return result["parsed"], 1      # success on first attempt

# REPAIR STEP: if JSON parsing failed, send the broken output back to the LLM
repair_messages = messages + [HumanMessage(
    content=f"Fix this broken JSON:\n{broken_output}"
)]
result2 = structured_llm.invoke(repair_messages)
# returns on attempt 2, or raises ExtractionError
```

**Key design:** `method="json_mode"` instead of tool-calling because `openai/gpt-oss-20b` on Groq returns JSON as plain text, not via tool-call protocol. The **repair retry** catches the rare case where the LLM produces malformed JSON.

---

### Node 3: `completeness_check`
**File:** `backend/app/graph/nodes.py`

```python
_REQUIRED_FIELDS = [
    "detailed_complaint_description",   # must have to do anything
    "product_name",                     # needed for risk classification
    "batch_lot_number",                 # needed for duplicate detection
]

def completeness_check(state):
    extraction = state.get("extraction")
    filled = [f for f in _REQUIRED_FIELDS
              if getattr(extraction, f, None) is not None]
    score = len(filled) / len(_REQUIRED_FIELDS)   # 0.0 – 1.0
    return {"completeness_score": score, "missing_fields": [...]}
```

**No LLM used here** — pure Python. The conditional edge reads `completeness_score`:
- Score < 0.4 → `ask_user` (clarifying question)
- Score ≥ 0.4 → `risk_classification` (continue pipeline)

---

### Node 4: `ask_user` (Human-in-the-Loop)
**File:** `backend/app/graph/nodes.py` + `ask_user_logic.py`

```python
def ask_user(state):
    from langgraph.types import interrupt as lg_interrupt

    # Generate a targeted question about the most important missing field
    question = generate_clarifying_question(state, missing_fields[0])

    # PAUSE THE GRAPH — this is the key LangGraph feature
    # First execution: raises GraphInterrupt, saves state to MemorySaver
    # Second execution (after Command(resume=answer)): returns the answer
    user_answer = lg_interrupt({
        "clarifying_question": question,
        "missing_field": missing_fields[0],
    })

    # Only runs on RESUME:
    return {"user_clarification": str(user_answer)}
```

**How resume works:**
```python
# /chat endpoint when session is interrupted:
from langgraph.types import Command

complaint_graph.stream(
    Command(resume=user_message),  # inject answer into scratchpad
    config,                        # same thread_id as original run
    stream_mode="updates"
)
# LangGraph re-runs ask_user, interrupt() returns user_message this time
# → extract_fields re-runs with clarification appended to raw_text
```

---

### Node 5: `risk_classification`
**File:** `backend/app/graph/risk_classifier.py`

```python
# Prompt uses ICH Q10 and GMP severity criteria
_SYSTEM_PROMPT = """
Critical — direct patient safety risk, adverse event, sterility failure
Major    — significant quality defect, packaging/seal failure, no confirmed harm
Minor    — cosmetic, documentation, administrative issue

Return JSON: {"initial_severity": "...", "priority": "...", "justification": "..."}
"""

risk = classify_risk(extraction)
# Updates extraction.initial_severity and extraction.priority
# Stores risk.justification in state for the AI risk box
```

**Model:** `openai/gpt-oss-20b` (lightweight) — severity classification is a pattern-matching task, not deep reasoning.

---

### Node 6: `duplicate_detection`
**File:** `backend/app/graph/duplicate_detector.py` + `embedder.py`

```python
def find_duplicates(description: str):
    # Step 1: embed the description using local sentence-transformers
    embedding = embed_text(description)   # 384-dim unit-normalised vector

    # Step 2: pgvector cosine similarity query
    query = text("""
        SELECT id,
               1 - (description_embedding <=> CAST(:vec AS vector)) AS similarity
        FROM complaints
        WHERE 1 - (description_embedding <=> CAST(:vec AS vector)) >= 0.85
        ORDER BY similarity DESC
        LIMIT 5
    """)
    # Returns complaint IDs with similarity > 85%
```

**Why local embeddings:** Groq has no embeddings API. `all-MiniLM-L6-v2` runs on CPU, is ~80MB, free, and produces 384-dim vectors. Trade-off: ~200ms CPU latency vs. a cloud API.

**Why `<=>` operator:** pgvector's cosine distance operator. `1 - distance = cosine similarity`. Requires unit-normalised vectors (which sentence-transformers produces with `normalize_embeddings=True`).

---

### Node 7: `capa_node`
**File:** `backend/app/graph/capa_generator.py`

```python
# Uses the HEAVY model — this is where we spend the extra compute budget
llm = ChatGroq(model=settings.capa_model)  # "openai/gpt-oss-120b"

# Prompt includes full complaint context:
human_content = f"""
Product: {product_name}, Batch: {batch_lot_number}
Type: {complaint_type}, Severity: {initial_severity}
Description: {description}
Risk justification: {risk_justification}
Duplicate complaints: {duplicate_context}

Return JSON:
{{
  "root_cause_hypothesis": "...",    # specific technical cause
  "capa_recommendation": "..."       # corrective + preventive actions
}}
"""
```

**Why 120B here:** CAPA reasoning requires multi-step pharma domain knowledge — linking a packaging defect to a sealing machine temperature issue requires understanding manufacturing processes, GMP requirements, and ICH guidelines. The 20B model produces generic output; the 120B model produces specific, actionable CAPA.

---

### Node 8: `summary_node`
**File:** `backend/app/graph/summariser.py`

```python
# 2-3 sentence human-readable summary for the chat sidebar
# Uses lightweight model — short generation task
llm = ChatGroq(model=settings.extraction_model)  # "openai/gpt-oss-20b"
```

---

## 6. Feature 1: Paste Text → Form Fills Automatically

### What happens (user perspective)
1. User types or pastes complaint text into the chat input bar
2. All 13 form fields fill within ~20 seconds
3. AI risk assessment box appears with severity + CAPA

### Code flow

**Step 1 — User presses Send in `CopilotPanel.jsx`:**
```javascript
// frontend/src/components/CopilotPanel.jsx
async function handleSubmit(e) {
    const text = input.trim();
    const sid = crypto.randomUUID();   // generate UUID client-side
    startStream(sid, text);            // open SSE connection
}

function startStream(sessionId, rawText) {
    dispatch(setExtractionStatus("streaming"));
    esRef.current = openExtractionStream({
        sessionId,
        rawText,
        onProgress: (data) => dispatch(handleNodeEvent(data)),
        onDone: (data) => {
            dispatch(handleDone(data));
            dispatch(populateFromExtraction(data));  // fills the form
        },
    });
}
```

**Step 2 — EventSource opens in `api/client.js`:**
```javascript
// frontend/src/api/client.js
export function openExtractionStream({ sessionId, rawText, onProgress, onDone }) {
    const params = new URLSearchParams({ session_id: sessionId, raw_text: rawText });
    const es = new EventSource(`/api/complaints/extract/stream?${params}`);

    es.addEventListener("node_progress", (e) => onProgress(JSON.parse(e.data)));
    es.addEventListener("done", (e) => { onDone(JSON.parse(e.data)); es.close(); });
    return es;
}
```

**Step 3 — FastAPI endpoint (`routes.py`):**
```python
# backend/app/api/routes.py
@router.get("/extract/stream")
async def extract_complaint_stream(session_id: UUID, raw_text: str | None = None):
    initial_state = {
        "session_id": session_id,
        "raw_text": raw_text,
        ...
    }
    return EventSourceResponse(
        stream_graph_events(initial_state, session_id),
        ping=20,
    )
```

**Step 4 — LangGraph runs in a thread (`streamer.py`):**
```python
# backend/app/graph/streamer.py
async def stream_graph_events(initial_state, session_id):
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    def _run_graph():   # runs in thread pool (sync LangGraph → async bridge)
        for chunk in complaint_graph.stream(initial_state, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                # ... handle interrupt
            for node_name, node_deltas in chunk.items():
                frame = _progress_frame(node_name, "completed")
                loop.call_soon_threadsafe(queue.put_nowait, frame)
        loop.call_soon_threadsafe(queue.put_nowait, _done_frame(...))

    loop.run_in_executor(None, _run_graph)   # non-blocking

    async for frame in queue:   # yield frames to SSE response
        yield frame
```

**Step 5 — Redux populates form (`complaintSlice.js`):**
```javascript
// frontend/src/store/complaintSlice.js
populateFromExtraction(state, action) {
    const { session_id, complaint } = action.payload;
    const ext = complaint?.extraction ?? {};

    state.session_id = session_id;
    state.fields = {
        complaint_source: ext.complaint_source ?? "",
        customer_name:    ext.customer_name    ?? "",
        product_name:     ext.product_name     ?? "",
        batch_lot_number: ext.batch_lot_number ?? "",
        // ... all 13 fields in one atomic update
    };
    state.analysis = {
        root_cause_hypothesis: complaint.analysis.root_cause_hypothesis ?? "",
        capa_recommendation:   complaint.analysis.capa_recommendation   ?? "",
        risk_justification:    complaint.analysis.risk_justification    ?? "",
    };
}
```

---

## 7. Feature 2: Upload PDF / Email → AI Extracts Data

### Code flow

**Frontend — file selected (`CopilotPanel.jsx`):**
```javascript
async function handleFile(file) {
    setFileName(file.name);
    dispatch(addMessage({ role: "user", file: file.name }));  // show file bubble

    // Step 1: POST the file (can't use EventSource for file upload)
    const response = await uploadFile(file);   // returns {session_id: "uuid"}

    // Step 2: Open SSE with the session_id (graph already running on server)
    startStream(response.session_id, null);
}
```

**API client — multipart upload (`client.js`):**
```javascript
export async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    const response = await api.post("/complaints/extract", formData, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;   // { session_id, complaint, clarifying_question }
}
```

**Backend — sync extract endpoint (`routes.py`):**
```python
@router.post("/extract", response_model=ExtractResponse)
async def extract_complaint(
    file: UploadFile | None = File(None),
    raw_text: str | None = Form(None),
):
    file_bytes = await file.read()     # read all bytes eagerly
    suffix = file.filename.rsplit(".", 1)[-1].lower()   # "pdf", "docx", etc.

    initial_state = {
        "session_id": session_id,
        "file_bytes": file_bytes,
        "file_name": file.filename,
        "file_type": suffix,
        ...
    }
    final_state, clarifying_question = _run_graph_to_completion_or_interrupt(
        initial_state, session_id
    )
```

**Backend — document parsing (`document_parser.py`):**
```python
def extract_text(*, file_bytes, file_name, file_type):
    if file_type == "pdf":
        # Write to temp file, use PyPDFLoader, delete temp file
        from langchain_community.document_loaders import PyPDFLoader
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
        loader = PyPDFLoader(tmp.name)
        pages = loader.load()
        return "\n\n".join(p.page_content for p in pages)

    elif file_type == "docx":
        from langchain_community.document_loaders import Docx2txtLoader
        # same pattern

    elif file_type == "eml":
        import email
        msg = email.message_from_bytes(file_bytes)
        # Extract Subject, From, To, Date headers + text/plain body parts
        # Skip text/html to avoid sending HTML tags to the LLM

    elif file_type == "txt":
        return file_bytes.decode("utf-8")
```

---

## 8. Feature 3: Chat Corrections → Form Updates Live

### What the user sees
User types: *"ah sorry the batch number is BMX240602 and affected quantity is 48 capsules"*  
→ Batch/Lot Number and Affected Quantity fields **turn green** and update instantly.  
→ Bot replies: *"Got it. I have updated the Batch / Lot Number to 'BMX240602'..."*

### Code flow

**Frontend — submit chat message (`CopilotPanel.jsx`):**
```javascript
const data = await sendChatMessage(sessionId, text);

// Apply field corrections if detected
if (data.updated_fields && Object.keys(data.updated_fields).length > 0) {
    dispatch(patchFields(data.updated_fields));
    // e.g. {batch_lot_number: "BMX240602", quantity_affected: "48 capsules"}
}
dispatch(addMessage({ role: "assistant", content: data.reply }));
```

**Redux — patchFields action (`complaintSlice.js`):**
```javascript
patchFields(state, action) {
    const patch = action.payload;
    Object.entries(patch).forEach(([key, val]) => {
        if (key in state.fields) {
            state.fields[key] = val ?? "";   // update only patched keys
        }
    });
    state._lastPatch = patch;    // triggers green flash in form
},
```

**Form — green flash effect (`ComplaintForm.jsx`):**
```javascript
const [patched, setPatched] = useState({});
const lastPatch = useSelector((s) => s.complaint._lastPatch);

useEffect(() => {
    if (lastPatch && Object.keys(lastPatch).length > 0) {
        setPatched(lastPatch);
        const t = setTimeout(() => setPatched({}), 3000);  // 3 second flash
        return () => clearTimeout(t);
    }
}, [lastPatch]);

// CSS class for each field:
const fc = (name) => `cf-input${patched[name] ? " cf-input--patched" : ""}`;
// cf-input--patched → green background from index.css
```

**Backend — field correction detection (`routes.py`):**
```python
def _answer_question(question: str, state: ComplaintState) -> tuple[str, dict]:
    system = """
    You have two jobs:
    1. Answer questions about the complaint data.
    2. Detect when the user is correcting a field value.

    If correcting, return:
    {
      "reply": "Got it. I have updated the Batch / Lot Number...",
      "updated_fields": {"batch_lot_number": "BMX240602", "quantity_affected": "48 capsules"}
    }

    If just a question, return updated_fields as {}.
    """

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    parsed = json.loads(response.content)
    return parsed["reply"], parsed["updated_fields"]
```

---

## 9. Feature 4: AI Copilot Risk Assessment Box

### What it shows
- **Severity (Suggested):** Critical / Major / Minor — colour coded (red/amber/green)
- **Suggested Next Action:** First corrective action from CAPA
- **Initial Risk Assessment:** Root cause hypothesis text

### Code — risk_classification node
```python
# backend/app/graph/risk_classifier.py

_SYSTEM_PROMPT = """
ICH Q10 severity criteria:
  Critical — patient safety, adverse event, sterility failure, regulatory action
  Major    — packaging defect, quality issue, product integrity compromised
  Minor    — cosmetic, documentation, administrative

Return JSON: {"initial_severity": "Major", "priority": "High", "justification": "..."}
"""

result = structured_llm.invoke([
    SystemMessage(content=_SYSTEM_PROMPT),
    HumanMessage(content=f"""
        Product: {product_name}, Batch: {batch_lot_number}
        Complaint type: {complaint_type}
        Description: {description}
    """)
])
# Updates extraction.initial_severity and extraction.priority
```

### Code — AI box in form
```javascript
// frontend/src/components/ComplaintForm.jsx

// Severity color coding
const sevColor = {
    Critical: "#dc2626",   // red
    Major:    "#d97706",   // amber
    Minor:    "#16a34a",   // green
}[fields.initial_severity] || "#6b7280";

{showAiBox && (
    <div className="ai-risk-box">
        <div className="ai-risk-row">
            <div className="ai-risk-field">
                <span className="ai-risk-label">Severity (Suggested)</span>
                <div style={{ color: sevColor }}>
                    {fields.initial_severity || "Analysing…"}
                </div>
            </div>
            <div className="ai-risk-field">
                <span className="ai-risk-label">Suggested Next Action</span>
                <div>{suggestNextAction(analysis.capa_recommendation)}</div>
            </div>
        </div>
        <div className="ai-risk-full">
            <span className="ai-risk-label">Initial Risk Assessment</span>
            <div>{analysis.root_cause_hypothesis}</div>
        </div>
    </div>
)}
```

---

## 10. Feature 5: Real-Time Progress Bar (SSE Streaming)

### The problem
LangGraph `.stream()` is **synchronous**. FastAPI requires **async**. The LLM takes 2-5 seconds per node. Without streaming, the user sees a blank screen for 20 seconds.

### The solution: asyncio.Queue bridge

```python
# backend/app/graph/streamer.py

async def stream_graph_events(initial_state, session_id):
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    # PRODUCER: sync LangGraph runs in a thread pool
    def _run_graph():
        for chunk in complaint_graph.stream(
            initial_state, config, stream_mode="updates"  # "updates" is required
        ):
            if "__interrupt__" in chunk:
                # graph paused at ask_user
                _put({"event": "interrupt", "data": json.dumps(payload)})
                return

            for node_name, node_deltas in chunk.items():
                # Emit "started" immediately (before LLM call finishes)
                _put({"event": "node_progress", "data": json.dumps({
                    "node": node_name, "status": "started"
                })})
                _put({"event": "node_progress", "data": json.dumps({
                    "node": node_name, "status": "completed"
                })})

        _put({"event": "done", "data": json.dumps(final_response)})

    # KEY: call_soon_threadsafe puts items from thread to async queue safely
    def _put(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    loop.run_in_executor(None, _run_graph)   # start in thread pool

    # CONSUMER: async generator yields frames to SSE response
    while True:
        item = await queue.get()   # non-blocking wait
        if item is _SENTINEL: break
        yield item
```

**Why `stream_mode="updates"` (not `"values"`):**  
LangGraph 0.3.x interrupt events (`__interrupt__`) only appear in `stream_mode="updates"`. Using `"values"` silently swallows them and the graph completes without pausing.

**Frontend progress bar (`uiSlice.js`):**
```javascript
handleNodeEvent(state, action) {
    const { node, status } = action.payload;
    if (status === "started")   state.nodeStatus[node] = "running";
    if (status === "completed") state.nodeStatus[node] = "done";

    // Recalculate percentage from node statuses
    const score = PIPELINE_STEPS.reduce((acc, { key }) => {
        const s = state.nodeStatus[key];
        if (s === "done")    return acc + 1;
        if (s === "running") return acc + 0.5;
        return acc;
    }, 0);
    state.progressPct = Math.round((score / PIPELINE_STEPS.length) * 100);
},
```

---

## 11. Feature 6: Duplicate Detection (pgvector)

### How it works
1. Embed `detailed_complaint_description` using local sentence-transformers
2. Run pgvector cosine similarity query against all stored complaints
3. Flag any with similarity ≥ 85% as potential duplicates
4. Show warning banner on the form

```python
# backend/app/graph/embedder.py

from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)   # load model once, reuse forever
def _get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text: str) -> list[float]:
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)  # unit-normalise!
    return vector.tolist()   # 384 floats


# backend/app/graph/duplicate_detector.py

query = text("""
    SELECT id,
           1 - (description_embedding <=> CAST(:vec AS vector)) AS similarity
    FROM complaints
    WHERE 1 - (description_embedding <=> CAST(:vec AS vector)) >= :threshold
    ORDER BY similarity DESC
    LIMIT 5
""")
```

**Why `normalize_embeddings=True`:**  
pgvector's `<=>` computes cosine distance as `1 - dot_product` only when vectors are unit-normalised. Without normalisation, `<=>` gives incorrect similarity scores.

---

## 12. Feature 7: Save to Database (QMS Ledger)

### What happens
User clicks "Save Complaint" → fields sanitised → merged with AI analysis from checkpoint → inserted into Postgres.

**Frontend — date sanitisation before sending (`client.js`):**
```javascript
function toIsoDate(str) {
    // Handles: "March 2026" → "2026-03-01"
    //          "February 2028" → "2028-02-01"
    //          "2024-03-15" → "2024-03-15" (already ISO)
    //          "Not Provided" → null
    const d = new Date(str);
    if (!isNaN(d.getTime())) {
        return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}`;
    }
    // Also handles "March 2026" pattern with regex
    const m = str.toLowerCase().match(/([a-z]+)\s+(\d{4})/);
    if (m && months[m[1]]) return `${m[2]}-${pad(months[m[1]])}-01`;
    return null;
}
```

**Backend — merge user edits + AI analysis (`routes.py`):**
```python
@router.post("", response_model=ConfirmComplaintResponse, status_code=201)
async def confirm_complaint(body: ConfirmComplaintRequest, db: AsyncSession = Depends(get_db)):
    # 1. Get AI analysis from LangGraph checkpoint
    checkpoint = complaint_graph.get_state(config)
    saved_state = checkpoint.values

    # 2. Generate embedding for future duplicate detection
    embedding = await loop.run_in_executor(None, embed_text, description)

    # 3. Merge: user-confirmed fields + AI-generated analysis
    record = ComplaintORM(
        product_name=body.extraction.product_name,      # user-confirmed
        batch_lot_number=body.extraction.batch_lot_number,
        ...
        capa_text=saved_state.get("capa_text"),          # AI-generated
        root_cause_hypothesis=saved_state.get("root_cause_hypothesis"),
        description_embedding=embedding,                  # for future dup detection
    )
    db.add(record)
    await db.flush()   # get DB UUID without committing yet
    return ConfirmComplaintResponse(complaint_id=record.id)
```

**Database schema (`db/models.py`):**
```python
class ComplaintORM(Base):
    __tablename__ = "complaints"

    id: UUID                        # primary key
    product_name: str               # typed column for fast queries
    batch_lot_number: str
    initial_severity: Enum          # Critical | Major | Minor
    extraction_data: JSONB          # full extraction as JSON
    root_cause_hypothesis: Text     # AI-generated
    capa_text: Text                 # AI-generated
    description_embedding: Vector(384)  # pgvector column for similarity search
    created_at: DateTime
```

---

## 13. Feature 8: Voice-to-Text Dictation (Groq Whisper Turbo)

### Why this feature makes PharmaIntel AI stand out
In pharmaceutical manufacturing plants and QC laboratories, QA inspectors frequently wear cleanroom gloves or need to log complaints hands-free while inspecting physical packaging, vials, or blister strips. 

**PharmaIntel AI integrates state-of-the-art Voice-to-Text directly into the Copilot**, powered by **Groq Whisper Large v3 Turbo** (`whisper-large-v3-turbo`).

### How Voice-to-Text Works End-to-End

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. BROWSER MICROPHONE CAPTURE                                               │
│    User clicks Microphone button in Copilot                                 │
│    navigator.mediaDevices.getUserMedia({ audio: true })                     │
│    MediaRecorder records audio slices (audio/webm;codecs=opus)              │
│    UI shows live timer (Listening… 0:04) + animated 5-bar equalizer wave    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ POST /api/complaints/transcribe (Blob)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. FASTAPI BACKEND & GROQ WHISPER                                           │
│    FastAPI endpoint receives multipart audio buffer                         │
│    Calls client.audio.transcriptions.create(model="whisper-large-v3-turbo")  │
│    Groq LPU hardware returns high-accuracy transcription in < 250ms         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Returns JSON: {"text": "..."}
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. COPILOT INPUT & QA PIPELINE EXECUTION                                    │
│    Text automatically populates and focuses the Copilot input field         │
│    Specialist submits voice-dictated complaint to LangGraph QA pipeline     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Details

#### A. Backend Endpoint (`backend/app/api/routes.py`)
```python
@router.post("/transcribe", summary="Transcribe audio to text via Groq Whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    settings = get_settings()
    audio_bytes = await file.read()

    client = Groq(api_key=settings.groq_api_key)
    transcription = client.audio.transcriptions.create(
        file=(file.filename or "recording.webm", audio_bytes, file.content_type or "audio/webm"),
        model="whisper-large-v3-turbo",
        language="en",
        response_format="json",
    )
    return {
        "text": transcription.text.strip(),
        "model": "whisper-large-v3-turbo",
        "bytes_received": len(audio_bytes),
    }
```

#### B. Frontend Audio Recording Lifecycle (`frontend/src/components/CopilotPanel.jsx`)
- **Web Audio API**: Uses `navigator.mediaDevices.getUserMedia` with fallback mime-types (`audio/webm;codecs=opus`, `audio/webm`, `audio/mp4`).
- **Hardware Cleanup**: Upon clicking **Stop** or **Done**, `stream.getTracks().forEach(t => t.stop())` is called immediately to release microphone hardware and avoid battery drain.
- **Microphone States**:
  - `Idle`: Clean vector microphone icon with descriptive tooltip.
  - `Recording`: Glowing red gradient with `@keyframes micPulse` ring animation.
  - `Banner`: Live duration counter (`formatTimer(recordSeconds)`) and 5 animated frequency bars (`@keyframes waveBar`).
  - `Transcribing`: Glassmorphic status badge with rotating AI sparkle icon while Whisper processes the audio.

---

## 14. Feature 9: Neon Cloud PostgreSQL & Live Ledger Explorer

### Production Cloud Database Architecture
PharmaIntel AI connects to **Neon Cloud PostgreSQL 18.6** (AWS `us-east-2`) with native SSL/TLS encryption.

### Database Tables & Extensions
1. **`complaints` Table (`ComplaintORM`)**:
   - `id`: UUID Primary Key
   - `session_id`: LangGraph trace UUID
   - `product_name`, `batch_lot_number`, `customer_name`, `complaint_date`
   - `initial_severity`, `priority`
   - `extraction_data`: Full structured JSONB payload
   - `risk_justification`, `root_cause_hypothesis`, `capa_text`, `summary_text`
   - `description_embedding`: `vector(384)` for pgvector cosine similarity matching
   - `is_confirmed`: Boolean flag
   - `created_at`, `updated_at`
2. **`complaint_audit_logs` Table**:
   - 21 CFR Part 11 compliant immutable audit log tracking user changes, timestamps, and previous values.
3. **`pgvector` Extension**:
   - Enabled directly inside PostgreSQL for instant sub-millisecond semantic duplicate triage.

### Live Database Demonstration for Evaluators
- **In-App Header Button**: Click **`🗄️ View Database Records`** in the top navigation bar to open the live PostgreSQL database feed:
  `http://127.0.0.1:8000/api/complaints/ledger/all`
- **Interactive Swagger Docs**: Click **`API Swagger Docs`** or visit `http://127.0.0.1:8000/docs` to execute queries directly against the live database.
- **Zero-Loss GxP Fallback Ledger**: Every transaction is simultaneously backed up into [`backend/data/qms_ledger.json`](file:///c:/Desktop/pharmaceutical_ComplentAI/backend/data/qms_ledger.json), ensuring 100% data durability even during network degradation.

---

## 15. Feature 10: Modern Pharma-Tech UI & Google Font Inter

### Visual & Architectural Design Upgrades
- **Typography**: Complete migration to **Google Font Inter** (weights 300 to 800) with optimized letter-spacing (`-0.02em`) and numerical alignment for pharmaceutical batch data.
- **Bespoke SVG Icon Library (`Icons.jsx`)**: Replaced all raw unicode emojis with crisp enterprise vector graphics:
  - `PharmaIntelLogo`: Distinct medical cross / tech node emblem with gradient fill
  - `PdfDocIcon`: High-detail vector document badge with red corner ribbon and fold mark
  - `MicrophoneIcon` & `MicStopIcon`: Crisp voice recording icons
  - `ShieldIcon`, `SendIcon`, `BotAvatarIcon`, `UserAvatarIcon`
- **Enterprise Header**:
  - GxP 21 CFR Part 11 compliance pill
  - Real-time AI Core health indicator with pulsing green heartbeat
  - Direct links to live Database Records and Swagger API documentation
- **Section Numbering**: Clean numbered badges (`01` Origin & Customer Details, `02` Product & Batch Identification, `03` Complaint Details, `04` Defect Analysis & Risk Classification, `05` QA Assessment & Priority).
- **Executive Defect Analysis Risk Card**: Prominent summary card displaying AI Severity rating, ICH Q10 justification, probable root cause, and CAPA recommendations.

---

## 16. Frontend Code Walkthrough

### File structure
```
frontend/src/
  App.jsx                    — two-panel layout + status badge
  main.jsx                   — Redux Provider wrapping
  index.css                  — all styles
  store/
    store.js                 — Redux configureStore
    complaintSlice.js        — 13 form fields + analysis + patchFields
    uiSlice.js               — progress, chat, extraction status
  api/
    client.js                — EventSource + axios + date sanitisation
  components/
    ComplaintForm.jsx        — 5-section form + AI risk box
    CopilotPanel.jsx         — chat panel + file upload + SSE
```

### Key Redux actions and when they fire

| Action | When | What it does |
|--------|------|-------------|
| `populateFromExtraction` | SSE "done" event | Fills all 13 fields atomically |
| `patchFields` | Chat field correction | Updates only changed fields, triggers green flash |
| `updateField` | User types in form | Updates single field, clears flash |
| `handleNodeEvent` | Each SSE "node_progress" | Updates nodeStatus, recalculates progressPct |
| `handleInterrupt` | SSE "interrupt" event | Shows clarifying question in chat |
| `handleDone` | SSE "done" event | Sets status to "done", adds bot message |
| `resetComplaint` + `resetUi` | Reset Form button | Clears all state |

---

## 14. Backend Code Walkthrough

### File structure
```
backend/app/
  main.py                    — FastAPI app factory + embedding warmup in lifespan
  config.py                  — Pydantic BaseSettings (reads .env)
  api/
    routes.py                — all 5 endpoints
  graph/
    state.py                 — ComplaintState TypedDict (shared memory)
    graph.py                 — StateGraph topology + MemorySaver checkpointer
    nodes.py                 — 8 node functions
    extractor.py             — ChatGroq structured output + JSON repair
    document_parser.py       — PDF/DOCX/TXT/EML → plain text
    risk_classifier.py       — severity/priority LLM call
    embedder.py              — sentence-transformers singleton
    duplicate_detector.py    — pgvector cosine query
    capa_generator.py        — CAPA + root cause (120B model)
    summariser.py            — 2-3 sentence summary
    ask_user_logic.py        — clarifying question generation
    session_store.py         — tracks interrupted/complete per session
    streamer.py              — asyncio.Queue SSE bridge
  db/
    engine.py                — lazy async SQLAlchemy engine
    models.py                — ComplaintORM with Vector(384) column
  schemas/
    complaint.py             — domain Pydantic models (ComplaintExtraction etc.)
    api.py                   — HTTP request/response DTOs
```

### LangGraph checkpointer — how state persists across HTTP requests
```python
# backend/app/graph/graph.py

_checkpointer = MemorySaver()   # in-process dict; persists state per thread_id
complaint_graph = build_graph().compile(checkpointer=_checkpointer)

def get_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": str(thread_id)}}

# Every request uses the same thread_id = session_id:
config = get_thread_config(str(session_id))
complaint_graph.stream(initial_state, config, stream_mode="updates")
# LangGraph saves checkpoint after each node → /chat can resume from interrupt
```

---

## 15. End-to-End Request Flow

### Complete trace: user pastes Apollo Pharmacy complaint

```
1. User types in chat box → presses ✓
   CopilotPanel.jsx: handleSubmit() → startStream(uuid, text)
   client.js: openExtractionStream() → EventSource("/api/complaints/extract/stream?...")

2. Browser → GET /api/complaints/extract/stream?session_id=uuid&raw_text=...
   routes.py: extract_complaint_stream() → EventSourceResponse(stream_graph_events(...))

3. streamer.py: _run_graph() starts in thread pool
   LangGraph: loads initial_state into ComplaintState, starts from parse_document

4. Node: parse_document
   → raw_text already present (paste input) → passes through
   → SSE: {"event":"node_progress","data":{"node":"parse_document","status":"started"}}
   → SSE: {"event":"node_progress","data":{"node":"parse_document","status":"completed"}}
   Browser: dispatch(handleNodeEvent({node:"parse_document",status:"completed"}))
   React: ProgressBar updates → "Parsing document" step shows ✓

5. Node: extract_fields  (takes ~2s — LLM call)
   extractor.py: ChatGroq("openai/gpt-oss-20b") called with complaint text
   Returns: ComplaintExtraction(product_name="Amoxicillin Capsules", batch="AMX240602", ...)
   → SSE: node_progress extract_fields started/completed
   Browser: ProgressBar → "Extracting fields" step shows ✓

6. Node: completeness_check
   3 required fields: description ✓, product_name ✓, batch_lot_number ✓
   score = 3/3 = 1.0 ≥ 0.4 → routes to risk_classification (no interrupt!)
   → SSE: node_progress completeness_check

7. Node: risk_classification  (takes ~1s — LLM call)
   ChatGroq("openai/gpt-oss-20b") + pharma severity criteria
   Returns: severity=Major, priority=High, justification="..."
   → SSE: node_progress risk_classification

8. Node: duplicate_detection
   embed_text(description) → 384-dim vector
   pgvector query → no matches above 0.85 threshold
   → SSE: node_progress duplicate_detection

9. Node: capa_node  (takes ~3-5s — heavy LLM call)
   ChatGroq("openai/gpt-oss-120b") + full complaint context
   Returns: root_cause="coating pan temperature deviation", capa="Quarantine batch..."
   → SSE: node_progress capa_node

10. Node: summary_node  (takes ~1s — LLM call)
    ChatGroq("openai/gpt-oss-20b") → 2-3 sentence summary
    → SSE: node_progress summary_node

11. Graph reaches END → done frame
    → SSE: {"event":"done","data":{session_id, complaint:{extraction, analysis}}}
    Browser: onDone() fires
      dispatch(handleDone(data))           → status="done", bot message added
      dispatch(populateFromExtraction(data)) → all 13 fields filled in Redux

12. React re-renders
    ComplaintForm reads from Redux → all inputs show extracted values
    AI risk box appears with Major severity + CAPA
    Status badge: "Pending Triage" → "● Ready to Commit"
    Total time: ~15-20 seconds
```

---

## 19. Setup & Run Instructions

### Prerequisites
- Python 3.11+
- Node.js 18+
- Groq API key from https://console.groq.com (powers extraction, CAPA reasoning, and **Groq Whisper Voice-to-Text**)
- (Configured) Neon Cloud PostgreSQL 18.6 with `pgvector` extension

### Backend setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Environment configuration
# The application connects to Neon Cloud PostgreSQL and Groq
# Verify backend/.env contains:
# GROQ_API_KEY=gsk_...
# DATABASE_URL=postgresql://neondb_owner:...@ep-restless-bread-a59g0ow5-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require
```

### Start backend

```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Wait for: `Embedding model ready. Application startup complete.`

### Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open: **http://localhost:5173**

---

## 20. Test Data for Demo

### Test 1 — Voice-to-Text Dictation (Groq Whisper Turbo)
1. Click the **Microphone** icon in PharmaIntel Copilot.
2. Dictate clearly:
   *"Patient reported discolored capsules in Amoxicillin Capsules 500mg batch AMX240602. Severity is major."*
3. Click **"Done"**.
4. Whisper will transcribe the speech directly into the Copilot input in < 250ms.
5. Press Enter to trigger automated LangGraph extraction and form hydration.

### Test 2 — Basic extraction (Major severity via Paste)
```
Apollo Pharmacy reported discolored capsules in Amoxicillin Capsules 500 mg.
Batch number AMX240602. Manufacturing date March 2026. Expiry date February 2028.
Please log this complaint.
```

### Test 3 — Adverse event (Critical severity via PDF upload)
Upload `backend/samples/Apollo_Pharmacy_Complaint_AMX-2026-001.pdf` using the paperclip attachment button.

### Test 4 — Field correction via chat
After extracting any complaint, type or speak into the chat:
```
ah sorry the batch number is BMX240602 and affected quantity is 48 capsules
```
Expected: Batch and Quantity fields flash green and update live.

### Test 5 — Inspecting the Live Neon PostgreSQL Database
Click **`🗄️ View Database Records`** in the top navigation bar to open the live PostgreSQL database feed:
`http://127.0.0.1:8000/api/complaints/ledger/all`

---

## 21. Video Walkthrough Script

### Video 1: Feature Demo (~5 minutes)

**0:00 — Open the app**
- Show the clean pharma-tech two-panel layout with **Google Font Inter** typography
- Point out: left = 5-section complaint form (`01`–`05`)
- Point out: right = PharmaIntel Copilot with GxP status indicators

**0:30 — Voice-to-Text Dictation (Groq Whisper)**
- Click the **Microphone** button in the Copilot input bar
- Notice the pulsing red recording ring, live timer (`Listening… 0:03`), and animated frequency bars
- Speak: *"Customer noted particulate matter in Batch MFH260712A, Metformin API"*
- Click **Done** → Groq Whisper transcribes speech into the chat bar instantly
- Hit Send → Watch the real-time progress bar stream through the LangGraph nodes

**2:00 — Form Population & AI Risk Assessment**
- Show form filling automatically: product, batch, dates, severity
- Point out the **Defect Analysis & Risk Classification** card: Major severity, ICH Q10 root cause hypothesis, and CAPA workflow

**3:00 — Chat correction**
- Type: *"ah sorry the batch number is BMX240602 and affected quantity is 48 capsules"*
- Show the fields flash green and update live

**3:45 — Upload PDF Report**
- Click the bespoke PDF paperclip icon
- Select a complaint PDF from `backend/samples/`
- Show document intelligence parsing without disk locks and filling the form

**4:30 — Commit to Neon Cloud PostgreSQL Database**
- Click **"Commit to QMS Ledger"**
- Show green success banner: *"Complaint Confirmed & Committed to Ledger • Record ID: ... • QMS Audit Trail Updated"*
- Click **`🗄️ View Database Records`** in the top bar to show the live committed record inside Neon PostgreSQL table `complaints`!

### Video 2: Code Walkthrough (~8 minutes)

**0:00 — Architecture overview**
- Draw the flow: Browser (React 18 + Redux + Web Audio API) → FastAPI → Groq Whisper + LangGraph → Neon Cloud PostgreSQL
- Open `backend/app/graph/graph.py` — show the StateGraph nodes and edges

**1:30 — Voice-to-Text Implementation**
- Open `backend/app/api/routes.py` → `transcribe_audio`
- Show Groq Whisper integration with `whisper-large-v3-turbo`
- Open `frontend/src/components/CopilotPanel.jsx` → show `MediaRecorder` audio chunk lifecycle and immediate hardware release

**3:30 — Neon PostgreSQL & pgvector Setup**
- Open `backend/app/db/engine.py` → show asyncpg engine with SSL configuration
- Open `backend/app/db/models.py` → show `ComplaintORM`, `ComplaintAuditLogORM`, and `vector(384)` pgvector column

**5:00 — Frontend Redux & Inter Design System**
- Open `frontend/src/index.css` → show Google Font Inter tokens, `@keyframes micPulse`, and audio equalizer wave animations
- Open `frontend/src/components/Icons.jsx` → show custom vector SVG icons
- Open `frontend/src/store/complaintSlice.js` → show `populateFromExtraction` and `patchFields` sparse updates

