"""
main.py — FastAPI application entry point.

This file's only job is to create the FastAPI app, attach middleware, and mount
routers.  No business logic lives here — that keeps main.py readable at a glance
and easy to reason about when onboarding new team members.

Lifespan context manager (startup/shutdown):
  FastAPI 0.93+ recommends using the `lifespan` parameter instead of the
  deprecated @app.on_event decorators.  We use it here to run any startup
  work (e.g. DB connection pool warm-up in Phase 4) and teardown cleanly.

Logging strategy:
  - Development: human-readable  "timestamp | LEVEL | logger | message"
  - Production:  JSON lines  — each record is a single JSON object so
    Render / Datadog / CloudWatch log drains can filter by field.

Middleware stack (applied bottom-to-top by Starlette):
  1. CORSMiddleware  — outermost, handles preflight before anything else
  2. RequestIDMiddleware — stamps every request with a UUID4, adds
     X-Request-ID header to responses, injects into logging context
"""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.api.routes import router


# ── Logging setup ─────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record — machine-parseable on Render/cloud."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        log_obj = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        # Carry any extra fields attached via logger.info(..., extra={...})
        for key, val in record.__dict__.items():
            if key not in (
                "args", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message",
                "module", "msecs", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread",
                "threadName",
            ) and not key.startswith("_"):
                log_obj[key] = val
        return json.dumps(log_obj)


def _configure_logging(app_env: str) -> None:
    """Set up root logger once at startup."""
    handler = logging.StreamHandler()
    if app_env == "production":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ── Request-ID middleware ─────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Stamp every request with a UUID4 `request_id`.

    - Reads X-Request-ID from the incoming request (so callers can trace
      their own IDs end-to-end); falls back to a fresh UUID4.
    - Attaches the ID to request.state so route handlers can log it.
    - Adds X-Request-ID to every response header so the browser/curl can
      correlate a response with the server log line.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: log model choices and memory.
    Shutdown: nothing special needed (connections close via GC).

    Embedding model warm-up was removed: embeddings are now handled by the
    HuggingFace Inference API (see embedder.py) so there is no local
    PyTorch model to load.  This saves ~300-400 MB RAM on startup.
    """
    settings = get_settings()
    logger.info("Starting PharmaIntel AI API [env=%s]", settings.app_env)
    logger.info("Extraction model : %s", settings.extraction_model)
    logger.info("CAPA model       : %s", settings.capa_model)
    logger.info("Embeddings       : HuggingFace Inference API (all-MiniLM-L6-v2, no local weights)")

    # Log current RSS memory so we can see the post-import RAM baseline.
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger.info("Startup RSS memory: %.1f MB", rss_mb)
    except Exception:
        pass  # resource module not available on Windows

    yield  # ← application opens port and serves requests

    logger.info("Shutting down PharmaIntel AI API")


def create_app() -> FastAPI:
    """
    Application factory pattern — returns a configured FastAPI instance.

    Using a factory (instead of a module-level `app = FastAPI()`) makes it
    easy to create a fresh app instance in tests without side-effects from
    import-time code.
    """
    settings = get_settings()
    _configure_logging(settings.app_env)

    app = FastAPI(
        title="PharmaIntel AI",
        description=(
            "AI-powered customer complaint management for pharmaceutical "
            "manufacturing — API & FDF quality assurance."
        ),
        version="0.1.0",
        lifespan=lifespan,
        # Disable docs in production — internal tool, no public API exposure
        docs_url="/docs" if settings.app_env == "development" else None,
        redoc_url="/redoc" if settings.app_env == "development" else None,
    )

    # ── Middleware stack ───────────────────────────────────────────────────────
    # Starlette applies middleware in reverse-registration order, so CORS
    # (outermost) must be added LAST so it wraps everything.
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=r"https://.*(\.vercel\.app|\.onrender\.com)",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ───────────────────────────────────────────────
    # Catches any unhandled exception that escapes route handlers.
    # Returns a clean JSON error (never a raw Python traceback) and logs the
    # full stack trace with the request_id so it can be found in the log drain.
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            "Unhandled exception [request_id=%s] %s: %s\n%s",
            request_id,
            type(exc).__name__,
            exc,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id,
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Health checks ─────────────────────────────────────────────────────────
    @app.get("/health", tags=["ops"])
    async def health():
        """Liveness probe — just confirms the process is alive."""
        return {"status": "ok"}

    @app.get("/health/ready", tags=["ops"])
    async def health_ready():
        """
        Readiness probe — checks that the DB connection pool is functional.
        Returns 503 if the DB is unreachable so load balancers can route away.
        """
        from app.db.engine import get_async_session_factory
        from sqlalchemy import text
        try:
            async with get_async_session_factory()() as session:
                await session.execute(text("SELECT 1"))
            return {"status": "ready", "db": "ok"}
        except Exception as exc:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={"status": "not ready", "db": str(exc)},
            )

    return app


# Create the module-level app instance that uvicorn picks up via
#   uvicorn app.main:app --reload
app = create_app()
