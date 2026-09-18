"""
main.py — FastAPI application entry point.

This file's only job is to create the FastAPI app, attach middleware, and mount
routers.  No business logic lives here — that keeps main.py readable at a glance
and easy to reason about when onboarding new team members.

Lifespan context manager (startup/shutdown):
  FastAPI 0.93+ recommends using the `lifespan` parameter instead of the
  deprecated @app.on_event decorators.  We use it here to run any startup
  work (e.g. DB connection pool warm-up in Phase 4) and teardown cleanly.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import router

# Configure a basic logger for the whole application.
# In production you'd swap this for a structured JSON logger (e.g. structlog).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm up the embedding model and log model choices.
    Shutdown: nothing special needed (connections close via GC).

    Why warm up the embedding model here?
      sentence-transformers loads ~80MB of weights on first call.  If we don't
      warm up at startup, the first /extract request that triggers
      duplicate_detection will have ~2s of unexpected latency.  Loading at
      startup makes that cost predictable and visible in the server logs.
    """
    import asyncio
    settings = get_settings()
    logger.info("Starting PharmaIntel AI API [env=%s]", settings.app_env)
    logger.info("Extraction model : %s", settings.extraction_model)
    logger.info("CAPA model       : %s", settings.capa_model)

    # Warm up embedding model in a thread so we don't block the event loop
    # during the synchronous model load.
    logger.info("Warming up sentence-transformers embedding model…")
    loop = asyncio.get_event_loop()
    try:
        from app.graph.embedder import _get_model
        await loop.run_in_executor(None, _get_model)
        logger.info("Embedding model ready.")
    except Exception as exc:
        # Non-fatal — the app still starts, duplicate detection degrades gracefully
        logger.warning("Embedding model warm-up failed: %s", exc)

    yield  # ← application serves requests here

    logger.info("Shutting down PharmaIntel AI API")


def create_app() -> FastAPI:
    """
    Application factory pattern — returns a configured FastAPI instance.

    Using a factory (instead of a module-level `app = FastAPI()`) makes it
    easy to create a fresh app instance in tests without side-effects from
    import-time code.
    """
    settings = get_settings()

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

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Allow the React dev server (Vite default: 5173) and CRA default (3000).
    # In production this list should be tightened to the actual frontend domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
