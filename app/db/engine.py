"""
db/engine.py — SQLAlchemy async engine and session factory.

Why lazy initialisation?
  Creating the engine at import time causes SQLAlchemy to resolve the DB dialect
  immediately.  This crashes the server on startup when Postgres is not running
  (common in development).  The lazy pattern means the engine is created on the
  first DB request — the app starts fine without Postgres, and only DB-dependent
  routes return a 500.

Why async (asyncpg)?
  FastAPI is async.  A synchronous engine would block the event loop on every
  query, defeating FastAPI's concurrency model entirely.  asyncpg is the
  standard high-performance async driver for Postgres.
"""

from __future__ import annotations

from typing import AsyncGenerator

# ── Lazy singletons ───────────────────────────────────────────────────────────
_engine = None
_factory = None


def _init():
    """Create engine + session factory on first call; cache for reuse."""
    global _engine, _factory
    if _engine is None:
        import re
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from app.config import get_settings

        s = get_settings()
        raw_url = s.database_url
        url = raw_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)

        clean_url = re.sub(r'\?.*$', '', url)
        connect_args = {
            "command_timeout": 20.0,
        }
        if "neon.tech" in raw_url or "ssl" in raw_url:
            connect_args["ssl"] = True

        _engine = create_async_engine(
            clean_url,
            echo=(s.app_env == "development"),
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        _factory = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _engine, _factory


async def get_db() -> AsyncGenerator:
    """
    FastAPI Depends() — yields one session per request, auto-commits/rolls back.
    If database is offline or unconfigured, gracefully yields None so routes can
    use local ledger fallback without crashing with a 500 error.
    """
    import logging
    _log = logging.getLogger(__name__)

    try:
        _, factory = _init()
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    except Exception as exc:
        _log.warning("Database unavailable (%s) — using fallback ledger", exc)
        yield None


def get_async_session_factory():
    """Return the session factory (used by seed_data.py and tests)."""
    return _init()[1]
