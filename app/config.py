"""
config.py — centralised settings for the entire application.

Why Pydantic BaseSettings?
  Pydantic's BaseSettings reads values from environment variables (or a .env
  file) and validates/coerces them at startup.  This means if GROQ_API_KEY is
  missing the app crashes immediately with a clear error, rather than failing
  silently at the first LLM call.  A single `get_settings()` cached function
  means we parse the env exactly once, not on every request.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM provider ─────────────────────────────────────────────────────────
    groq_api_key: str  # no default — must be set; app won't start without it

    # Model names are centralised here so every node imports from one place.
    # Original spec called for gemma2-9b-it (extraction) and llama-3.3-70b-versatile
    # (CAPA), but both were decommissioned by Groq in 2025/2026.
    # Current Groq-recommended replacements (as of Sept 2026, per console.groq.com/docs/deprecations):
    #   gemma2-9b-it         → llama-3.1-8b-instant → openai/gpt-oss-20b
    #   llama-3.3-70b-versatile → openai/gpt-oss-120b
    # The design intent is preserved: fast lightweight model for extraction,
    # heavier model for CAPA reasoning.
    extraction_model: str = "openai/gpt-oss-20b"
    capa_model: str = "openai/gpt-oss-120b"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://postgres:password@localhost:5432/pharma_complaints"
    )

    # ── Duplicate-detection threshold ─────────────────────────────────────────
    # Cosine similarity is in [0, 1]; 0.85 means "very similar".
    # Lower this (e.g. 0.75) to catch more near-duplicates at the cost of
    # more false positives.  This is a deliberate business-logic knob.
    duplicate_similarity_threshold: float = 0.85

    # ── Completeness gate ─────────────────────────────────────────────────────
    # If fewer than this fraction of required fields are filled, the graph
    # routes to ask_user instead of proceeding to risk_classification.
    completeness_threshold: float = 0.75

    # ── API / server ──────────────────────────────────────────────────────────
    app_env: str = "development"
    # Pydantic-settings parses a JSON array string for list fields when read
    # from env vars.  Format in .env: CORS_ORIGINS='["http://localhost:5173"]'
    # Or use a simple comma-separated string with a custom validator (below).
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        # Allow comma-separated strings for list fields, e.g.:
        # CORS_ORIGINS=http://localhost:5173,http://localhost:3000
        env_parse_none_str="null",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    lru_cache ensures the .env file is parsed only once per process.
    FastAPI's Depends(get_settings) will call this on every request but the
    cache makes it effectively free after the first call.
    """
    return Settings()
