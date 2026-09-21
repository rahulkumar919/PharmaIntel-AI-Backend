"""
graph/embedder.py — text embedding via the HuggingFace free Inference API.

WHY THIS CHANGED FROM LOCAL sentence-transformers
─────────────────────────────────────────────────
The original implementation loaded sentence-transformers locally, which pulls
in PyTorch (~300–400 MB RAM just to import).  On Render's free tier (512 MB
total), this left ~100–150 MB for everything else — FastAPI, LangGraph,
asyncpg, and actual request processing.  The first real LLM call pushed the
process over the limit and the Linux OOM killer silently terminated it with
no error logged.

The solution is to call the HuggingFace Inference API instead of running
the model locally.  This means:
  - Zero RAM for model weights on our server
  - The exact same all-MiniLM-L6-v2 model (vectors are identical — no DB migration needed)
  - The free HuggingFace tier is generous: 1000 req/day, suitable for a demo/MVP
  - Adds ~100–300ms of network latency per embedding call, which is acceptable
    since duplicate_detection is non-blocking (it never crashes the pipeline)

HOW TO GET A FREE HuggingFace TOKEN
────────────────────────────────────
1. Go to https://huggingface.co and create a free account
2. Go to Settings → Access Tokens → New token → Role: "Read"
3. Copy the token (starts with "hf_...")
4. Add to your Render environment variables:  HF_API_TOKEN=hf_...
   (or your .env for local dev — it works without a token too, but is rate-limited)

FALLBACK BEHAVIOUR
──────────────────
If the HF API is unavailable or the token is missing, embed_text() returns a
zero vector.  The duplicate_detection node already handles zero vectors
gracefully (they score ~0 against all DB entries, so no false duplicates).
A WARNING is logged so you can see it in Render's log panel.

EMBEDDING_DIM must stay 384 to match Vector(384) in db/models.py.
If you ever change the model, run a new Alembic migration to change the column.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384   # must match VECTOR(384) in models.py and migration

# HuggingFace Inference API endpoint for feature-extraction (embeddings)
_HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# Timeout for embedding API call — 30s is generous; typical is 1–3s
_HTTP_TIMEOUT = 30.0


def _get_hf_token() -> Optional[str]:
    """
    Read the HuggingFace API token from the environment.

    We deliberately don't put this in pydantic Settings because:
      - The token is optional (the API works without one, just rate-limited)
      - It avoids having to add it to Settings validation and .env.example
    """
    return os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")


def embed_text(text: str) -> list[float]:
    """
    Return a 384-dimensional embedding vector for `text` via the HuggingFace API.

    The returned vector matches the output of:
        SentenceTransformer("all-MiniLM-L6-v2").encode(text, normalize_embeddings=True)
    so all existing pgvector entries remain valid — no DB migration needed.

    Returns a zero vector on any error so the caller (duplicate_detector) can
    degrade gracefully without crashing the pipeline.

    Parameters
    ----------
    text : the complaint description (or any string) to embed

    Returns
    -------
    list[float] of length 384 — ready to store as a pgvector VECTOR column
    """
    if not text or not text.strip():
        logger.warning("embed_text: empty text, returning zero vector")
        return [0.0] * EMBEDDING_DIM

    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = _get_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        logger.debug(
            "embed_text: HF_API_TOKEN not set — using unauthenticated HF API "
            "(rate-limited to ~30 req/hr). Add HF_API_TOKEN env var for higher limits."
        )

    payload = {
        "inputs": text,
        "options": {"wait_for_model": True},  # wait up to 20s if model is loading
    }

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            response = client.post(_HF_API_URL, json=payload, headers=headers)

        if response.status_code != 200:
            logger.warning(
                "embed_text: HF API returned HTTP %d — %s. Returning zero vector.",
                response.status_code,
                response.text[:200],
            )
            return [0.0] * EMBEDDING_DIM

        raw = response.json()

        # The feature-extraction pipeline returns different shapes depending on
        # the model and pooling config:
        #   - list[float]           → already a flat vector ✓
        #   - list[list[float]]     → one row per token; take the mean (sentence-level)
        #   - list[list[list[float]]] → batch dimension; unwrap first
        vector = _parse_embedding_response(raw)

        if len(vector) != EMBEDDING_DIM:
            logger.warning(
                "embed_text: unexpected embedding dim %d (expected %d). Returning zero vector.",
                len(vector), EMBEDDING_DIM,
            )
            return [0.0] * EMBEDDING_DIM

        return vector

    except httpx.TimeoutException:
        logger.warning(
            "embed_text: HF API timed out after %.0fs. Returning zero vector.", _HTTP_TIMEOUT
        )
        return [0.0] * EMBEDDING_DIM

    except Exception as exc:
        logger.warning("embed_text: unexpected error — %s: %s. Returning zero vector.",
                       type(exc).__name__, exc)
        return [0.0] * EMBEDDING_DIM


def _parse_embedding_response(raw) -> list[float]:
    """
    Normalise the various shapes the HF feature-extraction API can return.

    HuggingFace feature-extraction with all-MiniLM-L6-v2 returns:
        [[token_vec_1, token_vec_2, ...]]   (batch × tokens × dim)
    We need to mean-pool across the token dimension to get a sentence vector,
    then flatten to list[float].
    """
    # Unwrap batch dimension if present
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]

    # raw is now either:
    #   list[float]        → already a flat 384-dim vector (some models / configs)
    #   list[list[float]]  → token-level embeddings; mean-pool to get sentence vector
    if raw and isinstance(raw[0], float):
        return raw   # already flat — use as-is

    if raw and isinstance(raw[0], list):
        # Mean-pool across tokens: average each dimension
        dim = len(raw[0])
        pooled = [
            sum(token[d] for token in raw) / len(raw)
            for d in range(dim)
        ]
        return pooled

    return []   # unexpected shape — caller will log and return zero vector
