"""
graph/embedder.py — sentence-transformers singleton for generating text embeddings.

Why sentence-transformers (not a cloud API)?
  Groq does not offer an embeddings endpoint — it is inference-only.
  We use sentence-transformers/all-MiniLM-L6-v2 which:
    - Runs entirely locally, no extra API key
    - Is ~80 MB on first download (cached in ~/.cache/huggingface after that)
    - Produces 384-dimensional vectors matching our VECTOR(384) DB column
    - Runs in ~20-200ms on CPU depending on text length
  Trade-off vs. a cloud embedding API (e.g. OpenAI text-embedding-3-small):
    + Free, no per-call cost, no rate limits, no extra credentials
    - Slightly lower quality for domain-specific pharma text
    - First call takes ~2s while the model loads from disk

Singleton pattern:
  Loading the SentenceTransformer model is expensive (~1s + file I/O).
  We load it once into a module-level variable (_model) and reuse it for
  every embed_text() call in the process lifetime.  Thread-safe because
  SentenceTransformer's encode() releases the GIL during the actual
  inference computation.

  In production, the lifespan hook in main.py should call _get_model() at
  startup so the first request doesn't pay the cold-start cost.

EMBEDDING_DIM must match Vector(384) in db/models.py and the migration.
If you ever swap models, update both constants together.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384   # must match VECTOR(384) in models.py and migration


@lru_cache(maxsize=1)
def _get_model() -> Any:
    """
    Load the SentenceTransformer model once and cache it.

    lru_cache(maxsize=1) is the idiomatic singleton for module-level objects
    that are expensive to initialise — the function body runs exactly once
    per process, regardless of how many times _get_model() is called.

    Why not a plain module-level variable?
      A module-level `_model = SentenceTransformer(...)` would execute at
      import time, adding ~1s to every cold start even for requests that
      never use duplicate_detection.  The lazy lru_cache pattern loads the
      model only on the first embed_text() call.
    """
    # Lazy import: only pay the torch/transformers import cost when the
    # embedder is actually needed (not on every cold start).
    from sentence_transformers import SentenceTransformer   # noqa: PLC0415

    logger.info("embedder: loading model '%s' (first call only)…", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info(
        "embedder: model loaded, embedding dim=%d",
        model.get_sentence_embedding_dimension(),
    )
    return model


def embed_text(text: str) -> list[float]:
    """
    Return a 384-dimensional unit-normalised embedding vector for `text`.

    Normalisation (normalize_embeddings=True) is required for cosine similarity
    to equal the dot product.  pgvector's <=> operator computes cosine distance
    as 1 - dot_product for unit vectors, which is equivalent to true cosine
    distance only when vectors are normalised.

    Parameters
    ----------
    text : the complaint description (or any string) to embed

    Returns
    -------
    list[float] of length 384 — ready to store as a pgvector VECTOR column
    """
    if not text or not text.strip():
        # Return a zero vector for empty text so we never store None in the
        # embedding column — a zero vector will score ~0 against all other
        # vectors, effectively disabling duplicate detection for this record.
        logger.warning("embed_text: empty text, returning zero vector")
        return [0.0] * EMBEDDING_DIM

    model = _get_model()

    # encode() returns a numpy array; convert to plain Python list for JSON
    # serialisability and SQLAlchemy / pgvector compatibility.
    vector: np.ndarray = model.encode(
        text,
        normalize_embeddings=True,   # unit-normalise so cosine sim = dot product
        show_progress_bar=False,
    )
    return vector.tolist()
