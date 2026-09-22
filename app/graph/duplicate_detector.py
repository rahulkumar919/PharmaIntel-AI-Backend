"""
graph/duplicate_detector.py — pgvector cosine-similarity search helper.

Why a separate module?
  The duplicate_detection node needs to make a synchronous DB call from inside
  a synchronous LangGraph node function.  All the async/sync bridging and the
  raw SQL construction lives here so the node itself stays a thin wrapper.

Query design:
  We use pgvector's <=> operator (cosine distance) in a raw SQL query rather
  than SQLAlchemy ORM for two reasons:
    1. The pgvector SQLAlchemy integration works differently across pgvector
       versions; raw SQL is always stable.
    2. ORDER BY with LIMIT is simpler to express with a literal SQL string
       than with ORM constructs when using a custom distance operator.

  Cosine distance (<=>) is in [0, 2].  We convert to cosine similarity with:
      similarity = 1 - distance
  A threshold of 0.85 (configurable) means "vectors are 85% similar".

Sync/async bridge:
  LangGraph node functions are synchronous.  The DB engine is async.
  We use asyncio.get_event_loop().run_until_complete() to call the async
  DB function from sync context.
  ASSUMPTION: The graph always runs in the main thread's event loop.
  In a multi-threaded deployment, use asyncio.run() in a new thread instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import UUID

from sqlalchemy import text

from app.config import get_settings
from app.db.engine import get_async_session_factory
from app.graph.embedder import embed_text

logger = logging.getLogger(__name__)


def find_duplicates(description: str) -> tuple[list[UUID], list[float]]:
    """
    Embed `description` and run a cosine-similarity search against the DB.

    Returns
    -------
    (duplicate_ids, scores) where each score is the cosine similarity (0-1)
    and duplicates are ordered from most to least similar.

    Only records with similarity >= settings.duplicate_similarity_threshold
    are returned.  Returns empty lists if the DB is unavailable or description
    is empty (graceful degradation — duplicate detection is non-blocking).
    """
    if not description or not description.strip():
        return [], []

    try:
        # Generate the embedding synchronously
        embedding = embed_text(description)
        if not embedding or all(abs(v) < 1e-6 for v in embedding):
            logger.info("find_duplicates: embedding is empty or zero vector — skipping duplicate search")
            return [], []

        # Run the async DB query via the event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an async context (e.g. running in an async test).
            # Create a new loop in a thread to avoid "event loop is running" error.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _query_db(embedding))
                return future.result()
        else:
            return loop.run_until_complete(_query_db(embedding))

    except Exception as exc:
        # Duplicate detection is best-effort — never crash the pipeline
        logger.warning("find_duplicates: error during DB search: %s", exc)
        return [], []


async def _query_db(
    embedding: list[float],
) -> tuple[list[UUID], list[float]]:
    """
    Run the pgvector cosine-similarity query and return filtered results.

    SQL explanation:
      1 - (description_embedding <=> :vec) converts cosine distance to
      cosine similarity.  We filter on the similarity value (>= threshold)
      rather than on the distance to keep the intent readable.
      LIMIT 5 caps results; in practice the threshold filter usually reduces
      this to 0-2 matches.
    """
    import math
    settings = get_settings()
    threshold = settings.duplicate_similarity_threshold

    # Format the embedding as a Postgres vector literal: '[0.1, 0.2, ...]'
    vec_literal = "[" + ",".join(str(v) for v in embedding) + "]"

    query = text("""
        SELECT
            id,
            1 - (description_embedding <=> CAST(:vec AS vector)) AS similarity
        FROM complaints
        WHERE description_embedding IS NOT NULL
          AND 1 - (description_embedding <=> CAST(:vec AS vector)) >= :threshold
        ORDER BY similarity DESC
        LIMIT 5
    """)

    async with get_async_session_factory()() as session:
        result = await session.execute(
            query,
            {"vec": vec_literal, "threshold": threshold},
        )
        rows = result.fetchall()

    ids: list[UUID] = []
    scores: list[float] = []
    for row in rows:
        try:
            score = float(row[1])
            # Guard against NaN/Inf which breaks JSON serialization (RFC 7159)
            if not math.isnan(score) and not math.isinf(score):
                ids.append(UUID(str(row[0])))
                scores.append(round(score, 4))
        except (ValueError, TypeError):
            continue

    if ids:
        logger.info(
            "find_duplicates: found %d duplicate(s), top score=%.3f",
            len(ids), scores[0],
        )

    return ids, scores
