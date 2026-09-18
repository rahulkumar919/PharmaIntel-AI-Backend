"""
db/models.py — SQLAlchemy ORM model for the complaints table.

Design decisions:
  - One table (complaints) stores the full complaint record as a mix of
    typed columns (for filtering/sorting) and a JSONB column (extraction_data)
    for the semi-structured extraction payload.  This avoids a wide table with
    13+ nullable columns while still allowing SQL queries on common fields.
  - The embedding column uses pgvector's VECTOR type.  We store the embedding
    of detailed_complaint_description here so duplicate_detection can run a
    single cosine-similarity index scan rather than a full table scan.
  - created_at / updated_at are filled by the DB server (server_default), not
    by the application.  This avoids timezone mismatches between the app and DB.

Why JSONB for extraction_data?
  The ComplaintExtraction schema may gain new fields over time.  Storing the
  full extraction as JSONB means we can add fields to the Pydantic model without
  running a schema migration — the existing rows stay valid and the new fields
  simply appear on new rows.  The most important fields (product_name,
  batch_lot_number, severity) are also stored as typed columns for fast queries.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as SAUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.schemas.complaint import Priority, Severity


class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.
    Alembic's env.py imports this to discover table metadata for migrations.
    """
    pass


class ComplaintORM(Base):
    """
    ORM representation of a confirmed, persisted complaint record.

    Column groups:
      Identity     — id (PK), session_id (link back to LangGraph thread)
      Typed fields — key complaint fields stored as proper SQL types so
                     dashboards and filters work without parsing JSONB
      Payload      — extraction_data (full ComplaintExtraction as JSONB)
      AI analysis  — risk_justification, root_cause_hypothesis, capa_text,
                     summary_text, duplicate_ids
      Embedding    — description_embedding (pgvector VECTOR for cosine search)
      Timestamps   — created_at, updated_at (server-side)
    """
    __tablename__ = "complaints"

    # ── Identity ──────────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        SAUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Application-level UUID — also used as the pgvector row key.",
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        SAUUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="LangGraph thread_id — links this record back to the graph checkpoint.",
    )

    # ── Frequently-queried typed columns ──────────────────────────────────────
    # These mirror the most important ComplaintExtraction fields.
    # Storing them here (in addition to extraction_data JSONB) avoids parsing
    # JSON for every dashboard query or filter.
    product_name: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    batch_lot_number: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    customer_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    complaint_date: Mapped[str | None] = mapped_column(String(20), nullable=True)  # ISO date string

    initial_severity: Mapped[str | None] = mapped_column(
        SAEnum(Severity, name="severity_enum", create_type=False),
        nullable=True,
        index=True,
    )
    priority: Mapped[str | None] = mapped_column(
        SAEnum(Priority, name="priority_enum", create_type=False),
        nullable=True,
        index=True,
    )

    # ── Full extraction payload ───────────────────────────────────────────────
    # Stores the entire ComplaintExtraction as JSONB.
    # Accessed as: complaint_orm.extraction_data["batch_lot_number"]
    extraction_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ── AI analysis fields ────────────────────────────────────────────────────
    risk_justification: Mapped[str | None] = mapped_column(Text, nullable=True)
    root_cause_hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    capa_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Stored as a JSON string list — no need for a separate join table for now
    duplicate_ids: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Duplicate-detection embedding ─────────────────────────────────────────
    # Vector dimension must match the output of the sentence-transformer model.
    # all-MiniLM-L6-v2 → 384 dimensions.
    # If you swap models, update this number AND rebuild the index.
    # ASSUMPTION: 384 dimensions hardcoded here; config.py could expose this
    #             as EMBEDDING_DIM if you need to swap models without code changes.
    description_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384),
        nullable=True,
        comment="Embedding of detailed_complaint_description for cosine similarity search.",
    )

    # ── Audit ─────────────────────────────────────────────────────────────────
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_confirmed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<ComplaintORM id={self.id} product={self.product_name!r} "
            f"batch={self.batch_lot_number!r} severity={self.initial_severity}>"
        )
