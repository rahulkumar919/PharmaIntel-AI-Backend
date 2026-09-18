"""Initial complaints table with pgvector embedding column.

Revision ID: 0001
Revises    : (none — this is the first migration)
Create Date: 2026-09-18

What this migration does:
  1. Enables the pgvector extension in Postgres (idempotent).
  2. Creates the ENUM types for severity and priority.
  3. Creates the complaints table with all columns including the
     VECTOR(384) embedding column for cosine similarity search.
  4. Creates an ivfflat index on description_embedding for fast ANN search.
     ivfflat is pgvector's approximate nearest-neighbour index — much faster
     than a brute-force scan for tables with >10k rows.

IMPORTANT: Before running this migration, your Postgres instance must have
the pgvector extension available.  Install it with:
  -- In psql as superuser:
  CREATE EXTENSION IF NOT EXISTS vector;
OR via the migration itself (op.execute below handles this).
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

# Alembic revision identifiers
revision: str = "0001"
down_revision = None          # first migration — no parent
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. Enable pgvector extension ─────────────────────────────────────────
    # IF NOT EXISTS makes this idempotent — safe to run multiple times.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── 2. Create ENUM types ──────────────────────────────────────────────────
    # SQLAlchemy's Enum creates a Postgres native enum type.
    # create_type=True here (migration context) — the ORM model uses
    # create_type=False to avoid trying to re-create it at runtime.
    severity_enum = sa.Enum(
        "Critical", "Major", "Minor",
        name="severity_enum",
    )
    priority_enum = sa.Enum(
        "High", "Medium", "Low",
        name="priority_enum",
    )
    severity_enum.create(op.get_bind(), checkfirst=True)
    priority_enum.create(op.get_bind(), checkfirst=True)

    # ── 3. Create complaints table ────────────────────────────────────────────
    op.create_table(
        "complaints",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", UUID(as_uuid=True), nullable=True, index=True),

        # Frequently-queried typed columns
        sa.Column("product_name", sa.String(256), nullable=True),
        sa.Column("batch_lot_number", sa.String(128), nullable=True),
        sa.Column("customer_name", sa.String(256), nullable=True),
        sa.Column("complaint_date", sa.String(20), nullable=True),
        sa.Column("initial_severity",
                  sa.Enum("Critical", "Major", "Minor",
                          name="severity_enum", create_type=False),
                  nullable=True),
        sa.Column("priority",
                  sa.Enum("High", "Medium", "Low",
                          name="priority_enum", create_type=False),
                  nullable=True),

        # Full extraction payload as JSONB
        sa.Column("extraction_data", JSONB, nullable=True),

        # AI analysis fields
        sa.Column("risk_justification", sa.Text, nullable=True),
        sa.Column("root_cause_hypothesis", sa.Text, nullable=True),
        sa.Column("capa_text", sa.Text, nullable=True),
        sa.Column("summary_text", sa.Text, nullable=True),
        sa.Column("duplicate_ids", sa.Text, nullable=True),

        # pgvector embedding — 384 dims for all-MiniLM-L6-v2
        sa.Column("description_embedding", Vector(384), nullable=True),

        # Audit
        sa.Column("raw_text", sa.Text, nullable=True),
        sa.Column("is_confirmed", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
    )

    # ── 4. Indexes ────────────────────────────────────────────────────────────
    op.create_index("ix_complaints_product_name",   "complaints", ["product_name"])
    op.create_index("ix_complaints_batch_lot",      "complaints", ["batch_lot_number"])
    op.create_index("ix_complaints_severity",       "complaints", ["initial_severity"])
    op.create_index("ix_complaints_priority",       "complaints", ["priority"])

    # ivfflat approximate nearest-neighbour index for cosine similarity search.
    # lists=100 is a safe starting value for tables up to ~1M rows.
    # Rebuild the index (REINDEX) after bulk inserts to refresh the centroids.
    # operator class vector_cosine_ops tells Postgres to use cosine distance.
    op.execute("""
        CREATE INDEX ix_complaints_embedding_cosine
        ON complaints
        USING ivfflat (description_embedding vector_cosine_ops)
        WITH (lists = 100)
    """)


def downgrade() -> None:
    op.drop_index("ix_complaints_embedding_cosine", table_name="complaints")
    op.drop_index("ix_complaints_priority",         table_name="complaints")
    op.drop_index("ix_complaints_severity",         table_name="complaints")
    op.drop_index("ix_complaints_batch_lot",        table_name="complaints")
    op.drop_index("ix_complaints_product_name",     table_name="complaints")
    op.drop_table("complaints")
    op.execute("DROP TYPE IF EXISTS severity_enum")
    op.execute("DROP TYPE IF EXISTS priority_enum")
