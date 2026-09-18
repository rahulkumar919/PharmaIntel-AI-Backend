"""
alembic/env.py — Alembic environment configuration.

Why async run_migrations_online?
  Our engine is async (asyncpg driver).  Alembic's default env.py uses a sync
  connection.  We use run_sync() to run the synchronous migration logic inside
  an async context — this is the pattern recommended in the Alembic docs for
  async engines.

Why import Base.metadata here?
  Alembic uses the metadata to detect schema differences between the ORM models
  and the actual database when generating autogenerate migrations.  Without this
  import, `alembic revision --autogenerate` would always produce empty migrations.
"""

import asyncio
import sys
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── Make app importable when running alembic from backend/ ────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load .env so Settings can resolve DATABASE_URL
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from app.config import get_settings
from app.db.models import Base          # imports all ORM models via metadata

# Alembic Config object provides access to alembic.ini values
config = context.config

# Inject the real DB URL from our Settings (overrides the empty placeholder in alembic.ini)
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

# Python logging setup from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is the metadata autogenerate uses to compare against the live DB schema
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode — generates SQL without a live DB connection.
    Useful for generating migration scripts to review before applying.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations online using the async engine.
    run_sync() bridges the async engine to Alembic's synchronous migration API.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,    # NullPool: no connection pooling during migrations
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
