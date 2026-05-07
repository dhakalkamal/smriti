from __future__ import annotations

import asyncio

import asyncpg
from pgvector.asyncpg import register_vector

from smriti.config import Settings, get_settings

_pool: asyncpg.Pool | None = None
_pool_dsn: str | None = None
_pool_lock = asyncio.Lock()


async def _init_connection(connection: asyncpg.Connection) -> None:
    """Initialize each pooled connection with pgvector codecs."""

    await register_vector(connection)


async def get_pool(settings: Settings | None = None) -> asyncpg.Pool:
    """Return a shared asyncpg pool configured for Smriti.

    The pool is process-global so the app and CLI can reuse connections while
    still allowing test overrides via an explicit settings object.
    """

    global _pool, _pool_dsn

    resolved_settings = settings or get_settings()
    dsn = resolved_settings.database_url

    if _pool is not None and _pool_dsn == dsn:
        return _pool

    async with _pool_lock:
        if _pool is not None and _pool_dsn != dsn:
            await _pool.close()
            _pool = None
            _pool_dsn = None

        if _pool is None:
            _pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=resolved_settings.database_min_pool_size,
                max_size=resolved_settings.database_max_pool_size,
                timeout=resolved_settings.database_connect_timeout,
                command_timeout=resolved_settings.database_command_timeout,
                init=_init_connection,
            )
            _pool_dsn = dsn

    if _pool is None:
        raise RuntimeError("Failed to initialize database connection pool")

    return _pool


async def close_pool() -> None:
    """Close and clear the shared asyncpg pool."""

    global _pool, _pool_dsn

    if _pool is None:
        return

    async with _pool_lock:
        if _pool is not None:
            await _pool.close()
            _pool = None
            _pool_dsn = None
