from __future__ import annotations

import re
from pathlib import Path

import asyncpg
import pytest
from testcontainers.postgres import PostgresContainer

from smriti.config import Settings
from smriti.db.migrate import apply_migrations


def _to_asyncpg_dsn(container_url: str) -> str:
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", container_url)


@pytest.mark.asyncio
async def test_migrations_create_schema_and_seed_embedding_model() -> None:
    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"

    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        database_url = _to_asyncpg_dsn(postgres.get_connection_url())
        settings = Settings(database_url=database_url)

        first_run = await apply_migrations(settings=settings, migrations_dir=migrations_dir)
        second_run = await apply_migrations(settings=settings, migrations_dir=migrations_dir)

        assert [migration.filename for migration in first_run] == ["001_init.sql", "002_scopes.sql"]
        assert second_run == []

        connection = await asyncpg.connect(dsn=database_url)
        try:
            expected_tables = [
                "users",
                "scopes",
                "conversations",
                "messages",
                "episodes",
                "embedding_models",
                "embeddings_768",
                "eval_scenarios",
                "eval_runs",
                "eval_run_results",
                "schema_migrations",
            ]

            for table_name in expected_tables:
                result = await connection.fetchval(
                    "SELECT to_regclass($1);", f"public.{table_name}"
                )
                assert result == table_name

            vector_loaded = await connection.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');"
            )
            assert vector_loaded is True

            default_model = await connection.fetchrow(
                """
                SELECT model_id, provider, dimensions
                FROM embedding_models
                WHERE model_id = 'nomic-embed-text';
                """
            )
            assert default_model is not None
            assert default_model["provider"] == "ollama"
            assert default_model["dimensions"] == 768

            for table_name in ["users", "conversations", "messages", "episodes"]:
                data_type = await connection.fetchval(
                    """
                    SELECT data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = $1
                      AND column_name = 'id';
                    """,
                    table_name,
                )
                assert data_type == "uuid"

            scope_id_type = await connection.fetchval(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'conversations'
                  AND column_name = 'scope_id';
                """
            )
            assert scope_id_type == "uuid"

            user_id = await connection.fetchval("INSERT INTO users DEFAULT VALUES RETURNING id;")
            scope_id = await connection.fetchval(
                """
                INSERT INTO scopes (user_id, name, system_prompt)
                VALUES ($1, $2, $3)
                RETURNING id;
                """,
                user_id,
                "Test Scope",
                "You are a scoped assistant.",
            )
            conversation_id = await connection.fetchval(
                """
                INSERT INTO conversations (user_id, scope_id, title)
                VALUES ($1, $2, $3)
                RETURNING id;
                """,
                user_id,
                scope_id,
                "Cascade test conversation",
            )

            before_delete = await connection.fetchval(
                "SELECT COUNT(*) FROM conversations WHERE id = $1;",
                conversation_id,
            )
            assert before_delete == 1

            await connection.execute("DELETE FROM scopes WHERE id = $1;", scope_id)

            after_delete = await connection.fetchval(
                "SELECT COUNT(*) FROM conversations WHERE id = $1;",
                conversation_id,
            )
            assert after_delete == 0
        finally:
            await connection.close()
