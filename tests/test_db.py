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
async def test_migrations_create_schema_and_seed_embedding_model(tmp_path: Path) -> None:
    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"
    stage_one_migrations_dir = tmp_path / "stage_one_migrations"
    stage_one_migrations_dir.mkdir()
    stage_one_migration = migrations_dir / "001_init.sql"
    (stage_one_migrations_dir / stage_one_migration.name).write_text(
        stage_one_migration.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        database_url = _to_asyncpg_dsn(postgres.get_connection_url())
        settings = Settings(database_url=database_url)

        first_run = await apply_migrations(
            settings=settings,
            migrations_dir=stage_one_migrations_dir,
        )

        connection = await asyncpg.connect(dsn=database_url)
        try:
            existing_user_id = await connection.fetchval(
                "INSERT INTO users DEFAULT VALUES RETURNING id;"
            )
            existing_conversation_id = await connection.fetchval(
                """
                INSERT INTO conversations (user_id, title)
                VALUES ($1, $2)
                RETURNING id;
                """,
                existing_user_id,
                "Pre-scope conversation",
            )
        finally:
            await connection.close()

        second_run = await apply_migrations(settings=settings, migrations_dir=migrations_dir)
        third_run = await apply_migrations(settings=settings, migrations_dir=migrations_dir)

        assert [migration.filename for migration in first_run] == ["001_init.sql"]
        assert [migration.filename for migration in second_run] == ["002_scopes.sql"]
        assert third_run == []

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

            for table_name in ["users", "scopes", "conversations", "messages", "episodes"]:
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

            scope_id_column = await connection.fetchrow(
                """
                SELECT data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'conversations'
                  AND column_name = 'scope_id';
                """
            )
            assert scope_id_column is not None
            assert scope_id_column["data_type"] == "uuid"
            assert scope_id_column["is_nullable"] == "NO"

            default_scope = await connection.fetchrow(
                """
                SELECT scopes.id, scopes.user_id, scopes.name, scopes.system_prompt
                FROM conversations
                JOIN scopes ON scopes.id = conversations.scope_id
                WHERE conversations.id = $1;
                """,
                existing_conversation_id,
            )
            assert default_scope is not None
            assert default_scope["user_id"] == existing_user_id
            assert default_scope["name"] == "Default"
            assert default_scope["system_prompt"] == ""

            same_user_constraint_exists = await connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'conversations_scope_user_fkey'
                      AND conrelid = 'conversations'::regclass
                );
                """
            )
            assert same_user_constraint_exists is True

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

            with pytest.raises(asyncpg.UniqueViolationError):
                await connection.execute(
                    """
                    INSERT INTO scopes (user_id, name, system_prompt)
                    VALUES ($1, $2, $3);
                    """,
                    user_id,
                    "Test Scope",
                    "Duplicate names in one user must be rejected.",
                )

            other_user_id = await connection.fetchval(
                "INSERT INTO users DEFAULT VALUES RETURNING id;"
            )
            same_name_other_user_scope_id = await connection.fetchval(
                """
                INSERT INTO scopes (user_id, name, system_prompt)
                VALUES ($1, $2, $3)
                RETURNING id;
                """,
                other_user_id,
                "Test Scope",
                "The same name is allowed for another user.",
            )
            assert same_name_other_user_scope_id is not None

            with pytest.raises(asyncpg.NotNullViolationError):
                await connection.execute(
                    """
                    INSERT INTO scopes (user_id, name, system_prompt)
                    VALUES (NULL, $1, $2);
                    """,
                    "Missing User Scope",
                    "Scopes must belong to a user.",
                )

            with pytest.raises(asyncpg.NotNullViolationError):
                await connection.execute(
                    """
                    INSERT INTO scopes (user_id, name, system_prompt)
                    VALUES ($1, NULL, $2);
                    """,
                    user_id,
                    "Scopes must have a user-facing name.",
                )

            with pytest.raises(asyncpg.NotNullViolationError):
                await connection.execute(
                    """
                    INSERT INTO conversations (user_id, title)
                    VALUES ($1, $2);
                    """,
                    user_id,
                    "Missing scope conversation",
                )

            other_scope_id = await connection.fetchval(
                """
                INSERT INTO scopes (user_id, name, system_prompt)
                VALUES ($1, $2, $3)
                RETURNING id;
                """,
                other_user_id,
                "Other User Scope",
                "Keep this separate.",
            )

            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await connection.execute(
                    """
                    INSERT INTO conversations (user_id, scope_id, title)
                    VALUES ($1, $2, $3);
                    """,
                    user_id,
                    other_scope_id,
                    "Cross-user scope conversation",
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
