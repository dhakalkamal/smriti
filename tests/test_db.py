from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

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
        assert [migration.filename for migration in second_run] == [
            "002_scopes.sql",
            "003_episode_scope_and_retrieval_provenance.sql",
            "004_assistant_retrieval_provenance.sql",
        ]
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
                "message_retrievals",
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

            episode_scope_id_column = await connection.fetchrow(
                """
                SELECT data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'episodes'
                  AND column_name = 'scope_id';
                """
            )
            assert episode_scope_id_column is not None
            assert episode_scope_id_column["data_type"] == "uuid"
            assert episode_scope_id_column["is_nullable"] == "NO"

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


@pytest.mark.asyncio
async def test_migration_003_backfills_existing_episode_scope_id(tmp_path: Path) -> None:
    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"
    stage_two_migrations_dir = tmp_path / "stage_two_migrations"
    stage_two_migrations_dir.mkdir()

    for filename in ["001_init.sql", "002_scopes.sql"]:
        migration = migrations_dir / filename
        (stage_two_migrations_dir / filename).write_text(
            migration.read_text(encoding="utf-8"),
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
        await apply_migrations(settings=settings, migrations_dir=stage_two_migrations_dir)

        connection = await asyncpg.connect(dsn=database_url)
        try:
            user_id = await connection.fetchval("INSERT INTO users DEFAULT VALUES RETURNING id;")
            scope_id = await connection.fetchval(
                """
                INSERT INTO scopes (user_id, name, system_prompt)
                VALUES ($1, $2, $3)
                RETURNING id;
                """,
                user_id,
                "Research Notes",
                "Keep this scoped.",
            )
            conversation_id = await connection.fetchval(
                """
                INSERT INTO conversations (user_id, scope_id, title)
                VALUES ($1, $2, $3)
                RETURNING id;
                """,
                user_id,
                scope_id,
                "Pre-003 conversation",
            )
            message_id = await connection.fetchval(
                """
                INSERT INTO messages (conversation_id, position, role, content, token_count)
                VALUES ($1, 1, 'user', $2, 3)
                RETURNING id;
                """,
                conversation_id,
                "Remember this.",
            )
            episode_id = await connection.fetchval(
                """
                INSERT INTO episodes (conversation_id, kind, message_id, content)
                VALUES ($1, 'message', $2, $3)
                RETURNING id;
                """,
                conversation_id,
                message_id,
                "Remember this.",
            )
        finally:
            await connection.close()

        applied = await apply_migrations(settings=settings, migrations_dir=migrations_dir)

        connection = await asyncpg.connect(dsn=database_url)
        try:
            row = await connection.fetchrow(
                """
                SELECT scope_id
                FROM episodes
                WHERE id = $1;
                """,
                episode_id,
            )
        finally:
            await connection.close()

        assert [migration.filename for migration in applied] == [
            "003_episode_scope_and_retrieval_provenance.sql",
            "004_assistant_retrieval_provenance.sql",
        ]
        assert row is not None
        assert row["scope_id"] == scope_id


@pytest.mark.asyncio
async def test_migration_004_backfills_existing_assistant_message_id(tmp_path: Path) -> None:
    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"
    stage_three_migrations_dir = tmp_path / "stage_three_migrations"
    stage_three_migrations_dir.mkdir()

    for filename in [
        "001_init.sql",
        "002_scopes.sql",
        "003_episode_scope_and_retrieval_provenance.sql",
    ]:
        migration = migrations_dir / filename
        (stage_three_migrations_dir / filename).write_text(
            migration.read_text(encoding="utf-8"),
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
        await apply_migrations(settings=settings, migrations_dir=stage_three_migrations_dir)

        connection = await asyncpg.connect(dsn=database_url)
        try:
            user_id = await connection.fetchval("INSERT INTO users DEFAULT VALUES RETURNING id;")
            scope_id = await connection.fetchval(
                """
                INSERT INTO scopes (user_id, name, system_prompt)
                VALUES ($1, $2, $3)
                RETURNING id;
                """,
                user_id,
                "Stage 7.1 Scope",
                "Keep provenance attached to assistant messages.",
            )
            conversation_id = await connection.fetchval(
                """
                INSERT INTO conversations (user_id, scope_id, title)
                VALUES ($1, $2, $3)
                RETURNING id;
                """,
                user_id,
                scope_id,
                "Pre-004 provenance",
            )
            query_message_id = await connection.fetchval(
                """
                INSERT INTO messages (conversation_id, position, role, content, token_count)
                VALUES ($1, 1, 'user', $2, 3)
                RETURNING id;
                """,
                conversation_id,
                "Use this memory.",
            )
            memory_message_id = await connection.fetchval(
                """
                INSERT INTO messages (conversation_id, position, role, content, token_count)
                VALUES ($1, 2, 'user', $2, 4)
                RETURNING id;
                """,
                conversation_id,
                "Existing scoped memory.",
            )
            assistant_message_id = await connection.fetchval(
                """
                INSERT INTO messages (conversation_id, position, role, content, token_count)
                VALUES ($1, 3, 'assistant', $2, 5)
                RETURNING id;
                """,
                conversation_id,
                "Here is the answer.",
            )
            episode_id = await connection.fetchval(
                """
                INSERT INTO episodes (conversation_id, scope_id, kind, message_id, content)
                VALUES ($1, $2, 'message', $3, $4)
                RETURNING id;
                """,
                conversation_id,
                scope_id,
                memory_message_id,
                "Existing scoped memory.",
            )
            embedding_model_id = await connection.fetchval(
                """
                SELECT id
                FROM embedding_models
                WHERE model_id = 'nomic-embed-text';
                """
            )
            retrieval_id = await connection.fetchval(
                """
                INSERT INTO message_retrievals (
                    query_message_id,
                    query_conversation_id,
                    scope_id,
                    episode_id,
                    embedding_model_id,
                    result_rank,
                    similarity,
                    recency_score,
                    access_score,
                    importance_score,
                    frequency_score,
                    score,
                    scoring_version
                )
                VALUES ($1, $2, $3, $4, $5, 1, 0.8, 0.5, 0.0, 0.0, 0.0, 0.54, $6)
                RETURNING id;
                """,
                query_message_id,
                conversation_id,
                scope_id,
                episode_id,
                embedding_model_id,
                "pre-stage-7.1",
            )
        finally:
            await connection.close()

        applied = await apply_migrations(settings=settings, migrations_dir=migrations_dir)

        connection = await asyncpg.connect(dsn=database_url)
        try:
            row = await connection.fetchrow(
                """
                SELECT assistant_message_id
                FROM message_retrievals
                WHERE id = $1;
                """,
                retrieval_id,
            )
            column = await connection.fetchrow(
                """
                SELECT data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'message_retrievals'
                  AND column_name = 'assistant_message_id';
                """
            )
        finally:
            await connection.close()

        assert [migration.filename for migration in applied] == [
            "004_assistant_retrieval_provenance.sql"
        ]
        assert row is not None
        assert row["assistant_message_id"] == assistant_message_id
        assert column is not None
        assert column["data_type"] == "uuid"
        assert column["is_nullable"] == "NO"


@pytest.mark.asyncio
async def test_message_retrievals_schema_and_basic_fk_behavior() -> None:
    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"

    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        database_url = _to_asyncpg_dsn(postgres.get_connection_url())
        settings = Settings(database_url=database_url)
        await apply_migrations(settings=settings, migrations_dir=migrations_dir)

        connection = await asyncpg.connect(dsn=database_url)
        try:
            user_id = await connection.fetchval("INSERT INTO users DEFAULT VALUES RETURNING id;")
            scope_id = await connection.fetchval(
                """
                INSERT INTO scopes (user_id, name, system_prompt)
                VALUES ($1, $2, $3)
                RETURNING id;
                """,
                user_id,
                "Retrieval Scope",
                "Keep retrieval provenance scoped.",
            )
            conversation_id = await connection.fetchval(
                """
                INSERT INTO conversations (user_id, scope_id, title)
                VALUES ($1, $2, $3)
                RETURNING id;
                """,
                user_id,
                scope_id,
                "Retrieval provenance",
            )
            query_message_id = await connection.fetchval(
                """
                INSERT INTO messages (conversation_id, position, role, content, token_count)
                VALUES ($1, 1, 'user', $2, 4)
                RETURNING id;
                """,
                conversation_id,
                "What did I say?",
            )
            memory_message_id = await connection.fetchval(
                """
                INSERT INTO messages (conversation_id, position, role, content, token_count)
                VALUES ($1, 2, 'user', $2, 5)
                RETURNING id;
                """,
                conversation_id,
                "You mentioned local memory.",
            )
            assistant_message_id = await connection.fetchval(
                """
                INSERT INTO messages (conversation_id, position, role, content, token_count)
                VALUES ($1, 3, 'assistant', $2, 5)
                RETURNING id;
                """,
                conversation_id,
                "Here is the local-memory answer.",
            )
            episode_id = await connection.fetchval(
                """
                INSERT INTO episodes (conversation_id, scope_id, kind, message_id, content)
                VALUES ($1, $2, 'message', $3, $4)
                RETURNING id;
                """,
                conversation_id,
                scope_id,
                memory_message_id,
                "You mentioned local memory.",
            )
            embedding_model_id = await connection.fetchval(
                """
                SELECT id
                FROM embedding_models
                WHERE model_id = 'nomic-embed-text';
                """
            )
            retrieval_id = await connection.fetchval(
                """
                INSERT INTO message_retrievals (
                    query_message_id,
                    assistant_message_id,
                    query_conversation_id,
                    scope_id,
                    episode_id,
                    embedding_model_id,
                    result_rank,
                    similarity,
                    recency_score,
                    access_score,
                    importance_score,
                    frequency_score,
                    score,
                    scoring_version
                )
                VALUES ($1, $2, $3, $4, $5, $6, 1, 0.8, 0.5, 0.0, 0.0, 0.0, 0.54, $7)
                RETURNING id;
                """,
                query_message_id,
                assistant_message_id,
                conversation_id,
                scope_id,
                episode_id,
                embedding_model_id,
                "test-v1",
            )
            assert retrieval_id is not None

            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await connection.execute(
                    """
                    INSERT INTO message_retrievals (
                        query_message_id,
                        assistant_message_id,
                        query_conversation_id,
                        scope_id,
                        episode_id,
                        embedding_model_id,
                        result_rank,
                        similarity,
                        recency_score,
                        access_score,
                        importance_score,
                        frequency_score,
                        score,
                        scoring_version
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, 2, 0.7, 0.5, 0.0, 0.0, 0.0, 0.49, $7);
                    """,
                    query_message_id,
                    assistant_message_id,
                    conversation_id,
                    scope_id,
                    uuid4(),
                    embedding_model_id,
                    "test-v1",
                )

            with pytest.raises(asyncpg.NotNullViolationError):
                await connection.execute(
                    """
                    INSERT INTO message_retrievals (
                        query_message_id,
                        query_conversation_id,
                        scope_id,
                        episode_id,
                        embedding_model_id,
                        result_rank,
                        similarity,
                        recency_score,
                        access_score,
                        importance_score,
                        frequency_score,
                        score,
                        scoring_version
                    )
                    VALUES ($1, $2, $3, $4, $5, 2, 0.7, 0.5, 0.0, 0.0, 0.0, 0.49, $6);
                    """,
                    query_message_id,
                    conversation_id,
                    scope_id,
                    episode_id,
                    embedding_model_id,
                    "test-v1",
                )

            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await connection.execute(
                    """
                    INSERT INTO message_retrievals (
                        query_message_id,
                        assistant_message_id,
                        query_conversation_id,
                        scope_id,
                        episode_id,
                        embedding_model_id,
                        result_rank,
                        similarity,
                        recency_score,
                        access_score,
                        importance_score,
                        frequency_score,
                        score,
                        scoring_version
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, 2, 0.7, 0.5, 0.0, 0.0, 0.0, 0.49, $7);
                    """,
                    query_message_id,
                    uuid4(),
                    conversation_id,
                    scope_id,
                    episode_id,
                    embedding_model_id,
                    "test-v1",
                )

            assistant_index_exists = await connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND tablename = 'message_retrievals'
                      AND indexname = 'idx_message_retrievals_assistant_message_id'
                );
                """
            )
            assert assistant_index_exists is True
        finally:
            await connection.close()
