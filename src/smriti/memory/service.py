from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import cast
from uuid import UUID

import asyncpg

from smriti.embeddings import Embedder, EmbeddingVector
from smriti.memory.errors import (
    ConversationNotFoundError,
    EmbeddingModelNotFoundError,
    ScopeNotFoundError,
    VectorDimensionError,
)
from smriti.memory.models import (
    AppendMessageRequest,
    ConversationRecord,
    CreateConversationRequest,
    CreateMessageEpisodeRequest,
    CreateScopeRequest,
    EpisodeRecord,
    ListScopesRequest,
    MessageRecord,
    MessageRole,
    ScopeRecord,
)

EMBEDDINGS_768_DIMENSIONS = 768


@dataclass(frozen=True)
class MemoryService:
    """Core memory operations shared by future HTTP and MCP layers."""

    pool: asyncpg.Pool
    embedder: Embedder
    embedding_model_id: str = "nomic-embed-text"

    async def create_scope(self, request: CreateScopeRequest) -> ScopeRecord:
        """Create a user-owned memory scope."""

        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO scopes (user_id, name, system_prompt)
                VALUES ($1, $2, $3)
                RETURNING id, user_id, name, system_prompt, created_at, updated_at;
                """,
                request.user_id,
                request.name,
                request.system_prompt,
            )

        if row is None:
            raise ScopeNotFoundError("Scope was not created")
        return _scope_from_row(row)

    async def list_scopes(self, request: ListScopesRequest) -> list[ScopeRecord]:
        """List scopes owned by one user."""

        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT id, user_id, name, system_prompt, created_at, updated_at
                FROM scopes
                WHERE user_id = $1
                ORDER BY created_at ASC, id ASC;
                """,
                request.user_id,
            )

        return [_scope_from_row(row) for row in rows]

    async def create_conversation(
        self,
        request: CreateConversationRequest,
    ) -> ConversationRecord:
        """Create a conversation inside a scope owned by the same user."""

        async with self.pool.acquire() as connection, connection.transaction():
            scope_exists = await connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM scopes
                    WHERE id = $1
                      AND user_id = $2
                );
                """,
                request.scope_id,
                request.user_id,
            )
            if scope_exists is not True:
                raise ScopeNotFoundError("Scope does not belong to the expected user")

            row = await connection.fetchrow(
                """
                INSERT INTO conversations (user_id, scope_id, title)
                VALUES ($1, $2, $3)
                RETURNING id, user_id, scope_id, title, created_at, updated_at;
                """,
                request.user_id,
                request.scope_id,
                request.title,
            )

        if row is None:
            raise ConversationNotFoundError("Conversation was not created")
        return _conversation_from_row(row)

    async def append_message(self, request: AppendMessageRequest) -> MessageRecord:
        """Append an immutable message to a conversation in the expected scope."""

        if request.token_count < 0:
            raise ValueError("token_count must be non-negative")

        async with self.pool.acquire() as connection, connection.transaction():
            await self._lock_conversation(
                connection=connection,
                user_id=request.user_id,
                scope_id=request.scope_id,
                conversation_id=request.conversation_id,
            )
            position = await self._next_message_position(
                connection=connection,
                conversation_id=request.conversation_id,
            )
            row = await connection.fetchrow(
                """
                INSERT INTO messages (conversation_id, position, role, content, token_count)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, conversation_id, position, role, content, token_count, created_at;
                """,
                request.conversation_id,
                position,
                request.role,
                request.content,
                request.token_count,
            )
            await connection.execute(
                """
                UPDATE conversations
                SET updated_at = NOW()
                WHERE id = $1;
                """,
                request.conversation_id,
            )

        if row is None:
            raise ConversationNotFoundError("Message was not appended")
        return _message_from_row(row)

    async def create_message_episode(
        self,
        request: CreateMessageEpisodeRequest,
    ) -> EpisodeRecord:
        """Create and embed the retrieval episode for one stored message."""

        message = await self._get_message_for_scope(request)
        vector = await self.embedder.embed_text(message.content)
        self._validate_embedding_vector(vector)

        async with self.pool.acquire() as connection, connection.transaction():
            await self._lock_conversation(
                connection=connection,
                user_id=request.user_id,
                scope_id=request.scope_id,
                conversation_id=request.conversation_id,
            )
            embedding_model_pk = await self._embedding_model_pk(connection)
            row = await connection.fetchrow(
                """
                INSERT INTO episodes (conversation_id, scope_id, kind, message_id, content)
                VALUES ($1, $2, 'message', $3, $4)
                RETURNING id, conversation_id, scope_id, message_id, content, created_at;
                """,
                request.conversation_id,
                request.scope_id,
                request.message_id,
                message.content,
            )
            if row is None:
                raise ConversationNotFoundError("Episode was not created")

            episode_id = cast(UUID, row["id"])
            await connection.execute(
                """
                INSERT INTO embeddings_768 (episode_id, model_id, embedding)
                VALUES ($1, $2, $3);
                """,
                episode_id,
                embedding_model_pk,
                list(vector),
            )

        return _episode_from_row(row, embedding_model_pk)

    async def _get_message_for_scope(
        self,
        request: CreateMessageEpisodeRequest,
    ) -> MessageRecord:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT
                    messages.id,
                    messages.conversation_id,
                    messages.position,
                    messages.role,
                    messages.content,
                    messages.token_count,
                    messages.created_at
                FROM messages
                JOIN conversations ON conversations.id = messages.conversation_id
                WHERE messages.id = $1
                  AND messages.conversation_id = $2
                  AND conversations.user_id = $3
                  AND conversations.scope_id = $4;
                """,
                request.message_id,
                request.conversation_id,
                request.user_id,
                request.scope_id,
            )

        if row is None:
            raise ConversationNotFoundError("Message is not in the expected conversation scope")
        return _message_from_row(row)

    async def _lock_conversation(
        self,
        connection: asyncpg.Connection,
        user_id: UUID,
        scope_id: UUID,
        conversation_id: UUID,
    ) -> None:
        row = await connection.fetchrow(
            """
            SELECT id
            FROM conversations
            WHERE id = $1
              AND user_id = $2
              AND scope_id = $3
            FOR UPDATE;
            """,
            conversation_id,
            user_id,
            scope_id,
        )
        if row is None:
            raise ConversationNotFoundError("Conversation is not in the expected scope")

    async def _next_message_position(
        self,
        connection: asyncpg.Connection,
        conversation_id: UUID,
    ) -> int:
        value = await connection.fetchval(
            """
            SELECT COALESCE(MAX(position), 0) + 1
            FROM messages
            WHERE conversation_id = $1;
            """,
            conversation_id,
        )
        return cast(int, value)

    async def _embedding_model_pk(self, connection: asyncpg.Connection) -> int:
        value = await connection.fetchval(
            """
            SELECT id
            FROM embedding_models
            WHERE model_id = $1
              AND dimensions = $2
              AND is_active = TRUE;
            """,
            self.embedding_model_id,
            EMBEDDINGS_768_DIMENSIONS,
        )
        if value is None:
            raise EmbeddingModelNotFoundError("Embedding model is not registered for 768d storage")
        return cast(int, value)

    def _validate_embedding_vector(self, vector: EmbeddingVector) -> None:
        if len(vector) != EMBEDDINGS_768_DIMENSIONS:
            raise VectorDimensionError("Embedding vector must have 768 dimensions")


def _scope_from_row(row: asyncpg.Record) -> ScopeRecord:
    return ScopeRecord(
        id=cast(UUID, row["id"]),
        user_id=cast(UUID, row["user_id"]),
        name=cast(str, row["name"]),
        system_prompt=cast(str, row["system_prompt"]),
        created_at=cast(datetime, row["created_at"]),
        updated_at=cast(datetime, row["updated_at"]),
    )


def _conversation_from_row(row: asyncpg.Record) -> ConversationRecord:
    return ConversationRecord(
        id=cast(UUID, row["id"]),
        user_id=cast(UUID, row["user_id"]),
        scope_id=cast(UUID, row["scope_id"]),
        title=cast(str | None, row["title"]),
        created_at=cast(datetime, row["created_at"]),
        updated_at=cast(datetime, row["updated_at"]),
    )


def _message_from_row(row: asyncpg.Record) -> MessageRecord:
    return MessageRecord(
        id=cast(UUID, row["id"]),
        conversation_id=cast(UUID, row["conversation_id"]),
        position=cast(int, row["position"]),
        role=cast(MessageRole, row["role"]),
        content=cast(str, row["content"]),
        token_count=cast(int, row["token_count"]),
        created_at=cast(datetime, row["created_at"]),
    )


def _episode_from_row(row: asyncpg.Record, embedding_model_id: int) -> EpisodeRecord:
    return EpisodeRecord(
        id=cast(UUID, row["id"]),
        conversation_id=cast(UUID, row["conversation_id"]),
        scope_id=cast(UUID, row["scope_id"]),
        message_id=cast(UUID, row["message_id"]),
        content=cast(str, row["content"]),
        created_at=cast(datetime, row["created_at"]),
        embedding_model_id=embedding_model_id,
    )
