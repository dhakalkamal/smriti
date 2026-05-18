from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast
from uuid import UUID

import asyncpg

from smriti.embeddings import Embedder, EmbeddingVector
from smriti.memory.errors import (
    ConversationAccessDeniedError,
    ConversationNotFoundError,
    EmbeddingModelNotFoundError,
    InvalidMemoryRequestError,
    InvalidProvenanceTargetError,
    InvalidRetrievalRequestError,
    ScopeAccessDeniedError,
    ScopeNotFoundError,
    VectorDimensionError,
)
from smriti.memory.models import (
    AppendAssistantResponseWithProvenanceRequest,
    AppendMessageRequest,
    AppendMessageWithEpisodeRequest,
    AssistantGenerationContextRecord,
    AssistantResponseRecord,
    ConversationRecord,
    CreateConversationRequest,
    CreateMessageEpisodeRequest,
    CreateScopeRequest,
    DeleteConversationRequest,
    EpisodeKind,
    EpisodeRecord,
    ListConversationsRequest,
    ListMessagesRequest,
    ListScopesRequest,
    LoadAssistantGenerationContextRequest,
    MessageEpisodeRecord,
    MessageRecord,
    MessageRole,
    ScopeRecord,
    ScoredEpisode,
)

EMBEDDINGS_768_DIMENSIONS = 768
RETRIEVAL_CANDIDATE_MULTIPLIER = 5
MIN_RETRIEVAL_CANDIDATES = 25
SECONDS_PER_DAY = 24 * 60 * 60
RECENCY_HALF_LIFE_SECONDS = 30 * SECONDS_PER_DAY
ACCESS_HALF_LIFE_SECONDS = 7 * SECONDS_PER_DAY
SIMILARITY_WEIGHT = 0.55
RECENCY_WEIGHT = 0.20
ACCESS_WEIGHT = 0.10
IMPORTANCE_WEIGHT = 0.10
FREQUENCY_WEIGHT = 0.05
FREQUENCY_NORMALIZATION_COUNT = 10.0
SCORING_VERSION = "stage-5.2-weighted-v1"


@dataclass(frozen=True)
class _ScoredEpisodeCandidate:
    id: UUID
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID
    kind: EpisodeKind
    message_id: UUID | None
    message_position: int | None
    range_start: int | None
    range_end: int | None
    content: str
    created_at: datetime
    importance: float
    access_count: int
    last_accessed_at: datetime | None
    embedding_model_id: int
    similarity: float
    recency_score: float
    access_score: float
    importance_score: float
    frequency_score: float
    score: float


@dataclass(frozen=True)
class _PreparedUsedMemories:
    used_episodes: list[ScoredEpisode]
    episode_ids: list[UUID]
    scoring_version: str
    retrieved_at: datetime


@dataclass(frozen=True)
class _ProvenanceQueryMessageContext:
    conversation_id: UUID
    position: int


@dataclass(frozen=True)
class MemoryService:
    """Core memory operations shared by future HTTP and MCP layers."""

    pool: asyncpg.Pool
    embedder: Embedder
    embedding_model_id: str = "nomic-embed-text"

    async def create_scope(self, request: CreateScopeRequest) -> ScopeRecord:
        """Create a user-owned memory scope."""

        try:
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
        except asyncpg.UniqueViolationError as exc:
            raise InvalidMemoryRequestError("Scope name already exists for this user") from exc

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

    async def list_conversations(
        self,
        request: ListConversationsRequest,
    ) -> list[ConversationRecord]:
        """List conversations owned by one user across all scopes."""

        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT id, user_id, scope_id, title, created_at, updated_at
                FROM conversations
                WHERE user_id = $1
                ORDER BY updated_at DESC, created_at DESC, id ASC;
                """,
                request.user_id,
            )

        return [_conversation_from_row(row) for row in rows]

    async def list_messages(self, request: ListMessagesRequest) -> list[MessageRecord]:
        """List recent messages in a conversation owned by one user."""

        if request.limit <= 0:
            raise InvalidMemoryRequestError("message list limit must be greater than zero")

        async with self.pool.acquire() as connection:
            await self._conversation_scope_for_user(
                connection=connection,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
            )
            rows = await connection.fetch(
                """
                SELECT id, conversation_id, position, role, content, token_count, created_at
                FROM messages
                WHERE conversation_id = $1
                ORDER BY position ASC, id ASC
                LIMIT $2;
                """,
                request.conversation_id,
                request.limit,
            )

        return [_message_from_row(row) for row in rows]

    async def load_assistant_generation_context(
        self,
        request: LoadAssistantGenerationContextRequest,
    ) -> AssistantGenerationContextRecord:
        """Load scope prompt and recent conversation messages for assistant generation."""

        if request.recent_message_limit < 1:
            raise InvalidMemoryRequestError("recent_message_limit must be at least one")

        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT
                    scopes.id AS scope_id,
                    scopes.user_id AS scope_user_id,
                    scopes.name AS scope_name,
                    scopes.system_prompt AS scope_system_prompt,
                    scopes.created_at AS scope_created_at,
                    scopes.updated_at AS scope_updated_at,
                    conversations.id AS conversation_id,
                    conversations.user_id AS conversation_user_id,
                    conversations.scope_id AS conversation_scope_id,
                    conversations.title AS conversation_title,
                    conversations.created_at AS conversation_created_at,
                    conversations.updated_at AS conversation_updated_at,
                    query_messages.id AS query_message_id,
                    query_messages.conversation_id AS query_conversation_id,
                    query_messages.position AS query_position,
                    query_messages.role AS query_role,
                    query_messages.content AS query_content,
                    query_messages.token_count AS query_token_count,
                    query_messages.created_at AS query_created_at
                FROM conversations
                INNER JOIN scopes
                    ON scopes.id = conversations.scope_id
                LEFT JOIN messages AS query_messages
                    ON query_messages.conversation_id = conversations.id
                   AND query_messages.id = $2
                WHERE conversations.id = $1;
                """,
                request.conversation_id,
                request.query_message_id,
            )
            if row is None:
                raise ConversationNotFoundError("Conversation does not exist")
            if cast(UUID, row["conversation_user_id"]) != request.user_id:
                raise ConversationAccessDeniedError("Conversation belongs to a different user")
            if cast(UUID, row["conversation_scope_id"]) != request.scope_id:
                raise ConversationNotFoundError("Conversation is not in the expected scope")
            if cast(UUID, row["scope_user_id"]) != request.user_id:
                raise ScopeAccessDeniedError("Scope belongs to a different user")
            if row["query_message_id"] is None:
                raise ConversationNotFoundError("Query message is not in the expected conversation")
            if cast(str, row["query_role"]) != "user":
                raise InvalidProvenanceTargetError(
                    "Assistant generation requires a user query message"
                )

            recent_rows = await connection.fetch(
                """
                SELECT id, conversation_id, position, role, content, token_count, created_at
                FROM (
                    SELECT id, conversation_id, position, role, content, token_count, created_at
                    FROM messages
                    WHERE conversation_id = $1
                      AND position <= $2
                    ORDER BY position DESC, id DESC
                    LIMIT $3
                ) AS recent_messages
                ORDER BY position ASC, id ASC;
                """,
                request.conversation_id,
                cast(int, row["query_position"]),
                request.recent_message_limit,
            )

        query_message = MessageRecord(
            id=cast(UUID, row["query_message_id"]),
            conversation_id=cast(UUID, row["query_conversation_id"]),
            position=cast(int, row["query_position"]),
            role=cast(MessageRole, row["query_role"]),
            content=cast(str, row["query_content"]),
            token_count=cast(int, row["query_token_count"]),
            created_at=cast(datetime, row["query_created_at"]),
        )

        return AssistantGenerationContextRecord(
            scope=ScopeRecord(
                id=cast(UUID, row["scope_id"]),
                user_id=cast(UUID, row["scope_user_id"]),
                name=cast(str, row["scope_name"]),
                system_prompt=cast(str, row["scope_system_prompt"]),
                created_at=cast(datetime, row["scope_created_at"]),
                updated_at=cast(datetime, row["scope_updated_at"]),
            ),
            conversation=ConversationRecord(
                id=cast(UUID, row["conversation_id"]),
                user_id=cast(UUID, row["conversation_user_id"]),
                scope_id=cast(UUID, row["conversation_scope_id"]),
                title=cast(str | None, row["conversation_title"]),
                created_at=cast(datetime, row["conversation_created_at"]),
                updated_at=cast(datetime, row["conversation_updated_at"]),
            ),
            query_message=query_message,
            recent_messages=tuple(_message_from_row(recent_row) for recent_row in recent_rows),
        )

    async def create_conversation(
        self,
        request: CreateConversationRequest,
    ) -> ConversationRecord:
        """Create a conversation inside a scope owned by the same user."""

        async with self.pool.acquire() as connection, connection.transaction():
            scope_owner_id = cast(
                UUID | None,
                await connection.fetchval(
                    """
                    SELECT user_id
                    FROM scopes
                    WHERE id = $1;
                    """,
                    request.scope_id,
                ),
            )
            if scope_owner_id is None:
                raise ScopeNotFoundError("Scope does not exist")
            if scope_owner_id != request.user_id:
                raise ScopeAccessDeniedError("Scope belongs to a different user")

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

    async def delete_conversation(
        self,
        request: DeleteConversationRequest,
    ) -> None:
        """Hard-delete a user-owned conversation and rely on schema cascades."""

        async with self.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """
                DELETE FROM conversations
                WHERE id = $1
                  AND user_id = $2
                RETURNING id;
                """,
                request.conversation_id,
                request.user_id,
            )

        if row is None:
            raise ConversationNotFoundError("Conversation does not exist")

    async def append_message(self, request: AppendMessageRequest) -> MessageRecord:
        """Append an immutable message to a conversation in the expected scope."""

        if request.token_count < 0:
            raise InvalidMemoryRequestError("token_count must be non-negative")

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

    async def append_message_with_episode(
        self,
        request: AppendMessageWithEpisodeRequest,
    ) -> MessageEpisodeRecord:
        """Append a message and its retrieval episode in one DB transaction."""

        if request.token_count < 0:
            raise InvalidMemoryRequestError("token_count must be non-negative")

        vector = await self.embedder.embed_text(request.content)
        self._validate_embedding_vector(vector)

        async with self.pool.acquire() as connection, connection.transaction():
            scope_id = await self._lock_conversation_for_user(
                connection=connection,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
            )
            embedding_model_pk = await self._embedding_model_pk(connection)
            position = await self._next_message_position(
                connection=connection,
                conversation_id=request.conversation_id,
            )
            message_row = await connection.fetchrow(
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
            if message_row is None:
                raise ConversationNotFoundError("Message was not appended")

            await connection.execute(
                """
                UPDATE conversations
                SET updated_at = NOW()
                WHERE id = $1;
                """,
                request.conversation_id,
            )

            message_id = cast(UUID, message_row["id"])
            episode_row = await connection.fetchrow(
                """
                INSERT INTO episodes (conversation_id, scope_id, kind, message_id, content)
                VALUES ($1, $2, 'message', $3, $4)
                RETURNING id, conversation_id, scope_id, message_id, content, created_at;
                """,
                request.conversation_id,
                scope_id,
                message_id,
                request.content,
            )
            if episode_row is None:
                raise ConversationNotFoundError("Episode was not created")

            await connection.execute(
                """
                INSERT INTO embeddings_768 (episode_id, model_id, embedding)
                VALUES ($1, $2, $3);
                """,
                cast(UUID, episode_row["id"]),
                embedding_model_pk,
                list(vector),
            )

        return MessageEpisodeRecord(
            message=_message_from_row(message_row),
            episode=_episode_from_row(episode_row, embedding_model_pk),
        )

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

    async def retrieve_scoped_episodes(
        self,
        user_id: UUID,
        scope_id: UUID,
        query: str,
        top_k: int,
        *,
        now: datetime | None = None,
    ) -> list[ScoredEpisode]:
        """Retrieve embedded episodes from exactly one user-owned scope."""

        if top_k <= 0:
            raise InvalidRetrievalRequestError("top_k must be greater than zero")

        scored_at = _resolve_scoring_now(now)

        query_vector = await self.embedder.embed_text(query)
        self._validate_embedding_vector(query_vector)
        # Accepted Stage 5.2 heuristic: fetch a bounded similarity-first
        # candidate set, then rerank in Python. Lower-similarity episodes with
        # high recency/importance may be missed until eval tuning improves this.
        candidate_limit = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, MIN_RETRIEVAL_CANDIDATES)

        async with self.pool.acquire() as connection, connection.transaction():
            await self._ensure_scope_belongs_to_user(
                connection=connection,
                user_id=user_id,
                scope_id=scope_id,
            )
            embedding_model_pk = await self._embedding_model_pk(connection)
            rows = await connection.fetch(
                """
                SELECT
                    episodes.id,
                    conversations.user_id,
                    episodes.scope_id,
                    episodes.conversation_id,
                    episodes.kind,
                    episodes.message_id,
                    messages.position AS message_position,
                    episodes.range_start,
                    episodes.range_end,
                    episodes.content,
                    episodes.created_at,
                    episodes.importance,
                    episodes.access_count,
                    episodes.last_accessed_at,
                    embeddings_768.model_id AS embedding_model_id,
                    -- pgvector cosine distance is 0 for identical vectors; convert it
                    -- to similarity so larger values rank higher in the SQL
                    -- candidate pool. Final weighted scoring happens in Python.
                    1.0 - (embeddings_768.embedding <=> $3::vector) AS similarity
                FROM episodes
                INNER JOIN conversations
                    ON conversations.id = episodes.conversation_id
                   AND conversations.scope_id = episodes.scope_id
                INNER JOIN embeddings_768
                    ON embeddings_768.episode_id = episodes.id
                   AND embeddings_768.model_id = $4
                LEFT JOIN messages
                    ON messages.id = episodes.message_id
                   AND messages.conversation_id = episodes.conversation_id
                WHERE episodes.scope_id = $1
                  AND conversations.user_id = $2
                ORDER BY similarity DESC, episodes.created_at DESC, episodes.id ASC
                LIMIT $5;
                """,
                scope_id,
                user_id,
                list(query_vector),
                embedding_model_pk,
                candidate_limit,
            )

            scored_candidates = [_scored_episode_candidate_from_row(row, scored_at) for row in rows]
            retrieved_episodes = [
                _scored_episode_from_candidate(candidate, result_rank)
                for result_rank, candidate in enumerate(
                    sorted(scored_candidates, key=_scored_episode_sort_key)[:top_k],
                    start=1,
                )
            ]
            if retrieved_episodes:
                await self._update_retrieved_episode_access_metadata(
                    connection=connection,
                    user_id=user_id,
                    scope_id=scope_id,
                    episode_ids=[episode.id for episode in retrieved_episodes],
                    accessed_at=scored_at,
                )

        return retrieved_episodes

    async def record_used_memories(
        self,
        user_id: UUID,
        scope_id: UUID,
        query_message_id: UUID,
        assistant_message_id: UUID,
        used: Sequence[ScoredEpisode],
        scoring_version: str = SCORING_VERSION,
        retrieved_at: datetime | None = None,
    ) -> None:
        """Persist immutable provenance snapshots for memories used in one assistant response."""

        prepared = self._prepare_used_memories(
            used=used,
            scoring_version=scoring_version,
            retrieved_at=retrieved_at,
        )
        if not prepared.used_episodes:
            return

        async with self.pool.acquire() as connection, connection.transaction():
            query_conversation_id = await self._provenance_message_conversation_id(
                connection=connection,
                user_id=user_id,
                scope_id=scope_id,
                query_message_id=query_message_id,
                assistant_message_id=assistant_message_id,
            )
            await self._ensure_used_episodes_belong_to_scope_user(
                connection=connection,
                user_id=user_id,
                scope_id=scope_id,
                episode_ids=prepared.episode_ids,
            )
            await self._insert_message_retrievals(
                connection=connection,
                query_message_id=query_message_id,
                assistant_message_id=assistant_message_id,
                query_conversation_id=query_conversation_id,
                scope_id=scope_id,
                used=prepared.used_episodes,
                scoring_version=prepared.scoring_version,
                retrieved_at=prepared.retrieved_at,
            )

    async def append_assistant_response_with_provenance(
        self,
        request: AppendAssistantResponseWithProvenanceRequest,
    ) -> AssistantResponseRecord:
        """Append an assistant message and its used-memory provenance atomically."""

        if request.token_count < 0:
            raise InvalidMemoryRequestError("token_count must be non-negative")

        prepared = self._prepare_used_memories(
            used=request.used,
            scoring_version=request.scoring_version or SCORING_VERSION,
            retrieved_at=request.retrieved_at,
        )

        async with self.pool.acquire() as connection, connection.transaction():
            query_context = await self._provenance_query_message_context(
                connection=connection,
                user_id=request.user_id,
                scope_id=request.scope_id,
                conversation_id=request.conversation_id,
                query_message_id=request.query_message_id,
            )
            if prepared.used_episodes:
                await self._ensure_used_episodes_belong_to_scope_user(
                    connection=connection,
                    user_id=request.user_id,
                    scope_id=request.scope_id,
                    episode_ids=prepared.episode_ids,
                )
            assistant_row = await connection.fetchrow(
                """
                INSERT INTO messages (conversation_id, position, role, content, token_count)
                VALUES (
                    $1,
                    (
                        SELECT COALESCE(MAX(position), 0) + 1
                        FROM messages
                        WHERE conversation_id = $1
                    ),
                    'assistant',
                    $2,
                    $3
                )
                RETURNING id, conversation_id, position, role, content, token_count, created_at;
                """,
                request.conversation_id,
                request.content,
                request.token_count,
            )
            if assistant_row is None:
                raise ConversationNotFoundError("Assistant response was not appended")

            assistant_message_id = cast(UUID, assistant_row["id"])
            if prepared.used_episodes:
                await self._insert_message_retrievals(
                    connection=connection,
                    query_message_id=request.query_message_id,
                    assistant_message_id=assistant_message_id,
                    query_conversation_id=query_context.conversation_id,
                    scope_id=request.scope_id,
                    used=prepared.used_episodes,
                    scoring_version=prepared.scoring_version,
                    retrieved_at=prepared.retrieved_at,
                )

        return AssistantResponseRecord(
            message=_message_from_row(assistant_row),
            used_episode_ids=tuple(prepared.episode_ids),
            scoring_version=prepared.scoring_version,
            retrieved_at=prepared.retrieved_at,
        )

    async def _get_message_for_scope(
        self,
        request: CreateMessageEpisodeRequest,
    ) -> MessageRecord:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT
                    conversations.user_id,
                    conversations.scope_id,
                    messages.id,
                    messages.conversation_id,
                    messages.position,
                    messages.role,
                    messages.content,
                    messages.token_count,
                    messages.created_at
                FROM conversations
                LEFT JOIN messages
                    ON messages.conversation_id = conversations.id
                   AND messages.id = $2
                WHERE conversations.id = $1;
                """,
                request.conversation_id,
                request.message_id,
            )

        if row is None:
            raise ConversationNotFoundError("Conversation does not exist")
        if cast(UUID, row["user_id"]) != request.user_id:
            raise ConversationAccessDeniedError("Conversation belongs to a different user")
        if cast(UUID, row["scope_id"]) != request.scope_id or row["id"] is None:
            raise ConversationNotFoundError("Message is not in the expected conversation scope")
        return _message_from_row(row)

    async def _ensure_scope_belongs_to_user(
        self,
        connection: asyncpg.Connection,
        user_id: UUID,
        scope_id: UUID,
    ) -> None:
        scope_owner_id = cast(
            UUID | None,
            await connection.fetchval(
                """
                SELECT user_id
                FROM scopes
                WHERE id = $1;
                """,
                scope_id,
            ),
        )
        if scope_owner_id is None:
            raise ScopeNotFoundError("Scope does not exist")
        if scope_owner_id != user_id:
            raise ScopeAccessDeniedError("Scope belongs to a different user")

    def _prepare_used_memories(
        self,
        used: Sequence[ScoredEpisode],
        scoring_version: str,
        retrieved_at: datetime | None,
    ) -> _PreparedUsedMemories:
        used_episodes = list(used)
        if scoring_version.strip() == "":
            raise InvalidRetrievalRequestError("scoring_version must not be empty")

        resolved_retrieved_at = (
            datetime.now(timezone.utc)  # noqa: UP017
            if retrieved_at is None
            else _ensure_aware_utc(retrieved_at)
        )
        episode_ids = [episode.id for episode in used_episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise InvalidRetrievalRequestError("used episodes must not contain duplicate ids")

        return _PreparedUsedMemories(
            used_episodes=used_episodes,
            episode_ids=episode_ids,
            scoring_version=scoring_version,
            retrieved_at=resolved_retrieved_at,
        )

    async def _provenance_query_message_context(
        self,
        connection: asyncpg.Connection,
        user_id: UUID,
        scope_id: UUID,
        conversation_id: UUID,
        query_message_id: UUID,
    ) -> _ProvenanceQueryMessageContext:
        row = await connection.fetchrow(
            """
            SELECT
                query_messages.conversation_id AS query_conversation_id,
                query_messages.position AS query_position,
                query_messages.role AS query_role,
                query_conversations.user_id AS query_user_id,
                query_conversations.scope_id AS query_scope_id
            FROM messages AS query_messages
            INNER JOIN conversations AS query_conversations
                ON query_conversations.id = query_messages.conversation_id
            WHERE query_messages.id = $1
            FOR UPDATE OF query_conversations;
            """,
            query_message_id,
        )
        query_conversation_id = self._validate_provenance_query_message(
            row=row,
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
        )
        if row is None:
            raise ConversationNotFoundError("Query message does not exist")
        return _ProvenanceQueryMessageContext(
            conversation_id=query_conversation_id,
            position=cast(int, row["query_position"]),
        )

    async def _provenance_message_conversation_id(
        self,
        connection: asyncpg.Connection,
        user_id: UUID,
        scope_id: UUID,
        query_message_id: UUID,
        assistant_message_id: UUID,
    ) -> UUID:
        row = await connection.fetchrow(
            """
            SELECT
                query_messages.conversation_id AS query_conversation_id,
                query_messages.role AS query_role,
                query_conversations.user_id AS query_user_id,
                query_conversations.scope_id AS query_scope_id,
                assistant_messages.conversation_id AS assistant_conversation_id,
                assistant_messages.role AS assistant_role,
                assistant_conversations.user_id AS assistant_user_id,
                assistant_conversations.scope_id AS assistant_scope_id
            FROM messages AS query_messages
            INNER JOIN conversations AS query_conversations
                ON query_conversations.id = query_messages.conversation_id
            LEFT JOIN messages AS assistant_messages
                ON assistant_messages.id = $2
            LEFT JOIN conversations AS assistant_conversations
                ON assistant_conversations.id = assistant_messages.conversation_id
            WHERE query_messages.id = $1;
            """,
            query_message_id,
            assistant_message_id,
        )
        query_conversation_id = self._validate_provenance_query_message(
            row=row,
            user_id=user_id,
            scope_id=scope_id,
        )
        self._validate_provenance_assistant_message(
            row=row,
            user_id=user_id,
            scope_id=scope_id,
            query_conversation_id=query_conversation_id,
        )

        return query_conversation_id

    def _validate_provenance_query_message(
        self,
        row: asyncpg.Record | None,
        user_id: UUID,
        scope_id: UUID,
        conversation_id: UUID | None = None,
    ) -> UUID:
        if row is None:
            raise ConversationNotFoundError("Query message does not exist")
        if cast(UUID, row["query_user_id"]) != user_id:
            raise ConversationAccessDeniedError("Conversation belongs to a different user")
        if cast(UUID, row["query_scope_id"]) != scope_id:
            raise ConversationNotFoundError("Query message is not in the expected scope")
        query_conversation_id = cast(UUID, row["query_conversation_id"])
        if conversation_id is not None and query_conversation_id != conversation_id:
            raise ConversationNotFoundError("Query message is not in the expected conversation")
        if cast(str, row["query_role"]) != "user":
            raise InvalidProvenanceTargetError("Retrieval provenance requires a user query message")
        return query_conversation_id

    def _validate_provenance_assistant_message(
        self,
        row: asyncpg.Record | None,
        user_id: UUID,
        scope_id: UUID,
        query_conversation_id: UUID,
    ) -> None:
        if row is None:
            raise ConversationNotFoundError("Query message does not exist")
        if row["assistant_conversation_id"] is None:
            raise ConversationNotFoundError("Assistant message does not exist")
        if cast(UUID, row["assistant_user_id"]) != user_id:
            raise ConversationAccessDeniedError("Conversation belongs to a different user")
        if cast(UUID, row["assistant_scope_id"]) != scope_id:
            raise ConversationNotFoundError("Assistant message is not in the expected scope")
        assistant_conversation_id = cast(UUID, row["assistant_conversation_id"])
        if assistant_conversation_id != query_conversation_id:
            raise ConversationNotFoundError("Assistant message is not in the query conversation")
        if cast(str, row["assistant_role"]) != "assistant":
            raise InvalidProvenanceTargetError(
                "Retrieval provenance requires an assistant response message"
            )

    async def _ensure_used_episodes_belong_to_scope_user(
        self,
        connection: asyncpg.Connection,
        user_id: UUID,
        scope_id: UUID,
        episode_ids: list[UUID],
    ) -> None:
        rows = await connection.fetch(
            """
            SELECT episodes.id
            FROM episodes
            INNER JOIN conversations
                ON conversations.id = episodes.conversation_id
               AND conversations.scope_id = episodes.scope_id
            WHERE episodes.id = ANY($1::uuid[])
              AND episodes.scope_id = $2
              AND conversations.user_id = $3;
            """,
            episode_ids,
            scope_id,
            user_id,
        )
        found_episode_ids = {cast(UUID, row["id"]) for row in rows}
        if found_episode_ids != set(episode_ids):
            raise InvalidRetrievalRequestError("used episodes must belong to the expected scope")

    async def _insert_message_retrievals(
        self,
        connection: asyncpg.Connection,
        query_message_id: UUID,
        assistant_message_id: UUID,
        query_conversation_id: UUID,
        scope_id: UUID,
        used: Sequence[ScoredEpisode],
        scoring_version: str,
        retrieved_at: datetime,
    ) -> None:
        await connection.executemany(
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
                scoring_version,
                retrieved_at
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            );
            """,
            [
                (
                    query_message_id,
                    assistant_message_id,
                    query_conversation_id,
                    scope_id,
                    episode.id,
                    episode.embedding_model_id,
                    episode.result_rank,
                    episode.similarity,
                    episode.recency_score,
                    episode.access_score,
                    episode.importance_score,
                    episode.frequency_score,
                    episode.score,
                    scoring_version,
                    retrieved_at,
                )
                for episode in used
            ],
        )

    async def _conversation_scope_for_user(
        self,
        connection: asyncpg.Connection,
        user_id: UUID,
        conversation_id: UUID,
    ) -> UUID:
        row = await connection.fetchrow(
            """
            SELECT user_id, scope_id
            FROM conversations
            WHERE id = $1;
            """,
            conversation_id,
        )
        if row is None:
            raise ConversationNotFoundError("Conversation does not exist")
        if cast(UUID, row["user_id"]) != user_id:
            raise ConversationAccessDeniedError("Conversation belongs to a different user")
        return cast(UUID, row["scope_id"])

    async def _lock_conversation_for_user(
        self,
        connection: asyncpg.Connection,
        user_id: UUID,
        conversation_id: UUID,
    ) -> UUID:
        row = await connection.fetchrow(
            """
            SELECT user_id, scope_id
            FROM conversations
            WHERE id = $1
            FOR UPDATE;
            """,
            conversation_id,
        )
        if row is None:
            raise ConversationNotFoundError("Conversation does not exist")
        if cast(UUID, row["user_id"]) != user_id:
            raise ConversationAccessDeniedError("Conversation belongs to a different user")
        return cast(UUID, row["scope_id"])

    async def _lock_conversation(
        self,
        connection: asyncpg.Connection,
        user_id: UUID,
        scope_id: UUID,
        conversation_id: UUID,
    ) -> None:
        actual_scope_id = await self._lock_conversation_for_user(
            connection=connection,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if actual_scope_id != scope_id:
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

    async def _update_retrieved_episode_access_metadata(
        self,
        connection: asyncpg.Connection,
        user_id: UUID,
        scope_id: UUID,
        episode_ids: list[UUID],
        accessed_at: datetime,
    ) -> None:
        await connection.execute(
            """
            UPDATE episodes
            SET access_count = access_count + 1,
                last_accessed_at = $4
            FROM conversations
            WHERE episodes.id = ANY($3::uuid[])
              AND episodes.scope_id = $1
              AND conversations.id = episodes.conversation_id
              AND conversations.scope_id = episodes.scope_id
              AND conversations.user_id = $2;
            """,
            scope_id,
            user_id,
            episode_ids,
            accessed_at,
        )

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


def _scored_episode_candidate_from_row(
    row: asyncpg.Record,
    scored_at: datetime,
) -> _ScoredEpisodeCandidate:
    similarity = cast(float, row["similarity"])
    created_at = cast(datetime, row["created_at"])
    access_count = cast(int, row["access_count"])
    last_accessed_at = cast(datetime | None, row["last_accessed_at"])
    recency_score = _half_life_score(
        timestamp=created_at,
        now=scored_at,
        half_life_seconds=RECENCY_HALF_LIFE_SECONDS,
    )
    access_score = (
        0.0
        if last_accessed_at is None
        else _half_life_score(
            timestamp=last_accessed_at,
            now=scored_at,
            half_life_seconds=ACCESS_HALF_LIFE_SECONDS,
        )
    )
    importance_score = _clamp_score(cast(float, row["importance"]))
    frequency_score = _frequency_score(access_count)
    score = (
        SIMILARITY_WEIGHT * similarity
        + RECENCY_WEIGHT * recency_score
        + ACCESS_WEIGHT * access_score
        + IMPORTANCE_WEIGHT * importance_score
        + FREQUENCY_WEIGHT * frequency_score
    )

    return _ScoredEpisodeCandidate(
        id=cast(UUID, row["id"]),
        user_id=cast(UUID, row["user_id"]),
        scope_id=cast(UUID, row["scope_id"]),
        conversation_id=cast(UUID, row["conversation_id"]),
        kind=cast(EpisodeKind, row["kind"]),
        message_id=cast(UUID | None, row["message_id"]),
        message_position=cast(int | None, row["message_position"]),
        range_start=cast(int | None, row["range_start"]),
        range_end=cast(int | None, row["range_end"]),
        content=cast(str, row["content"]),
        created_at=created_at,
        importance=cast(float, row["importance"]),
        access_count=access_count,
        last_accessed_at=last_accessed_at,
        embedding_model_id=cast(int, row["embedding_model_id"]),
        similarity=similarity,
        recency_score=recency_score,
        access_score=access_score,
        importance_score=importance_score,
        frequency_score=frequency_score,
        score=score,
    )


def _scored_episode_from_candidate(
    candidate: _ScoredEpisodeCandidate,
    result_rank: int,
) -> ScoredEpisode:
    return ScoredEpisode(
        result_rank=result_rank,
        id=candidate.id,
        user_id=candidate.user_id,
        scope_id=candidate.scope_id,
        conversation_id=candidate.conversation_id,
        kind=candidate.kind,
        message_id=candidate.message_id,
        message_position=candidate.message_position,
        range_start=candidate.range_start,
        range_end=candidate.range_end,
        content=candidate.content,
        created_at=candidate.created_at,
        importance=candidate.importance,
        access_count=candidate.access_count,
        last_accessed_at=candidate.last_accessed_at,
        embedding_model_id=candidate.embedding_model_id,
        similarity=candidate.similarity,
        recency_score=candidate.recency_score,
        access_score=candidate.access_score,
        importance_score=candidate.importance_score,
        frequency_score=candidate.frequency_score,
        score=candidate.score,
    )


def _scored_episode_sort_key(episode: _ScoredEpisodeCandidate) -> tuple[float, float, int]:
    return (
        -episode.score,
        -_datetime_timestamp(episode.created_at),
        episode.id.int,
    )


def _half_life_score(timestamp: datetime, now: datetime, half_life_seconds: int) -> float:
    age_seconds = max(0.0, _datetime_timestamp(now) - _datetime_timestamp(timestamp))
    return _clamp_score(0.5 ** (age_seconds / half_life_seconds))


def _frequency_score(access_count: int) -> float:
    safe_access_count = max(0.0, float(access_count))
    return _clamp_score(safe_access_count / (safe_access_count + FREQUENCY_NORMALIZATION_COUNT))


def _clamp_score(value: float) -> float:
    return min(1.0, max(0.0, value))


def _datetime_timestamp(value: datetime) -> float:
    return _ensure_aware_utc(value).timestamp()


def _resolve_scoring_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)  # noqa: UP017
    return _ensure_aware_utc(now)


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidRetrievalRequestError("retrieval scoring requires timezone-aware datetimes")
    return value.astimezone(timezone.utc)  # noqa: UP017
