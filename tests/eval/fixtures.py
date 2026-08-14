from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast
from uuid import UUID

import asyncpg

from smriti.memory import (
    AppendMessageWithEpisodeRequest,
    CreateConversationRequest,
    CreateScopeRequest,
    MemoryService,
    MessageRole,
)
from smriti.memory.eval import (
    Stage12Corpus,
    Stage12ResolvedEvalCase,
    TimingMode,
    resolve_stage12_corpus_cases,
)

TERRAFOLD_SCENARIO_ID = "tunde-terrafold-planted-facts-v1"

SUMMARY_13_24 = (
    "Tunde chose Terrafold as the studio name and rejected Kilnhouse. "
    "The landlord is Mr. Obafemi, and the lease signing is Friday, July 31, "
    "in person. The kiln budget has a hard cap of $3,650. Later recap turns "
    "asked about the name and kiln cap, and assistant replies repeated those facts."
)
SUMMARY_25_36 = (
    "Tunde has a severe latex allergy, so all gloves must be nitrile. Cousin "
    "Dele is the silent partner handling bookkeeping. Classes are capped at 9 "
    "because there are exactly 9 pottery wheels."
)


@dataclass(frozen=True)
class TerrafoldFixture:
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID
    ref_to_episode_id: Mapping[str, UUID]
    episode_ids: tuple[UUID, ...]


async def build_terrafold_fixture(
    service: MemoryService,
    corpus: Stage12Corpus,
    timing_mode: TimingMode = "app_realistic",
) -> tuple[TerrafoldFixture, list[Stage12ResolvedEvalCase]]:
    """Build the deterministic Terrafold Stage 12a fixture for one eval run."""

    user_id = await _create_user(service.pool)
    scope = await service.create_scope(
        CreateScopeRequest(
            user_id=user_id,
            name="Stage 12 Terrafold Eval",
            system_prompt="Keep the Terrafold eval scoped and local.",
        )
    )
    conversation = await service.create_conversation(
        CreateConversationRequest(
            user_id=user_id,
            scope_id=scope.id,
            title="Tunde Terrafold planted facts",
        )
    )

    ref_to_episode_id: dict[str, UUID] = {}
    episode_ids: list[UUID] = []

    for message in _base_messages():
        record = await service.append_message_with_episode(
            AppendMessageWithEpisodeRequest(
                user_id=user_id,
                conversation_id=conversation.id,
                role=message.role,
                content=message.content,
                token_count=len(message.content.split()),
            )
        )
        episode_ids.append(record.episode.id)
        if message.semantic_ref is not None:
            ref_to_episode_id[message.semantic_ref] = record.episode.id

        # Seed summary rows directly with explicit ranges: the fixture
        # intentionally has no 1..12 summary, while the production service now
        # catches up from the earliest uncovered window.
        if record.message.position == 24:
            summary_id = await _seed_summary_episode(
                service=service,
                conversation_id=conversation.id,
                scope_id=scope.id,
                range_start=13,
                range_end=24,
                content=SUMMARY_13_24,
            )
            ref_to_episode_id["summary_window_13_24"] = summary_id
            episode_ids.append(summary_id)

        if record.message.position == 36:
            summary_id = await _seed_summary_episode(
                service=service,
                conversation_id=conversation.id,
                scope_id=scope.id,
                range_start=25,
                range_end=36,
                content=SUMMARY_25_36,
            )
            ref_to_episode_id["summary_window_25_36"] = summary_id
            episode_ids.append(summary_id)

    current_query_conversation_id = conversation.id
    if timing_mode == "clean_memory":
        current_query_scope = await service.create_scope(
            CreateScopeRequest(
                user_id=user_id,
                name="Stage 12 Current Query Controls",
                system_prompt="Holds out current-query controls for clean-memory evals.",
            )
        )
        current_query_conversation = await service.create_conversation(
            CreateConversationRequest(
                user_id=user_id,
                scope_id=current_query_scope.id,
                title="Current query controls",
            )
        )
        current_query_conversation_id = current_query_conversation.id

    for semantic_ref, query in _current_query_refs(corpus).items():
        record = await service.append_message_with_episode(
            AppendMessageWithEpisodeRequest(
                user_id=user_id,
                conversation_id=current_query_conversation_id,
                role="user",
                content=query,
                token_count=len(query.split()),
            )
        )
        ref_to_episode_id[semantic_ref] = record.episode.id
        episode_ids.append(record.episode.id)

    fixture = TerrafoldFixture(
        user_id=user_id,
        scope_id=scope.id,
        conversation_id=conversation.id,
        ref_to_episode_id=ref_to_episode_id,
        episode_ids=tuple(episode_ids),
    )
    cases = resolve_stage12_corpus_cases(
        corpus=corpus,
        user_id=user_id,
        scope_id=scope.id,
        ref_to_episode_id=ref_to_episode_id,
    )
    return fixture, cases


async def reset_fixture_access_metadata(pool: asyncpg.Pool, episode_ids: tuple[UUID, ...]) -> None:
    """Reset retrieval mutation fields so eval cases do not affect later cases."""

    if not episode_ids:
        return

    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE episodes
            SET access_count = 0,
                last_accessed_at = NULL
            WHERE id = ANY($1::uuid[]);
            """,
            list(episode_ids),
        )


@dataclass(frozen=True)
class _FixtureMessage:
    role: MessageRole
    content: str
    semantic_ref: str | None = None


async def _create_user(pool: asyncpg.Pool) -> UUID:
    async with pool.acquire() as connection:
        return cast(
            UUID, await connection.fetchval("INSERT INTO users DEFAULT VALUES RETURNING id;")
        )


async def _seed_summary_episode(
    service: MemoryService,
    conversation_id: UUID,
    scope_id: UUID,
    range_start: int,
    range_end: int,
    content: str,
) -> UUID:
    vector = await service.embedder.embed_text(content)
    async with service.pool.acquire() as connection:
        embedding_model_pk = await connection.fetchval(
            """
            SELECT id
            FROM embedding_models
            WHERE model_id = $1
              AND dimensions = 768
              AND is_active = TRUE;
            """,
            service.embedding_model_id,
        )
        episode_id = await connection.fetchval(
            """
            INSERT INTO episodes (conversation_id, scope_id, kind, range_start, range_end, content)
            VALUES ($1, $2, 'summary', $3, $4, $5)
            RETURNING id;
            """,
            conversation_id,
            scope_id,
            range_start,
            range_end,
            content,
        )
        await connection.execute(
            """
            INSERT INTO embeddings_768 (episode_id, model_id, embedding)
            VALUES ($1, $2, $3);
            """,
            episode_id,
            embedding_model_pk,
            list(vector),
        )
    return cast(UUID, episode_id)


def _base_messages() -> tuple[_FixtureMessage, ...]:
    return (
        _FixtureMessage("user", "Tunde is sketching ideas for a neighborhood pottery studio."),
        _FixtureMessage("assistant", "Keep the pottery studio planning notes scoped."),
        _FixtureMessage("user", "The studio should feel calm, practical, and local."),
        _FixtureMessage("assistant", "I will keep the planning notes grounded in the messages."),
        _FixtureMessage("user", "Tunde is comparing lease timing, tools, and class setup."),
        _FixtureMessage("assistant", "Lease timing, tools, and class setup are in scope."),
        _FixtureMessage("user", "There may be signage, bookkeeping, and safety constraints later."),
        _FixtureMessage(
            "assistant", "I will wait for exact details before treating them as facts."
        ),
        _FixtureMessage("user", "Avoid inventing vendors, prices, dates, or names."),
        _FixtureMessage("assistant", "Understood. I will preserve only supplied facts."),
        _FixtureMessage("user", "This warmup exists so summaries start at the later window."),
        _FixtureMessage("assistant", "The first twelve messages are warmup context only."),
        _FixtureMessage(
            "user",
            "Tunde settled on the studio name Terrafold and rejected Kilnhouse.",
            "tunde_msg_01_studio_name",
        ),
        _FixtureMessage(
            "user",
            "The landlord is Mr. Obafemi, and lease signing is Friday, July 31, in person.",
            "tunde_msg_02_landlord_lease",
        ),
        _FixtureMessage(
            "user",
            "The kiln budget has a hard cap of $3,650.",
            "tunde_msg_03_kiln_budget",
        ),
        _FixtureMessage(
            "user",
            "Remind me later which studio name Tunde chose.",
            "tunde_recap_01_studio_name_question",
        ),
        _FixtureMessage(
            "assistant",
            "Tunde chose Terrafold and rejected Kilnhouse.",
            "tunde_echo_01_studio_name_answer",
        ),
        _FixtureMessage(
            "user",
            "Can you recap the kiln budget cap?",
            "tunde_recap_02_kiln_budget_question",
        ),
        _FixtureMessage(
            "assistant",
            "The kiln budget cap is $3,650.",
            "tunde_echo_02_kiln_budget_answer",
        ),
        _FixtureMessage(
            "user",
            "Kilnhouse appears in older brainstorming notes, but it is not the chosen name.",
            "tunde_distractor_01_kilnhouse",
        ),
        _FixtureMessage(
            "assistant", "Kilnhouse is older brainstorming context, not the chosen name."
        ),
        _FixtureMessage("user", "Tunde is still deciding the clay supplier."),
        _FixtureMessage("assistant", "The clay supplier is undecided."),
        _FixtureMessage("user", "End of the first fact-heavy window."),
        _FixtureMessage(
            "user",
            "Tunde has a severe latex allergy, so all gloves must be nitrile.",
            "tunde_msg_04_latex_allergy",
        ),
        _FixtureMessage(
            "user",
            "Cousin Dele is the silent partner handling bookkeeping.",
            "tunde_msg_05_dele_bookkeeping",
        ),
        _FixtureMessage(
            "user",
            "Class size is capped at 9 because there are exactly 9 pottery wheels.",
            "tunde_msg_06_class_size_wheels",
        ),
        _FixtureMessage(
            "user",
            "What was the safety note about latex again?",
            "tunde_recap_03_latex_question",
        ),
        _FixtureMessage(
            "assistant",
            "Tunde has a severe latex allergy, so gloves must be nitrile.",
            "tunde_echo_03_latex_answer",
        ),
        _FixtureMessage(
            "assistant",
            "Classes are capped at 9 because there are exactly 9 pottery wheels.",
            "tunde_echo_04_class_answer",
        ),
        _FixtureMessage(
            "user",
            "Vinyl aprons are optional, but glove material is not optional.",
            "tunde_distractor_02_vinyl_aprons",
        ),
        _FixtureMessage("assistant", "Vinyl aprons are a separate optional supply note."),
        _FixtureMessage("user", "Bookkeeping setup should stay simple for launch."),
        _FixtureMessage(
            "assistant",
            "Bookkeeping belongs with Dele's silent-partner role.",
            "tunde_echo_05_dele_bookkeeping_answer",
        ),
        _FixtureMessage("user", "Do not add extra wheels unless the budget changes."),
        _FixtureMessage("assistant", "The current wheel count remains exactly 9."),
    )


def _current_query_refs(corpus: Stage12Corpus) -> dict[str, str]:
    refs_to_query: dict[str, str] = {}
    for case in corpus.cases:
        for semantic_ref in case.expected_refs.current_query:
            refs_to_query[semantic_ref] = case.query
    return refs_to_query
