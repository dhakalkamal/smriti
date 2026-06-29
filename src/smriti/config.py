from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MemoryPolicy = Literal["legacy", "typed_v1"]
RetrievalCandidateMode = Literal["semantic", "hybrid_v1"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables.

    Centralizing configuration keeps local-only defaults explicit and makes
    service behavior deterministic across CLI, tests, and app entrypoints.
    """

    database_url: str = Field(default="postgresql://smriti:smriti@127.0.0.1:5432/smriti")
    database_min_pool_size: int = Field(default=1, ge=1)
    database_max_pool_size: int = Field(default=10, ge=1)
    database_command_timeout: float = Field(default=30.0, gt=0)
    database_connect_timeout: float = Field(default=10.0, gt=0)
    database_echo_sql: bool = Field(default=False)
    migrations_dir: Path = Field(default=Path("src/smriti/db/migrations"))
    local_user_id: UUID = Field(default=UUID("00000000-0000-4000-8000-000000000001"))
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", min_length=1)
    ollama_chat_model: str = Field(default="qwen2.5:7b", min_length=1)
    ollama_chat_num_ctx: int = Field(default=8192, gt=0)
    ollama_embed_num_ctx: int = Field(default=8192, gt=0)
    ollama_chat_timeout_seconds: float = Field(default=60.0, gt=0)
    summary_episode_memory_enabled: bool = Field(default=False)
    summary_episode_window_messages: int = Field(default=12, ge=1)
    memory_policy: MemoryPolicy = Field(default="legacy")
    memory_typed_v1_total_limit: int = Field(default=6, ge=0)
    memory_typed_v1_raw_source_limit: int = Field(default=4, ge=0)
    memory_typed_v1_summary_source_limit: int = Field(default=2, ge=0)
    memory_typed_v1_assistant_derived_limit: int = Field(default=0, ge=0)
    retrieval_candidate_mode: RetrievalCandidateMode = Field(default="semantic")

    model_config = SettingsConfigDict(
        env_prefix="SMRITI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance for the current process."""

    return Settings()
