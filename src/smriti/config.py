from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
