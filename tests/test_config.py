from __future__ import annotations

import pytest
from pydantic import ValidationError

from smriti.config import Settings


def test_settings_defaults_include_local_ollama_chat_configuration() -> None:
    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ollama_chat_model == "qwen2.5:7b"
    assert settings.ollama_chat_num_ctx == 8192
    assert settings.ollama_embed_num_ctx == 8192
    assert settings.ollama_chat_timeout_seconds == 60.0
    assert settings.summary_episode_memory_enabled is False
    assert settings.summary_episode_window_messages == 12
    assert settings.memory_policy == "legacy"
    assert settings.memory_typed_v1_total_limit == 6
    assert settings.memory_typed_v1_raw_source_limit == 4
    assert settings.memory_typed_v1_summary_source_limit == 2
    assert settings.memory_typed_v1_assistant_derived_limit == 0


def test_settings_loads_local_ollama_chat_configuration_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SMRITI_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("SMRITI_OLLAMA_CHAT_MODEL", "qwen3:8b")
    monkeypatch.setenv("SMRITI_OLLAMA_CHAT_NUM_CTX", "16384")
    monkeypatch.setenv("SMRITI_OLLAMA_EMBED_NUM_CTX", "4096")
    monkeypatch.setenv("SMRITI_OLLAMA_CHAT_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("SMRITI_SUMMARY_EPISODE_MEMORY_ENABLED", "true")
    monkeypatch.setenv("SMRITI_SUMMARY_EPISODE_WINDOW_MESSAGES", "24")
    monkeypatch.setenv("SMRITI_MEMORY_POLICY", "typed_v1")
    monkeypatch.setenv("SMRITI_MEMORY_TYPED_V1_TOTAL_LIMIT", "8")
    monkeypatch.setenv("SMRITI_MEMORY_TYPED_V1_RAW_SOURCE_LIMIT", "5")
    monkeypatch.setenv("SMRITI_MEMORY_TYPED_V1_SUMMARY_SOURCE_LIMIT", "3")
    monkeypatch.setenv("SMRITI_MEMORY_TYPED_V1_ASSISTANT_DERIVED_LIMIT", "1")

    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_chat_model == "qwen3:8b"
    assert settings.ollama_chat_num_ctx == 16384
    assert settings.ollama_embed_num_ctx == 4096
    assert settings.ollama_chat_timeout_seconds == 12.5
    assert settings.summary_episode_memory_enabled is True
    assert settings.summary_episode_window_messages == 24
    assert settings.memory_policy == "typed_v1"
    assert settings.memory_typed_v1_total_limit == 8
    assert settings.memory_typed_v1_raw_source_limit == 5
    assert settings.memory_typed_v1_summary_source_limit == 3
    assert settings.memory_typed_v1_assistant_derived_limit == 1


def test_settings_rejects_invalid_local_ollama_chat_configuration() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_chat_model="")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_chat_timeout_seconds=0.0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_chat_num_ctx=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_embed_num_ctx=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, summary_episode_window_messages=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, memory_policy="unsupported")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, memory_typed_v1_total_limit=-1)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, memory_typed_v1_assistant_derived_limit=-1)
