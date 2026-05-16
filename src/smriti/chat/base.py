from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One message passed to a local chat generator."""

    role: ChatRole
    content: str


@dataclass(frozen=True)
class ChatUsage:
    """Token counts reported by the local chat backend when available."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ChatRequest:
    """Complete non-streaming chat generation request."""

    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True)
class ChatResponse:
    """Complete non-streaming chat generation response."""

    content: str
    model: str
    finish_reason: str | None = None
    usage: ChatUsage = field(default_factory=ChatUsage)


@dataclass(frozen=True)
class ChatStreamToken:
    """One generated text fragment from a streaming chat backend."""

    text: str


@dataclass(frozen=True)
class ChatStreamFinal:
    """Final metadata from a successfully completed streaming chat response."""

    model: str
    finish_reason: str | None = None
    usage: ChatUsage = field(default_factory=ChatUsage)


ChatStreamEvent = ChatStreamToken | ChatStreamFinal


class ChatGenerator(Protocol):
    """Async interface for components that produce complete chat responses."""

    @property
    def model(self) -> str:
        """Return the configured chat model identifier."""

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Generate one complete assistant response."""


@runtime_checkable
class StreamingChatGenerator(Protocol):
    """Async interface for components that stream typed chat events."""

    @property
    def model(self) -> str:
        """Return the configured chat model identifier."""

    def generate_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        """Stream one assistant response as typed chunks and final metadata."""
