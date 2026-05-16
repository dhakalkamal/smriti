from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from smriti.chat.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    ChatStreamFinal,
    ChatStreamToken,
)
from smriti.chat.errors import ChatError


@dataclass
class FakeChatGenerator:
    """Deterministic, local-only chat generator for tests and eval fixtures."""

    response: ChatResponse = field(
        default_factory=lambda: ChatResponse(
            content="fake assistant response",
            model="fake-chat-generator",
            finish_reason="stop",
        )
    )
    responses: list[ChatResponse] = field(default_factory=list)
    error: ChatError | None = None
    requests: list[ChatRequest] = field(default_factory=list, init=False)

    @property
    def model(self) -> str:
        """Return the configured fake model identifier."""

        if self.responses:
            return self.responses[0].model
        return self.response.model

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Return a deterministic complete response without network access."""

        self.requests.append(request)

        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return self.response


@dataclass
class FakeStreamingChatGenerator:
    """Deterministic streaming chat generator for tests and local fixtures."""

    tokens: list[str] = field(default_factory=lambda: ["fake", " assistant", " response"])
    final: ChatStreamFinal | None = None
    error: ChatError | None = None
    fail_after_tokens: int | None = None
    requests: list[ChatRequest] = field(default_factory=list, init=False)

    @property
    def model(self) -> str:
        """Return the configured fake streaming model identifier."""

        if self.final is not None:
            return self.final.model
        return "fake-streaming-chat-generator"

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Collect the deterministic stream into one complete fake response."""

        content_parts: list[str] = []
        final: ChatStreamFinal | None = None
        async for event in self.generate_stream(request):
            if isinstance(event, ChatStreamToken):
                content_parts.append(event.text)
            else:
                final = event

        final_event = final or self._default_final()
        return ChatResponse(
            content="".join(content_parts),
            model=final_event.model,
            finish_reason=final_event.finish_reason,
            usage=final_event.usage,
        )

    async def generate_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        """Yield configured tokens and always terminate with final metadata unless failing."""

        self.requests.append(request)

        if self.error is not None and self.fail_after_tokens is None:
            raise self.error

        for emitted_tokens, token in enumerate(self.tokens):
            if self.error is not None and self.fail_after_tokens == emitted_tokens:
                raise self.error
            yield ChatStreamToken(text=token)

        if self.error is not None:
            raise self.error

        yield self.final or self._default_final()

    def _default_final(self) -> ChatStreamFinal:
        return ChatStreamFinal(model=self.model, finish_reason="stop")
