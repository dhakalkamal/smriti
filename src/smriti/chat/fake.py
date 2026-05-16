from __future__ import annotations

from dataclasses import dataclass, field

from smriti.chat.base import ChatRequest, ChatResponse
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
