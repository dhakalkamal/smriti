from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import ParseResult, urlparse

import httpx

from smriti.chat.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    ChatStreamFinal,
    ChatStreamToken,
    ChatUsage,
)
from smriti.chat.errors import (
    ChatConfigurationError,
    ChatConnectionError,
    ChatResponseError,
    ChatTimeoutError,
)

_LOCALHOST_NAMES = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class OllamaChatGenerator:
    """Async localhost-only chat generator for Ollama's `/api/chat` endpoint."""

    model: str = "qwen2.5:7b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 60.0
    num_ctx: int = 8192

    def __post_init__(self) -> None:
        if not self.model:
            raise ChatConfigurationError("Ollama model name must not be empty")
        if self.timeout_seconds <= 0:
            raise ChatConfigurationError("Ollama timeout must be positive")
        if self.num_ctx <= 0:
            raise ChatConfigurationError("Ollama context window must be positive")

        self._parse_base_url()

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Generate one complete assistant response through local Ollama."""

        payload: dict[str, object] = {
            "model": self.model,
            "messages": [self._message_payload(message) for message in request.messages],
            "stream": False,
            "options": {"num_ctx": self.num_ctx},
        }

        response = await self._post_json(payload)
        return self._parse_chat_response(response)

    async def generate_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        """Stream one assistant response through local Ollama."""

        payload: dict[str, object] = {
            "model": self.model,
            "messages": [self._message_payload(message) for message in request.messages],
            "stream": True,
            "options": {"num_ctx": self.num_ctx},
        }

        async for event in self._stream_json(payload):
            yield event

    async def _post_json(self, payload: dict[str, object]) -> Any:
        request_url = self._chat_url()

        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=self.timeout_seconds,
                trust_env=False,
            ) as client:
                response = await client.post(request_url, json=payload)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ChatTimeoutError(
                f"Ollama chat request timed out after {self.timeout_seconds} seconds"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)
            raise ChatResponseError(f"Ollama chat request failed: {detail}") from exc
        except httpx.RequestError as exc:
            message = f"Could not connect to Ollama at {self.base_url}"
            raise ChatConnectionError(message) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise ChatResponseError("Ollama returned non-JSON chat response") from exc

    async def _stream_json(self, payload: dict[str, object]) -> AsyncIterator[ChatStreamEvent]:
        request_url = self._chat_url()
        saw_final = False

        try:
            async with (
                httpx.AsyncClient(
                    follow_redirects=False,
                    timeout=self.timeout_seconds,
                    trust_env=False,
                ) as client,
                client.stream("POST", request_url, json=payload) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    event = self._parse_chat_stream_event(self._parse_stream_line(line))
                    if isinstance(event, ChatStreamFinal):
                        saw_final = True
                        yield event
                        return
                    if event.text:
                        yield event
        except httpx.TimeoutException as exc:
            raise ChatTimeoutError(
                f"Ollama chat request timed out after {self.timeout_seconds} seconds"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)
            raise ChatResponseError(f"Ollama chat request failed: {detail}") from exc
        except httpx.RequestError as exc:
            message = f"Could not connect to Ollama at {self.base_url}"
            raise ChatConnectionError(message) from exc

        if not saw_final:
            raise ChatResponseError("Ollama chat stream ended without final response")

    def _parse_base_url(self) -> ParseResult:
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme != "http":
            raise ChatConfigurationError("Ollama base URL must use http")
        if parsed_url.hostname not in _LOCALHOST_NAMES:
            raise ChatConfigurationError("Ollama base URL must point to localhost")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise ChatConfigurationError("Ollama base URL must not include credentials")
        if parsed_url.query or parsed_url.fragment:
            raise ChatConfigurationError("Ollama base URL must not include query or fragment")
        return parsed_url

    def _chat_url(self) -> str:
        parsed_url = self._parse_base_url()
        path_prefix = parsed_url.path.rstrip("/")
        request_path = f"{path_prefix}/api/chat" if path_prefix else "/api/chat"
        return parsed_url._replace(path=request_path).geturl()

    def _extract_error_detail(self, response: httpx.Response) -> str:
        try:
            parsed_body = response.json()
        except ValueError:
            return response.text or "unexpected HTTP error"

        if isinstance(parsed_body, dict):
            detail = parsed_body.get("error")
            if isinstance(detail, str):
                return detail
        return "unexpected HTTP error"

    def _parse_chat_response(self, response: Any) -> ChatResponse:
        if not isinstance(response, dict):
            raise ChatResponseError("Ollama chat response must be a JSON object")

        model = response.get("model")
        if not isinstance(model, str) or not model:
            raise ChatResponseError("Ollama chat response is missing model")

        message = response.get("message")
        if not isinstance(message, dict):
            raise ChatResponseError("Ollama chat response is missing message")

        role = message.get("role")
        if role != "assistant":
            raise ChatResponseError("Ollama chat response message role must be assistant")

        content = message.get("content")
        if not isinstance(content, str):
            raise ChatResponseError("Ollama chat response is missing message content")

        return ChatResponse(
            content=content,
            model=model,
            finish_reason=self._optional_string(response.get("done_reason")),
            usage=ChatUsage(
                prompt_tokens=self._optional_int(response.get("prompt_eval_count")),
                completion_tokens=self._optional_int(response.get("eval_count")),
            ),
        )

    def _parse_stream_line(self, line: str) -> Any:
        try:
            return json.loads(line)
        except ValueError as exc:
            raise ChatResponseError("Ollama returned non-JSON chat stream event") from exc

    def _parse_chat_stream_event(self, response: Any) -> ChatStreamEvent:
        if not isinstance(response, dict):
            raise ChatResponseError("Ollama chat stream event must be a JSON object")

        if response.get("done") is True:
            model = response.get("model")
            if not isinstance(model, str) or not model:
                raise ChatResponseError("Ollama final chat stream event is missing model")
            return ChatStreamFinal(
                model=model,
                finish_reason=self._optional_string(response.get("done_reason")),
                usage=ChatUsage(
                    prompt_tokens=self._optional_int(response.get("prompt_eval_count")),
                    completion_tokens=self._optional_int(response.get("eval_count")),
                ),
            )

        message = response.get("message")
        if not isinstance(message, dict):
            raise ChatResponseError("Ollama chat stream event is missing message")

        role = message.get("role")
        if role is not None and role != "assistant":
            raise ChatResponseError("Ollama chat stream message role must be assistant")

        content = message.get("content")
        if not isinstance(content, str):
            raise ChatResponseError("Ollama chat stream event is missing message content")

        return ChatStreamToken(text=content)

    def _message_payload(self, message: ChatMessage) -> dict[str, str]:
        return {"role": message.role, "content": message.content}

    def _optional_int(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        raise ChatResponseError("Ollama chat response contains a non-integer token count")

    def _optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        raise ChatResponseError("Ollama chat response contains a non-string finish reason")
