from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import ParseResult, urlparse

import httpx

from smriti.embeddings.base import EmbeddingVector
from smriti.embeddings.errors import (
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
)

_LOCALHOST_NAMES = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class OllamaEmbedder:
    """Async localhost-only embedder for Ollama's `/api/embed` endpoint."""

    model: str = "nomic-embed-text"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 30.0
    dimensions: int | None = None
    truncate: bool | None = None
    num_ctx: int = 8192

    def __post_init__(self) -> None:
        if not self.model:
            raise EmbeddingConfigurationError("Ollama model name must not be empty")
        if self.timeout_seconds <= 0:
            raise EmbeddingConfigurationError("Ollama timeout must be positive")
        if self.dimensions is not None and self.dimensions <= 0:
            raise EmbeddingConfigurationError("Ollama dimensions must be positive when set")
        if self.num_ctx <= 0:
            raise EmbeddingConfigurationError("Ollama context window must be positive")

        self._parse_base_url()

    async def embed_text(self, text: str) -> EmbeddingVector:
        """Embed one text value through local Ollama."""

        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed text values through local Ollama, preserving input order."""

        if not texts:
            return []

        payload: dict[str, object] = {
            "model": self.model,
            "input": list(texts),
            "options": {"num_ctx": self.num_ctx},
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        if self.truncate is not None:
            payload["truncate"] = self.truncate

        response = await self._post_json(payload)
        return self._parse_embeddings_response(response, expected_count=len(texts))

    async def _post_json(self, payload: dict[str, object]) -> Any:
        request_url = self._embed_url()

        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=self.timeout_seconds,
                trust_env=False,
            ) as client:
                response = await client.post(request_url, json=payload)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise EmbeddingTimeoutError(
                f"Ollama embedding request timed out after {self.timeout_seconds} seconds"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)
            raise EmbeddingResponseError(f"Ollama embedding request failed: {detail}") from exc
        except httpx.RequestError as exc:
            message = f"Could not connect to Ollama at {self.base_url}"
            raise EmbeddingConnectionError(message) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise EmbeddingResponseError("Ollama returned non-JSON embedding response") from exc

    def _parse_base_url(self) -> ParseResult:
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme != "http":
            raise EmbeddingConfigurationError("Ollama base URL must use http")
        if parsed_url.hostname not in _LOCALHOST_NAMES:
            raise EmbeddingConfigurationError("Ollama base URL must point to localhost")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise EmbeddingConfigurationError("Ollama base URL must not include credentials")
        if parsed_url.query or parsed_url.fragment:
            raise EmbeddingConfigurationError("Ollama base URL must not include query or fragment")
        return parsed_url

    def _embed_url(self) -> str:
        parsed_url = self._parse_base_url()
        path_prefix = parsed_url.path.rstrip("/")
        request_path = f"{path_prefix}/api/embed" if path_prefix else "/api/embed"
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

    def _parse_embeddings_response(
        self,
        response: Any,
        expected_count: int,
    ) -> list[EmbeddingVector]:
        if not isinstance(response, dict):
            raise EmbeddingResponseError("Ollama embedding response must be a JSON object")

        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingResponseError("Ollama embedding response is missing embeddings")
        if len(embeddings) != expected_count:
            message = "Ollama embedding response count does not match input count"
            raise EmbeddingResponseError(message)

        return [self._parse_vector(vector) for vector in embeddings]

    def _parse_vector(self, vector: Any) -> EmbeddingVector:
        if not isinstance(vector, list):
            raise EmbeddingResponseError("Ollama embedding vector must be a list")

        parsed_vector: list[float] = []
        for value in vector:
            if not isinstance(value, int | float):
                raise EmbeddingResponseError("Ollama embedding vector contains a non-numeric value")
            parsed_vector.append(float(value))

        if self.dimensions is not None and len(parsed_vector) != self.dimensions:
            message = "Ollama embedding vector dimension does not match configuration"
            raise EmbeddingResponseError(message)

        return tuple(parsed_vector)
