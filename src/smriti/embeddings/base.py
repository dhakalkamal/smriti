from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

EmbeddingVector = tuple[float, ...]


class Embedder(Protocol):
    """Async interface for components that turn text into embedding vectors."""

    @property
    def dimensions(self) -> int | None:
        """Return the configured vector dimension when it is known."""

    async def embed_text(self, text: str) -> EmbeddingVector:
        """Embed a single text value."""

    async def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed one or more text values, preserving input order."""
