from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from smriti.embeddings.base import EmbeddingVector
from smriti.embeddings.errors import EmbeddingConfigurationError

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class FakeEmbedder:
    """Deterministic, local-only embedder for tests and eval fixtures."""

    dimensions: int = 768
    seed: str = "smriti-fake-embedder-v1"

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise EmbeddingConfigurationError("FakeEmbedder dimensions must be positive")

    async def embed_text(self, text: str) -> EmbeddingVector:
        """Embed a single text value without network access."""

        return self._embed(text)

    async def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed text values deterministically, preserving input order."""

        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> EmbeddingVector:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN_PATTERN.findall(text.lower())

        for token in tokens:
            digest = hashlib.sha256(f"{self.seed}\0{token}".encode()).digest()
            index = int.from_bytes(digest[:8], byteorder="big") % self.dimensions
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return tuple(vector)

        return tuple(value / norm for value in vector)
