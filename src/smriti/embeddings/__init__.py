from __future__ import annotations

from smriti.embeddings.base import Embedder, EmbeddingVector
from smriti.embeddings.errors import (
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
)
from smriti.embeddings.fake import FakeEmbedder
from smriti.embeddings.ollama import OllamaEmbedder

__all__ = [
    "Embedder",
    "EmbeddingConfigurationError",
    "EmbeddingConnectionError",
    "EmbeddingError",
    "EmbeddingResponseError",
    "EmbeddingTimeoutError",
    "EmbeddingVector",
    "FakeEmbedder",
    "OllamaEmbedder",
]
