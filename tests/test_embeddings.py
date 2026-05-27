from __future__ import annotations

import asyncio
import json
import math

import pytest

from smriti.embeddings import (
    Embedder,
    EmbeddingConfigurationError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
    FakeEmbedder,
    OllamaEmbedder,
)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


@pytest.mark.asyncio
async def test_fake_embedder_is_deterministic_and_matches_protocol() -> None:
    embedder: Embedder = FakeEmbedder(dimensions=16)

    first = await embedder.embed_text("family research notes")
    second = await embedder.embed_text("family research notes")
    batch = await embedder.embed_texts(["family research notes", "coding helper"])

    assert first == second
    assert batch[0] == first
    assert len(first) == 16
    assert isinstance(first, tuple)


@pytest.mark.asyncio
async def test_fake_embedder_uses_stable_token_overlap_for_ranking_tests() -> None:
    embedder = FakeEmbedder(dimensions=128)

    query = await embedder.embed_text("family research")
    related = await embedder.embed_text("family notes")
    unrelated = await embedder.embed_text("database migration")

    assert math.isclose(_dot(query, query), 1.0)
    assert _dot(query, related) > _dot(query, unrelated)


def test_fake_embedder_rejects_invalid_dimensions() -> None:
    with pytest.raises(EmbeddingConfigurationError):
        FakeEmbedder(dimensions=0)


def test_ollama_embedder_rejects_non_localhost_base_url() -> None:
    with pytest.raises(EmbeddingConfigurationError):
        OllamaEmbedder(base_url="http://example.com:11434")

    with pytest.raises(EmbeddingConfigurationError):
        OllamaEmbedder(base_url="https://127.0.0.1:11434")


def test_ollama_embedder_rejects_invalid_context_window() -> None:
    with pytest.raises(EmbeddingConfigurationError):
        OllamaEmbedder(num_ctx=0)


@pytest.mark.asyncio
async def test_ollama_embedder_posts_batch_to_local_embed_endpoint() -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_head = await reader.readuntil(b"\r\n\r\n")
        headers = _parse_request_headers(request_head)
        body = await reader.readexactly(int(headers["content-length"]))
        request_line = request_head.decode("iso-8859-1").split("\r\n", maxsplit=1)[0]
        requests.append((request_line, json.loads(body.decode("utf-8"))))

        response_body = json.dumps(
            {
                "model": "nomic-embed-text",
                "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            }
        ).encode("utf-8")
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(response_body)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
            + response_body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    try:
        port = _server_port(server)
        embedder = OllamaEmbedder(
            base_url=f"http://127.0.0.1:{port}",
            model="nomic-embed-text",
            dimensions=2,
            timeout_seconds=1.0,
            num_ctx=6144,
        )

        vectors = await embedder.embed_texts(["one", "two"])
    finally:
        server.close()
        await server.wait_closed()

    assert vectors == [(1.0, 0.0), (0.0, 1.0)]
    assert requests == [
        (
            "POST /api/embed HTTP/1.1",
            {
                "model": "nomic-embed-text",
                "input": ["one", "two"],
                "options": {"num_ctx": 6144},
                "dimensions": 2,
            },
        )
    ]


@pytest.mark.asyncio
async def test_ollama_embedder_raises_typed_response_errors() -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _read_request(reader)
        response_body = json.dumps({"error": "model does not support embeddings"}).encode("utf-8")
        writer.write(
            b"HTTP/1.1 500 Internal Server Error\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(response_body)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
            + response_body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    try:
        port = _server_port(server)
        embedder = OllamaEmbedder(base_url=f"http://127.0.0.1:{port}", timeout_seconds=1.0)

        with pytest.raises(EmbeddingResponseError, match="model does not support embeddings"):
            await embedder.embed_text("hello")
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_ollama_embedder_raises_typed_timeout_error() -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await asyncio.sleep(1.0)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    try:
        port = _server_port(server)
        embedder = OllamaEmbedder(base_url=f"http://127.0.0.1:{port}", timeout_seconds=0.01)

        with pytest.raises(EmbeddingTimeoutError):
            await embedder.embed_text("hello")
    finally:
        server.close()
        await server.wait_closed()


def _parse_request_headers(request_head: bytes) -> dict[str, str]:
    lines = request_head.decode("iso-8859-1").split("\r\n")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, value = line.split(":", maxsplit=1)
        headers[name.lower()] = value.strip()
    return headers


async def _read_request(reader: asyncio.StreamReader) -> None:
    request_head = await reader.readuntil(b"\r\n\r\n")
    headers = _parse_request_headers(request_head)
    await reader.readexactly(int(headers["content-length"]))


def _server_port(server: asyncio.AbstractServer) -> int:
    sockets = server.sockets
    assert sockets is not None
    socket_name = sockets[0].getsockname()
    assert isinstance(socket_name, tuple)
    return int(socket_name[1])
