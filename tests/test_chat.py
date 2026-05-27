from __future__ import annotations

import asyncio
import json
import logging
from types import TracebackType

import httpx
import pytest

from smriti.chat import (
    ChatConfigurationError,
    ChatConnectionError,
    ChatError,
    ChatGenerator,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseError,
    ChatStreamFinal,
    ChatStreamToken,
    ChatTimeoutError,
    ChatUsage,
    FakeChatGenerator,
    FakeStreamingChatGenerator,
    OllamaChatGenerator,
    StreamingChatGenerator,
)


@pytest.mark.asyncio
async def test_fake_chat_generator_is_deterministic_and_matches_protocol() -> None:
    generator: ChatGenerator = FakeChatGenerator()
    request = ChatRequest(messages=(ChatMessage(role="user", content="hello"),))

    first = await generator.generate(request)
    second = await generator.generate(request)

    assert first == second
    assert first.content == "fake assistant response"
    assert first.model == "fake-chat-generator"
    assert first.finish_reason == "stop"


@pytest.mark.asyncio
async def test_fake_chat_generator_records_requests_and_consumes_programmed_responses() -> None:
    first_response = ChatResponse(content="first", model="fake-one", finish_reason="stop")
    second_response = ChatResponse(content="second", model="fake-two", finish_reason="length")
    generator = FakeChatGenerator(responses=[first_response, second_response])
    first_request = ChatRequest(messages=(ChatMessage(role="user", content="one"),))
    second_request = ChatRequest(messages=(ChatMessage(role="user", content="two"),))

    assert await generator.generate(first_request) == first_response
    assert await generator.generate(second_request) == second_response
    assert await generator.generate(first_request) == generator.response
    assert generator.requests == [first_request, second_request, first_request]


@pytest.mark.asyncio
async def test_fake_chat_generator_raises_configured_error() -> None:
    generator = FakeChatGenerator(error=ChatTimeoutError("local chat timed out"))

    with pytest.raises(ChatTimeoutError):
        await generator.generate(ChatRequest(messages=(ChatMessage(role="user", content="hi"),)))


@pytest.mark.asyncio
async def test_fake_streaming_chat_generator_yields_tokens_and_default_final() -> None:
    generator: StreamingChatGenerator = FakeStreamingChatGenerator(tokens=["hello", " world"])
    request = ChatRequest(messages=(ChatMessage(role="user", content="hi"),))

    events = [event async for event in generator.generate_stream(request)]

    assert events == [
        ChatStreamToken(text="hello"),
        ChatStreamToken(text=" world"),
        ChatStreamFinal(model="fake-streaming-chat-generator", finish_reason="stop"),
    ]


@pytest.mark.asyncio
async def test_fake_streaming_chat_generator_raises_mid_stream_error() -> None:
    generator = FakeStreamingChatGenerator(
        tokens=["before", "after"],
        error=ChatResponseError("stream failed"),
        fail_after_tokens=1,
    )
    stream = generator.generate_stream(
        ChatRequest(messages=(ChatMessage(role="user", content="hi"),))
    )

    assert await anext(stream) == ChatStreamToken(text="before")
    with pytest.raises(ChatResponseError):
        await anext(stream)


def test_chat_errors_mirror_embedding_error_names() -> None:
    assert ChatConfigurationError.__mro__[1] is ChatError
    assert ChatConnectionError.__mro__[1] is ChatError
    assert ChatTimeoutError.__mro__[1] is ChatError
    assert ChatResponseError.__mro__[1] is ChatError


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.com:11434",
        "https://127.0.0.1:11434",
        "http://user:pass@127.0.0.1:11434",
        "http://127.0.0.1:11434?debug=true",
        "http://127.0.0.1:11434#fragment",
    ],
)
def test_ollama_chat_generator_rejects_unsupported_base_urls(base_url: str) -> None:
    with pytest.raises(ChatConfigurationError):
        OllamaChatGenerator(base_url=base_url)


def test_ollama_chat_generator_rejects_invalid_model_timeout_and_context_window() -> None:
    with pytest.raises(ChatConfigurationError):
        OllamaChatGenerator(model="")

    with pytest.raises(ChatConfigurationError):
        OllamaChatGenerator(timeout_seconds=0.0)

    with pytest.raises(ChatConfigurationError):
        OllamaChatGenerator(num_ctx=0)


@pytest.mark.asyncio
async def test_ollama_chat_generator_posts_non_streaming_request_to_local_chat_endpoint() -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_head = await reader.readuntil(b"\r\n\r\n")
        headers = _parse_request_headers(request_head)
        body = await reader.readexactly(int(headers["content-length"]))
        request_line = request_head.decode("iso-8859-1").split("\r\n", maxsplit=1)[0]
        requests.append((request_line, json.loads(body.decode("utf-8"))))

        response_body = json.dumps(
            {
                "model": "qwen2.5:7b",
                "message": {"role": "assistant", "content": "hello from local ollama"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 11,
                "eval_count": 7,
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
        generator = OllamaChatGenerator(
            base_url=f"http://127.0.0.1:{port}",
            model="qwen2.5:7b",
            timeout_seconds=1.0,
            num_ctx=16384,
        )

        response = await generator.generate(
            ChatRequest(
                messages=(
                    ChatMessage(role="system", content="You are local."),
                    ChatMessage(role="user", content="Say hello."),
                )
            )
        )
    finally:
        server.close()
        await server.wait_closed()

    assert response == ChatResponse(
        content="hello from local ollama",
        model="qwen2.5:7b",
        finish_reason="stop",
        usage=ChatUsage(prompt_tokens=11, completion_tokens=7),
    )
    assert requests == [
        (
            "POST /api/chat HTTP/1.1",
            {
                "model": "qwen2.5:7b",
                "messages": [
                    {"role": "system", "content": "You are local."},
                    {"role": "user", "content": "Say hello."},
                ],
                "stream": False,
                "options": {"num_ctx": 16384},
            },
        )
    ]


@pytest.mark.asyncio
async def test_ollama_chat_generator_streams_local_chat_events() -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_head = await reader.readuntil(b"\r\n\r\n")
        headers = _parse_request_headers(request_head)
        body = await reader.readexactly(int(headers["content-length"]))
        request_line = request_head.decode("iso-8859-1").split("\r\n", maxsplit=1)[0]
        requests.append((request_line, json.loads(body.decode("utf-8"))))

        response_body = (
            json.dumps(
                {
                    "model": "qwen2.5:7b",
                    "message": {"role": "assistant", "content": "hel"},
                    "done": False,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "model": "qwen2.5:7b",
                    "message": {"role": "assistant", "content": "lo"},
                    "done": False,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "model": "qwen2.5:7b",
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 11,
                    "eval_count": 2,
                }
            )
            + "\n"
        ).encode("utf-8")
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/x-ndjson\r\n"
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
        generator = OllamaChatGenerator(
            base_url=f"http://127.0.0.1:{port}",
            model="qwen2.5:7b",
            timeout_seconds=1.0,
            num_ctx=4096,
        )

        events = [
            event
            async for event in generator.generate_stream(
                ChatRequest(messages=(ChatMessage(role="user", content="Say hello."),))
            )
        ]
    finally:
        server.close()
        await server.wait_closed()

    assert events == [
        ChatStreamToken(text="hel"),
        ChatStreamToken(text="lo"),
        ChatStreamFinal(
            model="qwen2.5:7b",
            finish_reason="stop",
            usage=ChatUsage(prompt_tokens=11, completion_tokens=2),
        ),
    ]
    assert requests == [
        (
            "POST /api/chat HTTP/1.1",
            {
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": "Say hello."}],
                "stream": True,
                "options": {"num_ctx": 4096},
            },
        )
    ]


@pytest.mark.asyncio
async def test_ollama_chat_generator_preserves_base_url_path_prefix() -> None:
    requests: list[str] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_head = await reader.readuntil(b"\r\n\r\n")
        await _read_request_body(request_head, reader)
        request_line = request_head.decode("iso-8859-1").split("\r\n", maxsplit=1)[0]
        requests.append(request_line)

        response_body = json.dumps(
            {
                "model": "qwen2.5:7b",
                "message": {"role": "assistant", "content": "ok"},
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
        generator = OllamaChatGenerator(
            base_url=f"http://127.0.0.1:{port}/ollama",
            timeout_seconds=1.0,
        )

        await generator.generate(ChatRequest(messages=(ChatMessage(role="user", content="hi"),)))
    finally:
        server.close()
        await server.wait_closed()

    assert requests == ["POST /ollama/api/chat HTTP/1.1"]


@pytest.mark.asyncio
async def test_ollama_chat_generator_raises_typed_response_errors() -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _read_request(reader)
        response_body = json.dumps({"error": "model not found"}).encode("utf-8")
        writer.write(
            b"HTTP/1.1 404 Not Found\r\n"
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
        generator = OllamaChatGenerator(base_url=f"http://127.0.0.1:{port}", timeout_seconds=1.0)

        with pytest.raises(ChatResponseError, match="model not found"):
            await generator.generate(
                ChatRequest(messages=(ChatMessage(role="user", content="hi"),))
            )
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_ollama_chat_generator_raises_typed_timeout_error() -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await asyncio.sleep(1.0)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    try:
        port = _server_port(server)
        generator = OllamaChatGenerator(base_url=f"http://127.0.0.1:{port}", timeout_seconds=0.01)

        with pytest.raises(ChatTimeoutError):
            await generator.generate(
                ChatRequest(messages=(ChatMessage(role="user", content="hi"),))
            )
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_ollama_chat_generator_maps_request_errors_to_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")

    class FailingAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs

        async def __aenter__(self) -> FailingAsyncClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            _ = (exc_type, exc, traceback)

        async def post(self, request_url: str, json: dict[str, object]) -> httpx.Response:
            _ = (request_url, json)
            raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr("smriti.chat.ollama.httpx.AsyncClient", FailingAsyncClient)
    generator = OllamaChatGenerator(timeout_seconds=1.0)

    with pytest.raises(ChatConnectionError):
        await generator.generate(ChatRequest(messages=(ChatMessage(role="user", content="hi"),)))


@pytest.mark.asyncio
async def test_ollama_chat_generator_rejects_invalid_success_response() -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _read_request(reader)
        response_body = b"not-json"
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
        generator = OllamaChatGenerator(base_url=f"http://127.0.0.1:{port}", timeout_seconds=1.0)

        with pytest.raises(ChatResponseError, match="non-JSON"):
            await generator.generate(
                ChatRequest(messages=(ChatMessage(role="user", content="hi"),))
            )
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_ollama_chat_generator_rejects_malformed_chat_response() -> None:
    generator = OllamaChatGenerator()

    with pytest.raises(ChatResponseError, match="message role"):
        generator._parse_chat_response(
            {
                "model": "qwen2.5:7b",
                "message": {"role": "user", "content": "not assistant"},
            }
        )


@pytest.mark.asyncio
async def test_ollama_chat_generator_does_not_log_prompt_or_response_content_at_info_or_below(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_sentinel = "PROMPT_SENTINEL_STAGE_7_2"
    response_sentinel = "RESPONSE_SENTINEL_STAGE_7_2"
    captured_client_kwargs: list[dict[str, object]] = []

    class MockAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            captured_client_kwargs.append(kwargs)

        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            _ = (exc_type, exc, traceback)

        async def post(self, request_url: str, json: dict[str, object]) -> httpx.Response:
            request = httpx.Request("POST", request_url)
            _ = json
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:7b",
                    "message": {"role": "assistant", "content": response_sentinel},
                    "done_reason": "stop",
                },
                request=request,
            )

    monkeypatch.setattr("smriti.chat.ollama.httpx.AsyncClient", MockAsyncClient)
    generator = OllamaChatGenerator(timeout_seconds=1.0)

    with caplog.at_level(logging.DEBUG):
        response = await generator.generate(
            ChatRequest(messages=(ChatMessage(role="user", content=prompt_sentinel),))
        )

    assert response.content == response_sentinel
    assert captured_client_kwargs == [
        {"follow_redirects": False, "timeout": 1.0, "trust_env": False}
    ]
    smriti_chat_records = [
        record
        for record in caplog.records
        if (record.name == "smriti.chat" or record.name.startswith("smriti.chat."))
        and record.levelno <= logging.INFO
    ]
    for record in smriti_chat_records:
        message = record.getMessage()
        assert prompt_sentinel not in message
        assert response_sentinel not in message


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
    await _read_request_body(request_head, reader)


async def _read_request_body(request_head: bytes, reader: asyncio.StreamReader) -> None:
    headers = _parse_request_headers(request_head)
    await reader.readexactly(int(headers["content-length"]))


def _server_port(server: asyncio.AbstractServer) -> int:
    sockets = server.sockets
    assert sockets is not None
    socket_name = sockets[0].getsockname()
    assert isinstance(socket_name, tuple)
    return int(socket_name[1])
