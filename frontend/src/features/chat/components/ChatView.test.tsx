import { useState } from "react";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi, type Mock } from "vitest";

import { readJsonRequestBody } from "../../../test/fetchBody";
import { renderWithQueryClient } from "../../../test/renderWithQueryClient";
import {
  createSseResponseFromChunks,
  createControlledSseStream,
  sseFrame,
  type ControlledSseStream,
} from "../../../test/sseStream";
import type { MessageResponse } from "../../messages/api/useMessages";
import { ChatView } from "./ChatView";

const userMessage = {
  id: "message-user-1",
  conversation_id: "conversation-1",
  role: "user",
  content: "Hello memory",
  token_count: 3,
  position: 1,
  created_at: "2026-01-01T00:00:01Z",
} satisfies MessageResponse;

const assistantMessage = {
  id: "message-assistant-1",
  conversation_id: "conversation-1",
  role: "assistant",
  content: "Persisted assistant reply.",
  token_count: 6,
  position: 2,
  created_at: "2026-01-01T00:00:02Z",
} satisfies MessageResponse;

const createdUserMessage = {
  ...userMessage,
  episode_id: "episode-user-1",
};

const conversationAUserMessage = {
  ...userMessage,
  id: "message-a-user",
  conversation_id: "conversation-a",
  content: "Conversation A message",
} satisfies MessageResponse;

const conversationBUserMessage = {
  ...userMessage,
  id: "message-b-user",
  conversation_id: "conversation-b",
  content: "Conversation B message",
} satisfies MessageResponse;

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("ChatView", () => {
  it("persists the user message, refetches it, then streams the assistant draft", async () => {
    const stream = createControlledSseStream();
    const messageListResponses: unknown[] = [[], [userMessage], [userMessage, assistantMessage]];
    let userVisibleWhenStreamStarted = false;
    const fetchMock = mockFetchImplementation((input, init) => {
      const url = requestUrl(input);

      if (url.pathname.endsWith("/messages") && init?.method !== "POST") {
        return nextJsonResponse(messageListResponses);
      }

      if (url.pathname.endsWith("/messages") && init?.method === "POST") {
        return jsonResponse(createdUserMessage, 201);
      }

      if (url.pathname.endsWith("/assistant-response/stream")) {
        userVisibleWhenStreamStarted = screen.queryByText("Hello memory") !== null;
        return stream.response;
      }

      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    const consoleSpies = spyOnConsole();

    renderWithQueryClient(
      <ChatView
        conversationId="conversation-1"
        scopeId="scope-1"
        title="Daily notes"
      />,
    );

    expect(await screen.findByText(/No messages yet/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Hello memory" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Hello memory")).toBeInTheDocument();

    await waitFor(() => {
      expect(streamPostCalls(fetchMock)).toHaveLength(1);
    });
    expect(userVisibleWhenStreamStarted).toBe(true);
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
    expect(screen.getByLabelText("Message")).toHaveValue("");
    expect(screen.getByLabelText("Message")).toBeDisabled();

    act(() => {
      stream.enqueue(sseFrame("start", { used_memory_episode_ids: [], chat_model: null }));
      stream.enqueue(sseFrame("token", { text: "Transient " }));
      stream.enqueue(sseFrame("token", { text: "draft" }));
    });

    expect(await screen.findByText(/Transient draft/)).toBeInTheDocument();

    act(() => {
      stream.enqueue(
        sseFrame("done", {
          assistant_message: assistantMessage,
          chat_model: "fake-chat",
          finish_reason: "stop",
          used_memory_episode_ids: [],
        }),
      );
      stream.close();
    });

    expect(await screen.findByText("Persisted assistant reply.")).toBeInTheDocument();
    expect(screen.queryByText(/Transient draft/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();

    const postCalls = fetchMock.mock.calls.filter((call) => call[1]?.method === "POST");
    expect(postCalls[0]?.[0]).toEqual(
      new URL("http://127.0.0.1:8000/conversations/conversation-1/messages"),
    );
    expect(readJsonRequestBody(postCalls[0]?.[1])).toEqual({
      role: "user",
      content: "Hello memory",
      token_count: 3,
    });
    expect(postCalls[1]?.[0]).toEqual(
      new URL(
        "http://127.0.0.1:8000/conversations/conversation-1/assistant-response/stream",
      ),
    );
    expect(readJsonRequestBody(postCalls[1]?.[1])).toEqual({
      scope_id: "scope-1",
      query_message_id: "message-user-1",
    });
    expect(consoleSpies.some((spy) => spy.mock.calls.length > 0)).toBe(false);
  });

  it("does not stream an assistant response when saving the user message fails", async () => {
    const fetchMock = mockFetchImplementation((input, init) => {
      const url = requestUrl(input);

      if (url.pathname.endsWith("/messages") && init?.method !== "POST") {
        return jsonResponse([], 200);
      }

      if (url.pathname.endsWith("/messages") && init?.method === "POST") {
        return new Response("nope", { status: 500 });
      }

      throw new Error(`Unexpected request: ${url.pathname}`);
    });

    renderWithQueryClient(
      <ChatView
        conversationId="conversation-1"
        scopeId="scope-1"
        title="Daily notes"
      />,
    );

    expect(await screen.findByText(/No messages yet/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Hello memory" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Message could not be saved.")).toBeInTheDocument();
    expect(screen.getByLabelText("Message")).toHaveValue("Hello memory");
    expect(streamPostCalls(fetchMock)).toHaveLength(0);
  });

  it("retries streaming with the original query message id", async () => {
    const streams: ControlledSseStream[] = [];
    const messageListResponses: unknown[] = [[], [userMessage], [userMessage, assistantMessage]];
    const fetchMock = mockFetchImplementation((input, init) => {
      const url = requestUrl(input);

      if (url.pathname.endsWith("/messages") && init?.method !== "POST") {
        return nextJsonResponse(messageListResponses);
      }

      if (url.pathname.endsWith("/messages") && init?.method === "POST") {
        return jsonResponse(createdUserMessage, 201);
      }

      if (url.pathname.endsWith("/assistant-response/stream")) {
        const stream = createControlledSseStream({ signal: init?.signal ?? undefined });
        streams.push(stream);
        return stream.response;
      }

      throw new Error(`Unexpected request: ${url.pathname}`);
    });

    renderWithQueryClient(
      <ChatView
        conversationId="conversation-1"
        scopeId="scope-1"
        title="Daily notes"
      />,
    );

    expect(await screen.findByText(/No messages yet/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Hello memory" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => {
      expect(streams).toHaveLength(1);
    });

    act(() => {
      streams[0]?.enqueue(sseFrame("start", { used_memory_episode_ids: [], chat_model: null }));
      streams[0]?.enqueue(sseFrame("token", { text: "failed draft" }));
      streams[0]?.enqueue(sseFrame("error", { code: "backend_error", message: "hidden" }));
    });

    expect(await screen.findByText("Assistant failed to respond.")).toBeInTheDocument();
    expect(screen.queryByText(/failed draft/)).not.toBeInTheDocument();
    expect(screen.getByText("Hello memory")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    await waitFor(() => {
      expect(streams).toHaveLength(2);
    });

    act(() => {
      streams[1]?.enqueue(sseFrame("start", { used_memory_episode_ids: [], chat_model: null }));
      streams[1]?.enqueue(
        sseFrame("done", {
          assistant_message: assistantMessage,
          chat_model: "fake-chat",
          finish_reason: "stop",
          used_memory_episode_ids: [],
        }),
      );
      streams[1]?.close();
    });

    expect(await screen.findByText("Persisted assistant reply.")).toBeInTheDocument();

    const messageCreateCalls = fetchMock.mock.calls.filter(
      (call) => requestUrl(call[0]).pathname.endsWith("/messages") && call[1]?.method === "POST",
    );
    const assistantCalls = streamPostCalls(fetchMock);

    expect(messageCreateCalls).toHaveLength(1);
    expect(assistantCalls).toHaveLength(2);
    expect(assistantCalls.map((call) => readJsonRequestBody(call[1]))).toEqual([
      {
        scope_id: "scope-1",
        query_message_id: "message-user-1",
      },
      {
        scope_id: "scope-1",
        query_message_id: "message-user-1",
      },
    ]);
  });

  it.each([
    {
      name: "non-2xx stream response",
      response: () => new Response("nope", { status: 500 }),
    },
    {
      name: "connection close without done",
      response: () =>
        createSseResponseFromChunks([
          sseFrame("start", { used_memory_episode_ids: [], chat_model: null }),
          sseFrame("token", { text: "discarded draft" }),
        ]),
    },
    {
      name: "malformed frame",
      response: () =>
        createSseResponseFromChunks([
          sseFrame("start", { used_memory_episode_ids: [], chat_model: null }),
          "event: token\nnot-data\n\n",
        ]),
    },
    {
      name: "token before start",
      response: () => createSseResponseFromChunks([sseFrame("token", { text: "too early" })]),
    },
    {
      name: "second start",
      response: () =>
        createSseResponseFromChunks([
          sseFrame("start", { used_memory_episode_ids: [], chat_model: null }),
          sseFrame("start", { used_memory_episode_ids: [], chat_model: null }),
        ]),
    },
    {
      name: "malformed JSON",
      response: () =>
        createSseResponseFromChunks([
          sseFrame("start", { used_memory_episode_ids: [], chat_model: null }),
          "event: token\ndata: nope\n\n",
        ]),
    },
  ])("shows retry UI for $name", async ({ response }) => {
    const messageListResponses: unknown[] = [[], [userMessage]];
    mockFetchImplementation((input, init) => {
      const url = requestUrl(input);

      if (url.pathname.endsWith("/messages") && init?.method !== "POST") {
        return nextJsonResponse(messageListResponses);
      }

      if (url.pathname.endsWith("/messages") && init?.method === "POST") {
        return jsonResponse(createdUserMessage, 201);
      }

      if (url.pathname.endsWith("/assistant-response/stream")) {
        return response();
      }

      throw new Error(`Unexpected request: ${url.pathname}`);
    });

    renderWithQueryClient(
      <ChatView
        conversationId="conversation-1"
        scopeId="scope-1"
        title="Daily notes"
      />,
    );

    expect(await screen.findByText(/No messages yet/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Hello memory" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Assistant failed to respond.")).toBeInTheDocument();
    expect(screen.queryByText(/discarded draft|too early/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    expect(screen.getByText("Hello memory")).toBeInTheDocument();
  });

  it("stops an active stream without showing retry UI", async () => {
    const streams: ControlledSseStream[] = [];
    const messageListResponses: unknown[] = [[], [userMessage]];
    const fetchMock = mockFetchImplementation((input, init) => {
      const url = requestUrl(input);

      if (url.pathname.endsWith("/messages") && init?.method !== "POST") {
        return nextJsonResponse(messageListResponses);
      }

      if (url.pathname.endsWith("/messages") && init?.method === "POST") {
        return jsonResponse(createdUserMessage, 201);
      }

      if (url.pathname.endsWith("/assistant-response/stream")) {
        const stream = createControlledSseStream({ signal: init?.signal ?? undefined });
        streams.push(stream);
        return stream.response;
      }

      throw new Error(`Unexpected request: ${url.pathname}`);
    });

    renderWithQueryClient(
      <ChatView
        conversationId="conversation-1"
        scopeId="scope-1"
        title="Daily notes"
      />,
    );

    expect(await screen.findByText(/No messages yet/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Hello memory" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => {
      expect(streams).toHaveLength(1);
    });

    act(() => {
      streams[0]?.enqueue(sseFrame("start", { used_memory_episode_ids: [], chat_model: null }));
      streams[0]?.enqueue(sseFrame("token", { text: "partial assistant draft" }));
    });

    expect(await screen.findByText(/partial assistant draft/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Stop" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument();
    });
    expect(screen.queryByText(/partial assistant draft/)).not.toBeInTheDocument();
    expect(screen.queryByText("Assistant failed to respond.")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
    expect(streamPostCalls(fetchMock)[0]?.[1]?.signal?.aborted).toBe(true);
  });

  it("aborts a switched-away conversation without leaking its draft into the new conversation", async () => {
    const streamsByConversation = new Map<string, ControlledSseStream>();
    const conversationAResponses: unknown[] = [
      [],
      [conversationAUserMessage],
      [conversationAUserMessage],
    ];
    const conversationBResponses: unknown[] = [[], [conversationBUserMessage]];
    const fetchMock = mockFetchImplementation((input, init) => {
      const url = requestUrl(input);

      if (url.pathname === "/conversations/conversation-a/messages" && init?.method !== "POST") {
        return nextJsonResponse(conversationAResponses);
      }

      if (url.pathname === "/conversations/conversation-b/messages" && init?.method !== "POST") {
        return nextJsonResponse(conversationBResponses);
      }

      if (url.pathname === "/conversations/conversation-a/messages" && init?.method === "POST") {
        return jsonResponse({ ...conversationAUserMessage, episode_id: "episode-a" }, 201);
      }

      if (url.pathname === "/conversations/conversation-b/messages" && init?.method === "POST") {
        return jsonResponse({ ...conversationBUserMessage, episode_id: "episode-b" }, 201);
      }

      if (url.pathname.endsWith("/assistant-response/stream")) {
        const conversationId = url.pathname.includes("conversation-a")
          ? "conversation-a"
          : "conversation-b";
        const stream = createControlledSseStream({ signal: init?.signal ?? undefined });
        streamsByConversation.set(conversationId, stream);
        return stream.response;
      }

      throw new Error(`Unexpected request: ${url.pathname}`);
    });

    renderWithQueryClient(<SwitchableChatView />);

    expect(await screen.findByText(/No messages yet/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Conversation A message" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => {
      expect(streamsByConversation.has("conversation-a")).toBe(true);
    });

    act(() => {
      streamsByConversation
        .get("conversation-a")
        ?.enqueue(sseFrame("start", { used_memory_episode_ids: [], chat_model: null }));
      streamsByConversation
        .get("conversation-a")
        ?.enqueue(sseFrame("token", { text: "draft from conversation A" }));
    });

    expect(await screen.findByText(/draft from conversation A/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Switch to B" }));

    expect(await screen.findByText("Conversation B")).toBeInTheDocument();
    expect(screen.queryByText(/draft from conversation A/)).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Conversation B message" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => {
      expect(streamsByConversation.has("conversation-b")).toBe(true);
    });

    const conversationBStreamBodies = streamPostCalls(fetchMock)
      .filter((call) => requestUrl(call[0]).pathname.includes("conversation-b"))
      .map((call) => readJsonRequestBody(call[1]));

    expect(conversationBStreamBodies).toEqual([
      { scope_id: "scope-1", query_message_id: "message-b-user" },
    ]);
    expect(
      conversationBStreamBodies.some((body) =>
        JSON.stringify(body).includes("message-a-user"),
      ),
    ).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Switch to A" }));

    expect(await screen.findByText("Conversation A message")).toBeInTheDocument();
    expect(screen.queryByText(/draft from conversation A/)).not.toBeInTheDocument();
  });
});

function SwitchableChatView() {
  const [conversationId, setConversationId] = useState("conversation-a");
  const isConversationA = conversationId === "conversation-a";

  return (
    <div>
      <button
        onClick={() => {
          setConversationId("conversation-a");
        }}
        type="button"
      >
        Switch to A
      </button>
      <button
        onClick={() => {
          setConversationId("conversation-b");
        }}
        type="button"
      >
        Switch to B
      </button>
      <ChatView
        conversationId={conversationId}
        key={conversationId}
        scopeId="scope-1"
        title={isConversationA ? "Conversation A" : "Conversation B"}
      />
    </div>
  );
}

function jsonResponse(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function nextJsonResponse(responseBodies: unknown[]): Response {
  if (responseBodies.length === 0) {
    throw new Error("Unexpected message-list request.");
  }

  return jsonResponse(responseBodies.shift(), 200);
}

function mockFetchImplementation(
  handler: (input: RequestInfo | URL, init: RequestInit | undefined) => Response,
): Mock<typeof fetch> {
  const fetchMock = vi.fn<typeof fetch>().mockImplementation((input, init) =>
    Promise.resolve(handler(input, init)),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function streamPostCalls(fetchMock: Mock<typeof fetch>) {
  return fetchMock.mock.calls.filter(
    (call) =>
      requestUrl(call[0]).pathname.endsWith("/assistant-response/stream") &&
      call[1]?.method === "POST",
  );
}

function requestUrl(input: RequestInfo | URL): URL {
  if (input instanceof URL) {
    return input;
  }

  if (typeof input === "string") {
    return new URL(input);
  }

  return new URL(input.url);
}

function spyOnConsole() {
  return [
    vi.spyOn(console, "log").mockImplementation(() => undefined),
    vi.spyOn(console, "info").mockImplementation(() => undefined),
    vi.spyOn(console, "warn").mockImplementation(() => undefined),
    vi.spyOn(console, "error").mockImplementation(() => undefined),
    vi.spyOn(console, "debug").mockImplementation(() => undefined),
  ];
}
