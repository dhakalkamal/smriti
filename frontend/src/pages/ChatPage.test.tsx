import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { readJsonRequestBody } from "../test/fetchBody";
import { renderWithQueryClient } from "../test/renderWithQueryClient";
import { createSseResponseFromChunks, sseFrame } from "../test/sseStream";
import ChatPage from "./ChatPage";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ChatPage", () => {
  it("renders the no-scopes empty state", async () => {
    vi.stubGlobal(
      "fetch",
      mockListFetch({
        scopes: [],
        conversations: [],
      }),
    );

    renderWithQueryClient(<ChatPage />);

    expect(await screen.findByText("Create your first scope")).toBeInTheDocument();
    expect(screen.getByText("Create the first scope to begin.")).toBeInTheDocument();
  });

  it("renders the no-conversations empty state for a selected scope", async () => {
    vi.stubGlobal(
      "fetch",
      mockListFetch({
        scopes: [
          {
            id: "scope-1",
            name: "Research Notes",
            system_prompt: "",
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
          },
        ],
        conversations: [],
      }),
    );

    renderWithQueryClient(<ChatPage />);

    expect(await screen.findByRole("option", { name: "Research Notes" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Scope"), {
      target: { value: "scope-1" },
    });

    expect(await screen.findByText("Create a conversation")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "+ New conversation" })).toBeInTheDocument();
  });

  it("clears assistant retry state when switching conversations", async () => {
    const fetchMock = mockConversationSwitchFetch();
    vi.stubGlobal("fetch", fetchMock);

    renderWithQueryClient(<ChatPage />);

    expect(await screen.findByRole("option", { name: "Research Notes" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Scope"), {
      target: { value: "scope-1" },
    });
    fireEvent.click(await screen.findByRole("button", { name: "Conversation A" }));

    expect(await screen.findByText(/No messages yet/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Hello from A" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Assistant failed to respond.")).toBeInTheDocument();
    expect(await screen.findByText("Hello from A")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Conversation B" }));

    expect(await screen.findByRole("heading", { name: "Conversation B" })).toBeInTheDocument();
    expect(await screen.findByText(/No messages yet/)).toBeInTheDocument();

    await waitFor(() => {
      expect(
        screen.queryByText("Assistant failed to respond."),
      ).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
    });

    const invalidAssistantRetry = fetchMock.mock.calls.some((call) => {
      const url = requestUrl(call[0]);

      return (
        call[1]?.method === "POST" &&
        url.pathname === "/conversations/conversation-b/assistant-response/stream" &&
        isQueryMessageBodyFromConversationA(readJsonRequestBody(call[1]))
      );
    });

    expect(invalidAssistantRetry).toBe(false);
  });
});

interface MockListFetchOptions {
  conversations: unknown[];
  scopes: unknown[];
}

function mockListFetch({ conversations, scopes }: MockListFetchOptions) {
  return vi.fn<typeof fetch>().mockImplementation((input) => {
    const url = requestUrl(input);

    if (url.pathname === "/scopes") {
      return Promise.resolve(jsonResponse(scopes));
    }

    if (url.pathname === "/conversations") {
      return Promise.resolve(jsonResponse(conversations));
    }

    throw new Error(`Unexpected request to ${url.pathname}.`);
  });
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function jsonResponseWithStatus(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function mockConversationSwitchFetch() {
  let conversationAMessageFetchCount = 0;

  return vi.fn<typeof fetch>().mockImplementation((input, init) => {
    const url = requestUrl(input);
    const method = init?.method ?? "GET";

    if (method === "GET" && url.pathname === "/scopes") {
      return Promise.resolve(
        jsonResponse([
          {
            id: "scope-1",
            name: "Research Notes",
            system_prompt: "",
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
          },
        ]),
      );
    }

    if (method === "GET" && url.pathname === "/conversations") {
      return Promise.resolve(
        jsonResponse([
          {
            id: "conversation-a",
            scope_id: "scope-1",
            title: "Conversation A",
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
          },
          {
            id: "conversation-b",
            scope_id: "scope-1",
            title: "Conversation B",
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
          },
        ]),
      );
    }

    if (method === "GET" && url.pathname === "/conversations/conversation-a/messages") {
      const messages =
        conversationAMessageFetchCount === 0
          ? []
          : [
              {
                id: "message-a",
                conversation_id: "conversation-a",
                role: "user",
                content: "Hello from A",
                token_count: 3,
                position: 1,
                created_at: "2026-01-01T00:00:01Z",
              },
            ];

      conversationAMessageFetchCount += 1;

      return Promise.resolve(jsonResponse(messages));
    }

    if (method === "GET" && url.pathname === "/conversations/conversation-b/messages") {
      return Promise.resolve(jsonResponse([]));
    }

    if (method === "POST" && url.pathname === "/conversations/conversation-a/messages") {
      return Promise.resolve(
        jsonResponseWithStatus(
          {
            id: "message-a",
            conversation_id: "conversation-a",
            role: "user",
            content: "Hello from A",
            token_count: 3,
            position: 1,
            created_at: "2026-01-01T00:00:01Z",
            episode_id: "episode-a",
          },
          201,
        ),
      );
    }

    if (
      method === "POST" &&
      url.pathname === "/conversations/conversation-a/assistant-response/stream"
    ) {
      return Promise.resolve(
        createSseResponseFromChunks([
          sseFrame("start", { used_memory_episode_ids: [], chat_model: null }),
          sseFrame("error", { code: "backend_error", message: "hidden" }),
        ]),
      );
    }

    if (
      method === "POST" &&
      url.pathname === "/conversations/conversation-b/assistant-response/stream"
    ) {
      return Promise.resolve(
        createSseResponseFromChunks([
          sseFrame("start", { used_memory_episode_ids: [], chat_model: null }),
          sseFrame("error", { code: "backend_error", message: "hidden" }),
        ]),
      );
    }

    throw new Error(`Unexpected ${method} request to ${url.pathname}.`);
  });
}

function isQueryMessageBodyFromConversationA(body: unknown): boolean {
  if (typeof body !== "object" || body === null) {
    return false;
  }

  const requestBody = body as Record<string, unknown>;

  return (
    "query_message_id" in requestBody && requestBody.query_message_id === "message-a"
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
