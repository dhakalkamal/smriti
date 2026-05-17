import { fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { readJsonRequestBody } from "../../../test/fetchBody";
import { renderWithQueryClient } from "../../../test/renderWithQueryClient";
import { messagesQueryKey, type MessageResponse } from "../../messages/api/useMessages";
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

const assistantGeneration = {};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ChatView", () => {
  it("posts the user message before creating the assistant response", async () => {
    const fetchMock = mockFetchResponses([
      jsonResponse([], 200),
      jsonResponse(createdUserMessage, 201),
      jsonResponse(assistantGeneration, 201),
      jsonResponse([userMessage, assistantMessage], 200),
    ]);
    vi.stubGlobal("fetch", fetchMock);
    const { queryClient } = renderWithQueryClient(
      <ChatView
        conversationId="conversation-1"
        scopeId="scope-1"
        title="Daily notes"
      />,
    );
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    expect(await screen.findByText(/No messages yet/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Hello memory" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(screen.getByText("Assistant is thinking...")).toBeInTheDocument();
    expect(await screen.findByText("Persisted assistant reply.")).toBeInTheDocument();

    const postCalls = fetchMock.mock.calls.filter((call) => call[1]?.method === "POST");
    expect(postCalls).toHaveLength(2);
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
        "http://127.0.0.1:8000/conversations/conversation-1/assistant-response",
      ),
    );
    expect(readJsonRequestBody(postCalls[1]?.[1])).toEqual({
      scope_id: "scope-1",
      query_message_id: "message-user-1",
    });
    expect(requestUrl(postCalls[1][0]).pathname).not.toContain("/stream");
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: messagesQueryKey("conversation-1"),
    });
  });

  it("does not create an assistant response when saving the user message fails", async () => {
    const fetchMock = mockFetchResponses([
      jsonResponse([], 200),
      new Response("nope", { status: 500 }),
    ]);
    vi.stubGlobal("fetch", fetchMock);
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

    const assistantCalls = fetchMock.mock.calls.filter((call) =>
      requestUrl(call[0]).pathname.includes("/assistant-response"),
    );
    expect(assistantCalls).toHaveLength(0);
  });

  it("retries assistant generation with the original query message id", async () => {
    const fetchMock = mockFetchResponses([
      jsonResponse([], 200),
      jsonResponse(createdUserMessage, 201),
      new Response("nope", { status: 500 }),
      jsonResponse([userMessage], 200),
      jsonResponse(assistantGeneration, 201),
      jsonResponse([userMessage, assistantMessage], 200),
    ]);
    vi.stubGlobal("fetch", fetchMock);
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

    expect(
      await screen.findByText("Assistant response could not be created."),
    ).toBeInTheDocument();
    expect(screen.getByText("Hello memory")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry assistant response" }));

    expect(await screen.findByText("Persisted assistant reply.")).toBeInTheDocument();

    const messageCreateCalls = fetchMock.mock.calls.filter(
      (call) => requestUrl(call[0]).pathname.endsWith("/messages") && call[1]?.method === "POST",
    );
    const assistantCalls = fetchMock.mock.calls.filter(
      (call) =>
        requestUrl(call[0]).pathname.endsWith("/assistant-response") &&
        call[1]?.method === "POST",
    );

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
});

function jsonResponse(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function mockFetchResponses(responses: Response[]) {
  return vi.fn<typeof fetch>().mockImplementation(() => {
    const response = responses.shift();

    if (response === undefined) {
      throw new Error("Unexpected fetch request.");
    }

    return Promise.resolve(response);
  });
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
