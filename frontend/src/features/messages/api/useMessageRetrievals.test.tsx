import { screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderWithQueryClient } from "../../../test/renderWithQueryClient";
import {
  type MessageRetrievalsResponse,
  useMessageRetrievals,
} from "./useMessageRetrievals";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useMessageRetrievals", () => {
  it("reports loading while retrievals are being fetched", () => {
    const fetchMock = vi.fn<typeof fetch>().mockReturnValue(new Promise<Response>(() => undefined));
    vi.stubGlobal("fetch", fetchMock);

    renderWithQueryClient(
      <MessageRetrievalsHarness conversationId="conversation-1" messageId="message-1" />,
    );

    expect(screen.getByTestId("status")).toHaveTextContent("pending");
  });

  it("returns retrievals on success", async () => {
    const response = messageRetrievalsResponse({ total: 1 });
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(response), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    renderWithQueryClient(
      <MessageRetrievalsHarness conversationId="conversation/1" messageId="message 1" />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("success");
    });

    expect(screen.getByTestId("total")).toHaveTextContent("1");
    expect(fetchMock).toHaveBeenCalledWith(
      new URL(
        "http://127.0.0.1:8000/conversations/conversation%2F1/messages/message%201/retrievals",
      ),
      {
        headers: new Headers(),
        body: undefined,
      },
    );
  });

  it("reports errors", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Assistant message not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    renderWithQueryClient(
      <MessageRetrievalsHarness conversationId="conversation-1" messageId="message-1" />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("error");
    });

    expect(screen.getByTestId("error")).toHaveTextContent("error");
  });

  it("does not fetch when disabled", () => {
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", fetchMock);

    renderWithQueryClient(
      <MessageRetrievalsHarness
        conversationId="conversation-1"
        enabled={false}
        messageId="message-1"
      />,
    );

    expect(fetchMock).not.toHaveBeenCalled();
  });
});

function MessageRetrievalsHarness({
  conversationId,
  enabled = true,
  messageId,
}: {
  conversationId: string;
  enabled?: boolean;
  messageId: string;
}) {
  const query = useMessageRetrievals(conversationId, messageId, { enabled });

  return (
    <div>
      <p data-testid="status">{query.status}</p>
      <p data-testid="total">{query.data?.total ?? ""}</p>
      <p data-testid="error">{query.isError ? "error" : ""}</p>
    </div>
  );
}

function messageRetrievalsResponse({
  total,
}: {
  total: number;
}): MessageRetrievalsResponse {
  return {
    assistant_message_id: "assistant-1",
    total,
    retrievals: [
      {
        rank: 1,
        similarity: 0.91,
        score: 0.82,
        recency_score: 0.73,
        access_score: 0.64,
        frequency_score: 0.55,
        importance_score: 0.46,
        scoring_version: "test-v1",
        retrieved_at: "2026-01-01T12:00:00Z",
        query: {
          message_id: "message-1",
          content: "What did I ask?",
        },
        episode: {
          id: "episode-1",
          kind: "message",
          content: "Remember this.",
          source_conversation_id: "conversation-1",
          source_conversation_title: "Source conversation",
          source_scope_id: "scope-1",
          source_scope_name: "Research",
        },
      },
    ],
  };
}
