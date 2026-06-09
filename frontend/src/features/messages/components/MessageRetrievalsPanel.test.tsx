import { fireEvent, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { MessageRetrievalsResponse } from "../api/useMessageRetrievals";
import { renderWithQueryClient } from "../../../test/renderWithQueryClient";
import { MessageRetrievalsPanel } from "./MessageRetrievalsPanel";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("MessageRetrievalsPanel", () => {
  it("renders retrievals ordered by rank", async () => {
    stubRetrievalsResponse(
      retrievalsResponse({
        retrievals: [
          retrievalEntry({ rank: 2, content: "Second ranked memory." }),
          retrievalEntry({ rank: 1, content: "First ranked memory." }),
        ],
      }),
    );

    renderWithQueryClient(
      <MessageRetrievalsPanel
        conversationId="conversation-1"
        isOpen
        messageId="assistant-1"
      />,
    );

    await screen.findByLabelText("Retrieved memory rank 1");

    const renderedRetrievals = screen.getAllByRole("article", {
      name: /Retrieved memory rank/,
    });
    expect(
      renderedRetrievals.map((retrieval) =>
        within(retrieval).getByText(/ranked memory/i).textContent,
      ),
    ).toEqual(["First ranked memory.", "Second ranked memory."]);
  });

  it("renders the empty state", async () => {
    stubRetrievalsResponse(retrievalsResponse({ retrievals: [] }));

    renderWithQueryClient(
      <MessageRetrievalsPanel
        conversationId="conversation-1"
        isOpen
        messageId="assistant-1"
      />,
    );

    expect(await screen.findByText("No retrieved memories recorded.")).toBeInTheDocument();
  });

  it("truncates and expands long content", async () => {
    const longContent = "Long memory detail. ".repeat(30);
    const expandedContent = longContent.trim();
    stubRetrievalsResponse(
      retrievalsResponse({
        retrievals: [retrievalEntry({ content: longContent, rank: 1 })],
      }),
    );

    renderWithQueryClient(
      <MessageRetrievalsPanel
        conversationId="conversation-1"
        isOpen
        messageId="assistant-1"
      />,
    );

    await screen.findByText("Show more");

    expect(screen.queryByText(expandedContent)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show more" }));

    expect(screen.getByText(expandedContent)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show less" })).toBeInTheDocument();
  });

  it("uses the untitled conversation fallback for null source titles", async () => {
    stubRetrievalsResponse(
      retrievalsResponse({
        retrievals: [
          retrievalEntry({
            rank: 1,
            sourceConversationTitle: null,
            sourceScopeName: "Family Companion",
          }),
        ],
      }),
    );

    renderWithQueryClient(
      <MessageRetrievalsPanel
        conversationId="conversation-1"
        isOpen
        messageId="assistant-1"
      />,
    );

    expect(
      await screen.findByText("Family Companion › Untitled conversation"),
    ).toBeInTheDocument();
  });

  it("normalizes retrieval source kind labels to memory", async () => {
    stubRetrievalsResponse(
      retrievalsResponse({
        retrievals: [retrievalEntry({ kind: "summary", rank: 1 })],
      }),
    );

    renderWithQueryClient(
      <MessageRetrievalsPanel
        conversationId="conversation-1"
        isOpen
        messageId="assistant-1"
      />,
    );

    const retrieval = await screen.findByLabelText("Retrieved memory rank 1");

    expect(within(retrieval).getByText("memory")).toBeInTheDocument();
    expect(within(retrieval).queryByText("summary")).not.toBeInTheDocument();
  });
});

function stubRetrievalsResponse(response: MessageRetrievalsResponse) {
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(response), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ),
  );
}

function retrievalsResponse({
  retrievals,
}: {
  retrievals: MessageRetrievalsResponse["retrievals"];
}): MessageRetrievalsResponse {
  return {
    assistant_message_id: "assistant-1",
    total: retrievals.length,
    retrievals,
  };
}

function retrievalEntry({
  content = "Remember the appointment.",
  kind = "message",
  rank,
  sourceConversationTitle = "Source conversation",
  sourceScopeName = "Research Notes",
}: {
  content?: string;
  kind?: "message" | "summary";
  rank: number;
  sourceConversationTitle?: string | null;
  sourceScopeName?: string;
}): MessageRetrievalsResponse["retrievals"][number] {
  return {
    rank,
    similarity: 0.91,
    score: 0.82,
    recency_score: 0.73,
    access_score: 0.64,
    frequency_score: 0.55,
    importance_score: 0.46,
    scoring_version: "test-v1",
    retrieved_at: "2026-01-01T12:00:00Z",
    query: {
      message_id: "query-1",
      content: "What did I ask?",
    },
    episode: {
      id: `episode-${rank.toString()}`,
      kind,
      content,
      source_conversation_id: "conversation-source",
      source_conversation_title: sourceConversationTitle,
      source_scope_id: "scope-1",
      source_scope_name: sourceScopeName,
    },
  };
}
