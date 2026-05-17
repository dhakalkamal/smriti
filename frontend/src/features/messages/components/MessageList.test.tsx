import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { MessageResponse } from "../api/useMessages";
import { MessageList } from "./MessageList";

const unsortedMessages = [
  {
    id: "message-2",
    conversation_id: "conversation-1",
    role: "assistant",
    content: "Second persisted message.",
    token_count: 5,
    position: 2,
    created_at: "2026-01-01T00:00:02Z",
  },
  {
    id: "message-1",
    conversation_id: "conversation-1",
    role: "user",
    content: "First persisted message.",
    token_count: 6,
    position: 1,
    created_at: "2026-01-01T00:00:01Z",
  },
] satisfies MessageResponse[];

describe("MessageList", () => {
  it("renders messages sorted by position", () => {
    render(
      <MessageList
        assistantError={false}
        isError={false}
        isLoading={false}
        isRetrying={false}
        messages={unsortedMessages}
        onRetryAssistant={vi.fn()}
      />,
    );

    const list = screen.getByLabelText("Messages");
    const renderedMessages = within(list).getAllByRole("article");

    expect(renderedMessages.map((message) => message.textContent)).toEqual([
      "YouFirst persisted message.",
      "AssistantSecond persisted message.",
    ]);
  });

  it("renders the empty message state", () => {
    render(
      <MessageList
        assistantError={false}
        isError={false}
        isLoading={false}
        isRetrying={false}
        messages={[]}
        onRetryAssistant={vi.fn()}
      />,
    );

    expect(screen.getByText(/No messages yet/)).toBeInTheDocument();
  });
});
