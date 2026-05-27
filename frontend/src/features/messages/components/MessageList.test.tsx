import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithQueryClient } from "../../../test/renderWithQueryClient";
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
    renderWithQueryClient(
      <MessageList
        isError={false}
        isLoading={false}
        messages={unsortedMessages}
      />,
    );

    const list = screen.getByLabelText("Messages");
    const renderedMessages = within(list).getAllByRole("article");

    expect(
      renderedMessages.map((message) => within(message).getByText(/persisted message/).textContent),
    ).toEqual(["First persisted message.", "Second persisted message."]);
  });

  it("renders the empty message state", () => {
    renderWithQueryClient(
      <MessageList
        isError={false}
        isLoading={false}
        messages={[]}
      />,
    );

    expect(screen.getByText(/No messages yet/)).toBeInTheDocument();
  });
});
