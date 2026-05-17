import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ConversationResponse } from "../api/useConversations";
import { ConversationList } from "./ConversationList";

const conversations = [
  {
    id: "conversation-a",
    scope_id: "scope-a",
    title: "Scope A thread",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
  {
    id: "conversation-b",
    scope_id: "scope-b",
    title: "Scope B thread",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
] satisfies ConversationResponse[];

describe("ConversationList", () => {
  it("renders conversations filtered by selected scope", () => {
    render(
      <ConversationList
        conversations={conversations}
        onSelectConversation={vi.fn()}
        selectedConversationId={null}
        selectedScopeId="scope-a"
      />,
    );

    expect(screen.getByRole("button", { name: "Scope A thread" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Scope B thread" })).not.toBeInTheDocument();
  });
});
