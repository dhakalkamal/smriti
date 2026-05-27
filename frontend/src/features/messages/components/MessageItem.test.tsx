import { fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderWithQueryClient } from "../../../test/renderWithQueryClient";
import type { MessageResponse } from "../api/useMessages";
import { MessageItem } from "./MessageItem";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("MessageItem", () => {
  it("shows the retrieval affordance for assistant messages", () => {
    renderWithQueryClient(<MessageItem message={messageWithRole("assistant")} />);

    expect(
      screen.getByRole("button", { name: "View retrieved memories" }),
    ).toBeInTheDocument();
  });

  it("does not show the retrieval affordance for user messages", () => {
    renderWithQueryClient(<MessageItem message={messageWithRole("user")} />);

    expect(
      screen.queryByRole("button", { name: "View retrieved memories" }),
    ).not.toBeInTheDocument();
  });

  it("does not show the retrieval affordance for system messages", () => {
    renderWithQueryClient(<MessageItem message={messageWithRole("system")} />);

    expect(
      screen.queryByRole("button", { name: "View retrieved memories" }),
    ).not.toBeInTheDocument();
  });

  it("expands the retrieval panel when clicked", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockResolvedValue(
        new Response(
          JSON.stringify({
            assistant_message_id: "message-1",
            total: 0,
            retrievals: [],
          }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          },
        ),
      ),
    );
    renderWithQueryClient(<MessageItem message={messageWithRole("assistant")} />);

    fireEvent.click(screen.getByRole("button", { name: "View retrieved memories" }));

    expect(
      screen.getByRole("button", { name: "Hide retrieved memories" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("No retrieved memories recorded.")).toBeInTheDocument();
  });
});

function messageWithRole(role: MessageResponse["role"]): MessageResponse {
  return {
    id: "message-1",
    conversation_id: "conversation-1",
    position: 1,
    role,
    content: `${role} content`,
    token_count: 2,
    created_at: "2026-01-01T12:00:00Z",
  };
}
