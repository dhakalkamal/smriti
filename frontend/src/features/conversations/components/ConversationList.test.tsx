import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderWithQueryClient } from "../../../test/renderWithQueryClient";
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
  {
    id: "conversation-c",
    scope_id: "scope-a",
    title: "Older notes",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
  {
    id: "conversation-untitled",
    scope_id: "scope-a",
    title: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
] satisfies ConversationResponse[];

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ConversationList", () => {
  it("renders conversations filtered by selected scope", () => {
    renderWithQueryClient(
      <ConversationList
        conversations={conversations}
        onSelectConversation={vi.fn()}
        selectedConversationId={null}
        selectedScopeId="scope-a"
      />,
    );

    expect(screen.getByRole("button", { name: "Scope A thread" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Older notes" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Scope B thread" })).not.toBeInTheDocument();
  });

  it("shows delete affordances for inactive rows only", () => {
    renderWithQueryClient(
      <ConversationList
        conversations={conversations}
        onSelectConversation={vi.fn()}
        selectedConversationId="conversation-a"
        selectedScopeId="scope-a"
      />,
    );

    expect(
      screen.queryByRole("button", { name: "Delete Scope A thread" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete Older notes" })).toBeInTheDocument();
  });

  it("opens the confirmation modal and triggers deletion", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    const onSelectConversation = vi.fn();

    renderWithQueryClient(
      <ConversationList
        conversations={conversations}
        onSelectConversation={onSelectConversation}
        selectedConversationId="conversation-a"
        selectedScopeId="scope-a"
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete Older notes" }));

    expect(screen.getByRole("dialog", { name: "Delete conversation?" })).toHaveTextContent(
      "Older notes",
    );
    expect(screen.getByRole("button", { name: "Older notes" })).toBeInTheDocument();
    expect(onSelectConversation).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        new URL("http://127.0.0.1:8000/conversations/conversation-c"),
        {
          method: "DELETE",
          headers: new Headers(),
          body: undefined,
        },
      );
    });
  });

  it("uses the untitled fallback in the confirmation modal", () => {
    renderWithQueryClient(
      <ConversationList
        conversations={conversations}
        onSelectConversation={vi.fn()}
        selectedConversationId="conversation-a"
        selectedScopeId="scope-a"
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete Untitled conversation" }));

    expect(screen.getByRole("dialog", { name: "Delete conversation?" })).toHaveTextContent(
      "Untitled conversation",
    );
  });

  it("keeps the modal open with a generic error when deletion fails", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ detail: "raw backend detail" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    renderWithQueryClient(
      <ConversationList
        conversations={conversations}
        onSelectConversation={vi.fn()}
        selectedConversationId="conversation-a"
        selectedScopeId="scope-a"
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete Older notes" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Conversation could not be deleted.",
    );
    expect(screen.getByRole("dialog", { name: "Delete conversation?" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Older notes" })).toBeInTheDocument();
    expect(screen.queryByText("raw backend detail")).not.toBeInTheDocument();
  });

  it("disables dialog actions while deletion is pending", async () => {
    let resolveDelete: (response: Response) => void = () => undefined;
    const deletePromise = new Promise<Response>((resolve) => {
      resolveDelete = resolve;
    });
    const fetchMock = vi.fn<typeof fetch>().mockReturnValue(deletePromise);
    vi.stubGlobal("fetch", fetchMock);

    renderWithQueryClient(
      <ConversationList
        conversations={conversations}
        onSelectConversation={vi.fn()}
        selectedConversationId="conversation-a"
        selectedScopeId="scope-a"
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete Older notes" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Delete" })).toBeDisabled();
    });
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Older notes" })).toBeInTheDocument();

    resolveDelete(new Response(null, { status: 204 }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });
});
