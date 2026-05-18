import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderWithQueryClient } from "../../../test/renderWithQueryClient";
import { messagesQueryKey } from "../../messages/api/useMessages";
import { useDeleteConversation } from "./useDeleteConversation";
import { conversationsQueryKey } from "./useConversations";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useDeleteConversation", () => {
  it("deletes a conversation and cleans up the related query cache on success", async () => {
    let resolveDelete: (response: Response) => void = () => undefined;
    const deletePromise = new Promise<Response>((resolve) => {
      resolveDelete = resolve;
    });
    const fetchMock = vi.fn<typeof fetch>().mockReturnValue(deletePromise);
    vi.stubGlobal("fetch", fetchMock);
    const { queryClient } = renderWithQueryClient(
      <DeleteConversationHarness conversationId="conversation-1" />,
    );
    queryClient.setQueryData(conversationsQueryKey, [
      {
        id: "conversation-1",
        scope_id: "scope-1",
        title: "Daily notes",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ]);
    queryClient.setQueryData(messagesQueryKey("conversation-1"), [
      {
        id: "message-1",
        conversation_id: "conversation-1",
        role: "user",
        content: "cached message",
        token_count: 2,
        position: 1,
        created_at: "2026-01-01T00:00:00Z",
      },
    ]);
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const removeSpy = vi.spyOn(queryClient, "removeQueries");

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("pending");
    });
    expect(queryClient.getQueryData(messagesQueryKey("conversation-1"))).toBeDefined();

    resolveDelete(new Response(null, { status: 204 }));

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("success");
    });

    expect(fetchMock).toHaveBeenCalledWith(
      new URL("http://127.0.0.1:8000/conversations/conversation-1"),
      {
        method: "DELETE",
        headers: new Headers(),
        body: undefined,
      },
    );
    expect(removeSpy).toHaveBeenCalledWith({
      queryKey: messagesQueryKey("conversation-1"),
    });
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: conversationsQueryKey });
    expect(queryClient.getQueryData(messagesQueryKey("conversation-1"))).toBeUndefined();
  });

  it("keeps caches intact when deletion fails", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Conversation not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const { queryClient } = renderWithQueryClient(
      <DeleteConversationHarness conversationId="conversation-1" />,
    );
    const cachedMessages = [
      {
        id: "message-1",
        conversation_id: "conversation-1",
        role: "user",
        content: "cached message",
        token_count: 2,
        position: 1,
        created_at: "2026-01-01T00:00:00Z",
      },
    ];
    queryClient.setQueryData(conversationsQueryKey, []);
    queryClient.setQueryData(messagesQueryKey("conversation-1"), cachedMessages);
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const removeSpy = vi.spyOn(queryClient, "removeQueries");

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("error");
    });

    expect(removeSpy).not.toHaveBeenCalled();
    expect(invalidateSpy).not.toHaveBeenCalled();
    expect(queryClient.getQueryData(messagesQueryKey("conversation-1"))).toEqual(
      cachedMessages,
    );
    expect(screen.getByTestId("error")).toHaveTextContent("error");
  });
});

function DeleteConversationHarness({ conversationId }: { conversationId: string }) {
  const deleteConversation = useDeleteConversation();

  return (
    <div>
      <p data-testid="status">{deleteConversation.status}</p>
      <p data-testid="error">{deleteConversation.isError ? "error" : ""}</p>
      <button
        onClick={() => {
          deleteConversation.mutate(conversationId);
        }}
        type="button"
      >
        Delete
      </button>
    </div>
  );
}
