import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { readJsonRequestBody } from "../../../test/fetchBody";
import { renderWithQueryClient } from "../../../test/renderWithQueryClient";
import { conversationsQueryKey } from "../api/useConversations";
import { CreateConversationForm } from "./CreateConversationForm";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CreateConversationForm", () => {
  it("posts a conversation for the selected scope", async () => {
    const createdConversation = {
      id: "conversation-new",
      scope_id: "scope-selected",
      title: "Daily notes",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    };
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(createdConversation), {
        status: 201,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const handleCreated = vi.fn();
    const { queryClient } = renderWithQueryClient(
      <CreateConversationForm
        onCreated={handleCreated}
        selectedScopeId="scope-selected"
      />,
    );
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    fireEvent.change(screen.getByLabelText("Conversation title"), {
      target: { value: "Daily notes" },
    });
    fireEvent.click(screen.getByRole("button", { name: "+ New conversation" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    const firstCall = fetchMock.mock.calls[0];

    expect(firstCall[0]).toEqual(new URL("http://127.0.0.1:8100/conversations"));
    expect(firstCall[1]?.method).toBe("POST");
    expect(readJsonRequestBody(firstCall[1])).toEqual({
      scope_id: "scope-selected",
      title: "Daily notes",
    });

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: conversationsQueryKey });
      expect(handleCreated).toHaveBeenCalledWith(createdConversation);
    });
  });
});
