import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { readJsonRequestBody } from "../../../test/fetchBody";
import { renderWithQueryClient } from "../../../test/renderWithQueryClient";
import { scopesQueryKey } from "../api/useScopes";
import { CreateScopeForm } from "./CreateScopeForm";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CreateScopeForm", () => {
  it("posts a new scope and invalidates the scope list", async () => {
    const createdScope = {
      id: "scope-new",
      name: "Research Notes",
      system_prompt: "Keep the notes concise.",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    };
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(createdScope), {
        status: 201,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const handleCreated = vi.fn();
    const { queryClient } = renderWithQueryClient(
      <CreateScopeForm onCreated={handleCreated} />,
    );
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    fireEvent.change(screen.getByLabelText("New scope"), {
      target: { value: "Research Notes" },
    });
    fireEvent.change(screen.getByLabelText("System prompt"), {
      target: { value: "Keep the notes concise." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create scope" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    const firstCall = fetchMock.mock.calls[0];

    expect(firstCall[0]).toEqual(new URL("http://127.0.0.1:8000/scopes"));
    expect(firstCall[1]?.method).toBe("POST");
    expect(readJsonRequestBody(firstCall[1])).toEqual({
      name: "Research Notes",
      system_prompt: "Keep the notes concise.",
    });

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: scopesQueryKey });
      expect(handleCreated).toHaveBeenCalledWith(createdScope);
    });
  });
});
