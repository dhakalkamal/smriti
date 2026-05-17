import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { createQueryClient } from "./lib/queryClient";

function renderApp() {
  return render(
    <QueryClientProvider client={createQueryClient()}>
      <App />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the chat shell", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockImplementation((input) => {
        const url = requestUrl(input);

        if (url.pathname === "/scopes" || url.pathname === "/conversations") {
          return Promise.resolve(
            new Response(JSON.stringify([]), {
              status: 200,
              headers: { "Content-Type": "application/json" },
            }),
          );
        }

        throw new Error(`Unexpected request to ${url.pathname}.`);
      }),
    );

    renderApp();

    expect(screen.getByRole("heading", { name: "Chat" })).toBeInTheDocument();
    expect(await screen.findByText("Create your first scope")).toBeInTheDocument();
  });
});

function requestUrl(input: RequestInfo | URL): URL {
  if (input instanceof URL) {
    return input;
  }

  if (typeof input === "string") {
    return new URL(input);
  }

  return new URL(input.url);
}
