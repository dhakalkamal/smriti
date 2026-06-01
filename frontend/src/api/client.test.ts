import { afterEach, describe, expect, it, vi } from "vitest";

import { readJsonRequestBody } from "../test/fetchBody";
import { apiFetch, ApiClientError, getApiBaseUrl, postStream } from "./client";
import type { HealthResponse } from "../features/health/api/useHealthCheck";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("getApiBaseUrl", () => {
  it("defaults to the local FastAPI service", () => {
    expect(getApiBaseUrl("")).toBe("http://127.0.0.1:8100");
  });

  it("accepts localhost HTTP API URLs", () => {
    expect(getApiBaseUrl("http://localhost:8100")).toBe("http://localhost:8100");
  });

  it("rejects external API URLs", () => {
    expect(() => getApiBaseUrl("https://example.com")).toThrow(/local http service/);
  });

  it("rejects URL credentials", () => {
    expect(() => getApiBaseUrl("http://user:pass@127.0.0.1:8100")).toThrow(/credentials/);
  });
});

describe("apiFetch", () => {
  it("requests through the local API boundary", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ status: "ok", mode: "local" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(apiFetch<HealthResponse>("/health")).resolves.toEqual({
      status: "ok",
      mode: "local",
    });
    expect(fetchMock).toHaveBeenCalledWith(new URL("http://127.0.0.1:8100/health"), {
      headers: new Headers(),
      body: undefined,
    });
  });

  it("throws typed errors for failed responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockResolvedValue(
        new Response("nope", { status: 503 }),
      ),
    );

    await expect(apiFetch<HealthResponse>("/health")).rejects.toBeInstanceOf(ApiClientError);
  });
});

describe("postStream", () => {
  it("posts JSON through the local API boundary and returns the raw response", async () => {
    const response = new Response("event stream", {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(response);
    const controller = new AbortController();
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      postStream(
        "/conversations/conversation-1/assistant-response/stream",
        { scope_id: "scope-1", query_message_id: "message-1" },
        controller.signal,
      ),
    ).resolves.toBe(response);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toEqual(
      new URL(
        "http://127.0.0.1:8100/conversations/conversation-1/assistant-response/stream",
      ),
    );
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBe("POST");
    expect(fetchMock.mock.calls[0]?.[1]?.signal).toBe(controller.signal);
    expect(readJsonRequestBody(fetchMock.mock.calls[0]?.[1])).toEqual({
      scope_id: "scope-1",
      query_message_id: "message-1",
    });

    const headers = fetchMock.mock.calls[0]?.[1]?.headers;
    expect(headers).toBeInstanceOf(Headers);
    expect((headers as Headers).get("Accept")).toBe("text/event-stream");
  });
});
