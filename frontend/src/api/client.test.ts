import { describe, expect, it, vi } from "vitest";

import { apiFetch, ApiClientError, getApiBaseUrl } from "./client";
import type { HealthResponse } from "../features/health/api/useHealthCheck";

describe("getApiBaseUrl", () => {
  it("defaults to the local FastAPI service", () => {
    expect(getApiBaseUrl("")).toBe("http://127.0.0.1:8000");
  });

  it("accepts localhost HTTP API URLs", () => {
    expect(getApiBaseUrl("http://localhost:8000")).toBe("http://localhost:8000");
  });

  it("rejects external API URLs", () => {
    expect(() => getApiBaseUrl("https://example.com")).toThrow(/local http service/);
  });

  it("rejects URL credentials", () => {
    expect(() => getApiBaseUrl("http://user:pass@127.0.0.1:8000")).toThrow(/credentials/);
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
    expect(fetchMock).toHaveBeenCalledWith(new URL("http://127.0.0.1:8000/health"), {
      headers: new Headers(),
      body: undefined,
    });

    vi.unstubAllGlobals();
  });

  it("throws typed errors for failed responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockResolvedValue(
        new Response("nope", { status: 503 }),
      ),
    );

    await expect(apiFetch<HealthResponse>("/health")).rejects.toBeInstanceOf(ApiClientError);

    vi.unstubAllGlobals();
  });
});
