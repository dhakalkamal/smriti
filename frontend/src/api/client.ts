import type { paths } from "./types";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8100";
const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost"]);

type ApiPath = keyof paths;
type ConversationApiPath =
  | `/conversations/${string}`
  | `/conversations/${string}/assistant-response`
  | `/conversations/${string}/assistant-response/stream`
  | `/conversations/${string}/messages`
  | `/conversations/${string}/messages/${string}/retrievals`;
type LocalApiPath = ApiPath | ConversationApiPath;

interface ApiRequestOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
}

export class ApiClientError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
  }
}

export function getApiBaseUrl(envValue?: string): string {
  const trimmedValue = envValue?.trim();
  const baseUrl = trimmedValue === undefined || trimmedValue === "" ? DEFAULT_API_BASE_URL : trimmedValue;
  const url = new URL(baseUrl);

  if (url.protocol !== "http:" || !LOCAL_HOSTS.has(url.hostname)) {
    throw new Error("VITE_API_BASE_URL must point to a local http service.");
  }

  if (url.username !== "" || url.password !== "" || url.search !== "" || url.hash !== "") {
    throw new Error("VITE_API_BASE_URL must not include credentials, query, or fragment.");
  }

  return url.origin;
}

export async function apiFetch<TResponse>(
  path: LocalApiPath,
  options: ApiRequestOptions = {},
): Promise<TResponse> {
  const response = await fetch(buildApiUrl(path), buildRequestInit(options));

  if (!response.ok) {
    throw new ApiClientError(
      `Local API request failed with status ${response.status.toString()}.`,
      response.status,
    );
  }

  if (response.status === 204) {
    return undefined as TResponse;
  }

  return (await response.json()) as TResponse;
}

export async function postStream(
  path: string,
  body: unknown,
  signal: AbortSignal,
): Promise<Response> {
  return fetch(
    buildApiUrl(path),
    buildRequestInit({
      method: "POST",
      body,
      signal,
      headers: { Accept: "text/event-stream" },
    }),
  );
}

export const client = {
  apiFetch,
  postStream,
} as const;

function buildApiUrl(path: string): URL {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return new URL(normalizedPath, `${getConfiguredApiBaseUrl()}/`);
}

function getConfiguredApiBaseUrl(): string {
  const env = import.meta.env as { readonly VITE_API_BASE_URL?: string };
  return getApiBaseUrl(env.VITE_API_BASE_URL);
}

function buildRequestInit(options: ApiRequestOptions): RequestInit {
  const headers = new Headers(options.headers);
  let body: BodyInit | undefined;

  if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.body);
  }

  return {
    ...options,
    headers,
    body,
  };
}
