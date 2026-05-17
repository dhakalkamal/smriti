export function readJsonRequestBody(init: RequestInit | undefined): unknown {
  if (typeof init?.body !== "string") {
    throw new Error("Expected a JSON string request body.");
  }

  return JSON.parse(init.body) as unknown;
}
