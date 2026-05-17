import type { components } from "../api/types";

type MessageResponse = components["schemas"]["MessageResponse"];

// Mirrors backend Pydantic schemas in src/smriti/api/schemas.py:
// AssistantStreamStartData, AssistantStreamTokenData, AssistantStreamDoneData,
// and AssistantStreamErrorData.
export interface AssistantStreamStartData {
  used_memory_episode_ids: string[];
  chat_model?: string | null;
}

export interface AssistantStreamTokenData {
  text: string;
}

export interface AssistantStreamDoneData {
  assistant_message: MessageResponse;
  chat_model: string;
  finish_reason: string | null;
  used_memory_episode_ids: string[];
}

export interface AssistantStreamErrorData {
  code: string;
  message: string;
}

export type AssistantSseEvent =
  | { event: "start"; data: AssistantStreamStartData }
  | { event: "token"; data: AssistantStreamTokenData }
  | { event: "done"; data: AssistantStreamDoneData }
  | { event: "error"; data: AssistantStreamErrorData };

export class SseParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SseParseError";
  }
}

export async function* parseAssistantSseStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<AssistantSseEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    let readResult = await reader.read();

    while (!readResult.done) {
      buffer += decoder.decode(readResult.value, { stream: true });

      let frameBoundary = buffer.indexOf("\n\n");

      while (frameBoundary !== -1) {
        const frame = buffer.slice(0, frameBoundary);
        buffer = buffer.slice(frameBoundary + 2);

        if (frame.trim() !== "") {
          yield parseAssistantSseFrame(frame);
        }

        frameBoundary = buffer.indexOf("\n\n");
      }

      readResult = await reader.read();
    }

    buffer += decoder.decode();

    if (buffer.trim() !== "") {
      throw new SseParseError("SSE stream ended with a partial frame.");
    }
  } finally {
    reader.releaseLock();
  }
}

function parseAssistantSseFrame(frame: string): AssistantSseEvent {
  const lines = frame.split(/\r?\n/u);
  const eventLines = lines.filter((line) => line.startsWith("event:"));
  const dataLines = lines.filter((line) => line.startsWith("data:"));

  if (
    eventLines.length !== 1 ||
    dataLines.length !== 1 ||
    eventLines.length + dataLines.length !== lines.length
  ) {
    throw new SseParseError("Malformed SSE frame.");
  }

  const eventName = eventLines[0].slice("event:".length).trim();
  const dataText = dataLines[0].slice("data:".length).trim();
  const data = parseJsonObject(dataText);

  switch (eventName) {
    case "start":
      return { event: "start", data: parseStartData(data) };
    case "token":
      return { event: "token", data: parseTokenData(data) };
    case "done":
      return { event: "done", data: parseDoneData(data) };
    case "error":
      return { event: "error", data: parseErrorData(data) };
    default:
      throw new SseParseError("Unrecognized SSE event.");
  }
}

function parseJsonObject(json: string): Record<string, unknown> {
  try {
    const value: unknown = JSON.parse(json);

    if (!isRecord(value)) {
      throw new SseParseError("SSE data must be a JSON object.");
    }

    return value;
  } catch (error) {
    if (error instanceof SseParseError) {
      throw error;
    }

    throw new SseParseError("SSE data is not valid JSON.");
  }
}

function parseStartData(data: Record<string, unknown>): AssistantStreamStartData {
  assertOnlyKeys(data, ["used_memory_episode_ids", "chat_model"]);

  if (!isStringArray(data.used_memory_episode_ids)) {
    throw new SseParseError("Malformed start event data.");
  }

  if (
    data.chat_model !== undefined &&
    data.chat_model !== null &&
    typeof data.chat_model !== "string"
  ) {
    throw new SseParseError("Malformed start event data.");
  }

  return {
    used_memory_episode_ids: data.used_memory_episode_ids,
    chat_model: data.chat_model,
  };
}

function parseTokenData(data: Record<string, unknown>): AssistantStreamTokenData {
  assertOnlyKeys(data, ["text"]);

  if (typeof data.text !== "string") {
    throw new SseParseError("Malformed token event data.");
  }

  return { text: data.text };
}

function parseDoneData(data: Record<string, unknown>): AssistantStreamDoneData {
  assertOnlyKeys(data, [
    "assistant_message",
    "chat_model",
    "finish_reason",
    "used_memory_episode_ids",
  ]);

  if (
    !isMessageResponse(data.assistant_message) ||
    typeof data.chat_model !== "string" ||
    (data.finish_reason !== null && typeof data.finish_reason !== "string") ||
    !isStringArray(data.used_memory_episode_ids)
  ) {
    throw new SseParseError("Malformed done event data.");
  }

  return {
    assistant_message: data.assistant_message,
    chat_model: data.chat_model,
    finish_reason: data.finish_reason,
    used_memory_episode_ids: data.used_memory_episode_ids,
  };
}

function parseErrorData(data: Record<string, unknown>): AssistantStreamErrorData {
  assertOnlyKeys(data, ["code", "message"]);

  if (typeof data.code !== "string" || typeof data.message !== "string") {
    throw new SseParseError("Malformed error event data.");
  }

  return {
    code: data.code,
    message: data.message,
  };
}

function assertOnlyKeys(data: Record<string, unknown>, allowedKeys: string[]): void {
  const allowed = new Set(allowedKeys);
  const keys = Object.keys(data);

  if (keys.some((key) => !allowed.has(key))) {
    throw new SseParseError("SSE data contains unexpected fields.");
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isMessageResponse(value: unknown): value is MessageResponse {
  if (!isRecord(value)) {
    return false;
  }

  return (
    typeof value.id === "string" &&
    typeof value.conversation_id === "string" &&
    isMessageRole(value.role) &&
    typeof value.content === "string" &&
    typeof value.token_count === "number" &&
    typeof value.position === "number" &&
    typeof value.created_at === "string"
  );
}

function isMessageRole(value: unknown): value is MessageResponse["role"] {
  return value === "system" || value === "user" || value === "assistant";
}
