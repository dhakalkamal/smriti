import { describe, expect, it } from "vitest";

import { createSseResponseFromChunks, sseFrame } from "../test/sseStream";
import { parseAssistantSseStream, SseParseError, type AssistantSseEvent } from "./sse";

const assistantMessage = {
  id: "message-assistant-1",
  conversation_id: "conversation-1",
  role: "assistant",
  content: "Persisted assistant reply.",
  token_count: 6,
  position: 2,
  created_at: "2026-01-01T00:00:02Z",
} as const;

describe("parseAssistantSseStream", () => {
  it("parses frames split across chunk boundaries", async () => {
    const response = createSseResponseFromChunks([
      "event: sta",
      'rt\ndata: {"used_memory_episode_ids":[],"chat_model":null}\n\n',
      sseFrame("token", { text: "Hello" }),
      sseFrame("done", {
        assistant_message: assistantMessage,
        chat_model: "fake-chat",
        finish_reason: "stop",
        used_memory_episode_ids: [],
      }),
    ]);

    await expect(collectEvents(response)).resolves.toEqual([
      {
        event: "start",
        data: { used_memory_episode_ids: [], chat_model: null },
      },
      { event: "token", data: { text: "Hello" } },
      {
        event: "done",
        data: {
          assistant_message: assistantMessage,
          chat_model: "fake-chat",
          finish_reason: "stop",
          used_memory_episode_ids: [],
        },
      },
    ]);
  });

  it("rejects malformed frames", async () => {
    const response = createSseResponseFromChunks(['event: token\n{"text":"Hello"}\n\n']);

    await expect(collectEvents(response)).rejects.toBeInstanceOf(SseParseError);
  });

  it("rejects partial frames at end of stream", async () => {
    const response = createSseResponseFromChunks(["event: token\ndata: {\"text\":\"Hello\"}"]);

    await expect(collectEvents(response)).rejects.toBeInstanceOf(SseParseError);
  });

  it("rejects malformed JSON data", async () => {
    const response = createSseResponseFromChunks(["event: token\ndata: nope\n\n"]);

    await expect(collectEvents(response)).rejects.toBeInstanceOf(SseParseError);
  });

  it("rejects unrecognized events", async () => {
    const response = createSseResponseFromChunks([sseFrame("mystery", {})]);

    await expect(collectEvents(response)).rejects.toBeInstanceOf(SseParseError);
  });
});

async function collectEvents(response: Response): Promise<AssistantSseEvent[]> {
  if (response.body === null) {
    throw new Error("Expected a response body.");
  }

  const events: AssistantSseEvent[] = [];

  for await (const event of parseAssistantSseStream(response.body)) {
    events.push(event);
  }

  return events;
}
