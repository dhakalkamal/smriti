import { StrictMode, type ReactElement } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { client } from "../../../api/client";
import { createQueryClient } from "../../../lib/queryClient";
import { renderWithQueryClient } from "../../../test/renderWithQueryClient";
import {
  createControlledSseStream,
  sseFrame,
  type ControlledSseStream,
} from "../../../test/sseStream";
import { messagesQueryKey } from "../../messages/api/useMessages";
import { useAssistantResponseStream } from "./useAssistantResponseStream";

const createdUserMessage = {
  id: "message-user-1",
  conversation_id: "conversation-1",
  role: "user",
  content: "Hello memory",
  token_count: 3,
  position: 1,
  created_at: "2026-01-01T00:00:01Z",
  episode_id: "episode-user-1",
};

const assistantMessage = {
  id: "message-assistant-1",
  conversation_id: "conversation-1",
  role: "assistant",
  content: "Persisted assistant reply.",
  token_count: 6,
  position: 2,
  created_at: "2026-01-01T00:00:02Z",
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("useAssistantResponseStream", () => {
  it("transitions from idle to streaming and back to idle on done", async () => {
    mockCreateMessageFetch();
    let stream: ControlledSseStream | null = null;
    vi.spyOn(client, "postStream").mockImplementation((_path, _body, signal) => {
      stream = createControlledSseStream({ signal });
      return Promise.resolve(stream.response);
    });

    renderWithQueryClient(<HookHarness />);

    expect(screen.getByTestId("status")).toHaveTextContent("idle");

    fireEvent.click(screen.getByRole("button", { name: "Submit" }));

    await waitFor(() => {
      expect(client.postStream).toHaveBeenCalledTimes(1);
    });
    expect(screen.getByTestId("status")).toHaveTextContent("streaming");

    act(() => {
      stream?.enqueue(sseFrame("start", { used_memory_episode_ids: [], chat_model: null }));
      stream?.enqueue(sseFrame("token", { text: "Hello" }));
    });

    expect(await screen.findByTestId("draft")).toHaveTextContent("Hello");

    act(() => {
      stream?.enqueue(
        sseFrame("done", {
          assistant_message: assistantMessage,
          chat_model: "fake-chat",
          finish_reason: "stop",
          used_memory_episode_ids: [],
        }),
      );
      stream?.close();
    });

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("idle");
    });
    expect(screen.getByTestId("draft")).toBeEmptyDOMElement();
  });

  it("starts streaming after React StrictMode remounts effects in development", async () => {
    mockCreateMessageFetch();
    vi.spyOn(client, "postStream").mockImplementation((_path, _body, signal) =>
      Promise.resolve(createControlledSseStream({ signal }).response),
    );

    renderStrictWithQueryClient(<HookHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Submit" }));

    await waitFor(() => {
      expect(client.postStream).toHaveBeenCalledTimes(1);
    });
  });

  it("transitions to error when the stream emits an error event", async () => {
    mockCreateMessageFetch();
    let stream: ControlledSseStream | null = null;
    vi.spyOn(client, "postStream").mockImplementation((_path, _body, signal) => {
      stream = createControlledSseStream({ signal });
      return Promise.resolve(stream.response);
    });

    renderWithQueryClient(<HookHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Submit" }));

    await waitFor(() => {
      expect(client.postStream).toHaveBeenCalledTimes(1);
    });

    act(() => {
      stream?.enqueue(sseFrame("start", { used_memory_episode_ids: [], chat_model: null }));
      stream?.enqueue(sseFrame("token", { text: "partial" }));
      stream?.enqueue(sseFrame("error", { code: "backend_error", message: "hidden" }));
    });

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("error");
    });
    expect(screen.getByTestId("draft")).toBeEmptyDOMElement();
  });

  it("returns to idle on abort without entering the error state", async () => {
    mockCreateMessageFetch();
    let stream: ControlledSseStream | null = null;
    vi.spyOn(client, "postStream").mockImplementation((_path, _body, signal) => {
      stream = createControlledSseStream({ signal });
      return Promise.resolve(stream.response);
    });

    renderWithQueryClient(<HookHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Submit" }));

    await waitFor(() => {
      expect(client.postStream).toHaveBeenCalledTimes(1);
    });

    act(() => {
      stream?.enqueue(sseFrame("start", { used_memory_episode_ids: [], chat_model: null }));
      stream?.enqueue(sseFrame("token", { text: "partial" }));
    });

    expect(await screen.findByTestId("draft")).toHaveTextContent("partial");

    fireEvent.click(screen.getByRole("button", { name: "Stop" }));

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("idle");
    });
    expect(screen.getByTestId("draft")).toBeEmptyDOMElement();
  });

  it("retries with the same query message id without creating another user message", async () => {
    const fetchMock = mockCreateMessageFetch();
    const streams: ControlledSseStream[] = [];
    const postStreamSpy = vi
      .spyOn(client, "postStream")
      .mockImplementation((_path, _body, signal) => {
        const stream = createControlledSseStream({ signal });
        streams.push(stream);
        return Promise.resolve(stream.response);
      });

    renderWithQueryClient(<HookHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Submit" }));

    await waitFor(() => {
      expect(postStreamSpy).toHaveBeenCalledTimes(1);
    });

    act(() => {
      streams[0]?.enqueue(sseFrame("start", { used_memory_episode_ids: [], chat_model: null }));
      streams[0]?.enqueue(sseFrame("error", { code: "backend_error", message: "hidden" }));
    });

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("error");
    });

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    await waitFor(() => {
      expect(postStreamSpy).toHaveBeenCalledTimes(2);
    });

    act(() => {
      streams[1]?.enqueue(sseFrame("start", { used_memory_episode_ids: [], chat_model: null }));
      streams[1]?.enqueue(
        sseFrame("done", {
          assistant_message: assistantMessage,
          chat_model: "fake-chat",
          finish_reason: "stop",
          used_memory_episode_ids: [],
        }),
      );
      streams[1]?.close();
    });

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("idle");
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(postStreamSpy.mock.calls.map((call) => call[1])).toEqual([
      { scope_id: "scope-1", query_message_id: "message-user-1" },
      { scope_id: "scope-1", query_message_id: "message-user-1" },
    ]);
  });

  it("does not write streamed assistant data into the messages query cache", async () => {
    mockCreateMessageFetch();
    let stream: ControlledSseStream | null = null;
    vi.spyOn(client, "postStream").mockImplementation((_path, _body, signal) => {
      stream = createControlledSseStream({ signal });
      return Promise.resolve(stream.response);
    });
    const { queryClient } = renderWithQueryClient(<HookHarness />);
    const setQueryDataSpy = vi.spyOn(queryClient, "setQueryData");

    fireEvent.click(screen.getByRole("button", { name: "Submit" }));

    await waitFor(() => {
      expect(client.postStream).toHaveBeenCalledTimes(1);
    });

    act(() => {
      stream?.enqueue(sseFrame("start", { used_memory_episode_ids: [], chat_model: null }));
      stream?.enqueue(sseFrame("token", { text: "Transient draft" }));
      stream?.enqueue(
        sseFrame("done", {
          assistant_message: assistantMessage,
          chat_model: "fake-chat",
          finish_reason: "stop",
          used_memory_episode_ids: [],
        }),
      );
      stream?.close();
    });

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("idle");
    });
    expect(setQueryDataSpy).not.toHaveBeenCalledWith(
      messagesQueryKey("conversation-1"),
      expect.anything(),
    );
  });
});

function HookHarness() {
  const assistantStream = useAssistantResponseStream({
    conversationId: "conversation-1",
    scopeId: "scope-1",
  });

  return (
    <div>
      <p data-testid="status">{assistantStream.streamState.status}</p>
      <p data-testid="draft">{assistantStream.streamState.draft}</p>
      <button
        onClick={() => {
          void assistantStream.submitMessage("Hello memory");
        }}
        type="button"
      >
        Submit
      </button>
      <button
        onClick={() => {
          void assistantStream.retryAssistant();
        }}
        type="button"
      >
        Retry
      </button>
      <button onClick={assistantStream.stopStreaming} type="button">
        Stop
      </button>
    </div>
  );
}

function renderStrictWithQueryClient(ui: ReactElement): void {
  const queryClient = createQueryClient();

  render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>
    </StrictMode>,
  );
}

function mockCreateMessageFetch() {
  const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
    new Response(JSON.stringify(createdUserMessage), {
      status: 201,
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
