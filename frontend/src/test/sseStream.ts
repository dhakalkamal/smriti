const encoder = new TextEncoder();

interface SseResponseOptions {
  signal?: AbortSignal;
  status?: number;
}

export interface ControlledSseStream {
  response: Response;
  enqueue: (chunk: string) => void;
  close: () => void;
  error: (error?: unknown) => void;
}

export function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export function createSseResponseFromChunks(
  chunks: string[],
  options: SseResponseOptions = {},
): Response {
  const queuedChunks = [...chunks];

  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        attachAbortSignal(controller, options.signal);
      },
      pull(controller) {
        if (options.signal?.aborted) {
          controller.error(createAbortError());
          return;
        }

        const chunk = queuedChunks.shift();

        if (chunk === undefined) {
          controller.close();
          return;
        }

        controller.enqueue(encoder.encode(chunk));
      },
    }),
    responseInit(options.status),
  );
}

export function createControlledSseStream(
  options: SseResponseOptions = {},
): ControlledSseStream {
  let streamController: ReadableStreamDefaultController<Uint8Array> | null = null;
  let isClosed = false;

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      streamController = controller;
      attachAbortSignal(controller, options.signal);
    },
  });

  return {
    response: new Response(stream, responseInit(options.status)),
    enqueue(chunk: string) {
      if (isClosed || streamController === null) {
        return;
      }

      streamController.enqueue(encoder.encode(chunk));
    },
    close() {
      if (isClosed || streamController === null) {
        return;
      }

      isClosed = true;
      streamController.close();
    },
    error(error: unknown = new Error("Controlled SSE stream error.")) {
      if (isClosed || streamController === null) {
        return;
      }

      isClosed = true;
      streamController.error(error);
    },
  };
}

function attachAbortSignal(
  controller: ReadableStreamDefaultController<Uint8Array>,
  signal: AbortSignal | undefined,
): void {
  if (signal === undefined) {
    return;
  }

  if (signal.aborted) {
    controller.error(createAbortError());
    return;
  }

  signal.addEventListener(
    "abort",
    () => {
      controller.error(createAbortError());
    },
    { once: true },
  );
}

function createAbortError(): DOMException {
  return new DOMException("The operation was aborted.", "AbortError");
}

function responseInit(status = 200): ResponseInit {
  return {
    status,
    headers: { "Content-Type": "text/event-stream" },
  };
}
