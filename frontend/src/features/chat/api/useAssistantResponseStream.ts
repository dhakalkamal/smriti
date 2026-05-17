import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { client } from "../../../api/client";
import { parseAssistantSseStream } from "../../../lib/sse";
import {
  estimateUserMessageTokenCount,
  useCreateMessage,
} from "../../messages/api/useCreateMessage";
import { messagesQueryKey } from "../../messages/api/useMessages";

export type AssistantStreamState =
  | { status: "idle"; draft: "" }
  | { status: "streaming"; draft: string }
  | { status: "error"; queryMessageId: string; draft: "" };

interface UseAssistantResponseStreamParams {
  conversationId: string;
  scopeId: string;
}

interface UseAssistantResponseStreamResult {
  streamState: AssistantStreamState;
  acceptedSubmissionCount: number;
  composerError: string | null;
  isSubmitting: boolean;
  isStreaming: boolean;
  /**
   * Resolves true once the user message was accepted and the lifecycle reached
   * a terminal state; true does not imply assistant success.
   */
  submitMessage: (content: string) => Promise<boolean>;
  retryAssistant: () => Promise<void>;
  stopStreaming: () => void;
}

class AssistantStreamFailure extends Error {
  constructor() {
    super("Assistant stream failed.");
    this.name = "AssistantStreamFailure";
  }
}

export function useAssistantResponseStream({
  conversationId,
  scopeId,
}: UseAssistantResponseStreamParams): UseAssistantResponseStreamResult {
  const queryClient = useQueryClient();
  const createMessage = useCreateMessage(conversationId);
  const activeControllerRef = useRef<AbortController | null>(null);
  const isMountedRef = useRef(true);
  const isSubmitInFlightRef = useRef(false);
  const [streamState, setStreamState] = useState<AssistantStreamState>({
    status: "idle",
    draft: "",
  });
  const [acceptedSubmissionCount, setAcceptedSubmissionCount] = useState(0);
  const [composerError, setComposerError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const setStreamStateIfMounted = useCallback((state: AssistantStreamState) => {
    if (isMountedRef.current) {
      setStreamState(state);
    }
  }, []);

  const setComposerErrorIfMounted = useCallback((message: string | null) => {
    if (isMountedRef.current) {
      setComposerError(message);
    }
  }, []);

  const setIsSubmittingIfMounted = useCallback((value: boolean) => {
    if (isMountedRef.current) {
      setIsSubmitting(value);
    }
  }, []);

  const markSubmissionAcceptedIfMounted = useCallback(() => {
    if (isMountedRef.current) {
      setAcceptedSubmissionCount((currentCount) => currentCount + 1);
    }
  }, []);

  const refetchMessages = useCallback(
    async (throwOnError: boolean) => {
      await queryClient.refetchQueries(
        { queryKey: messagesQueryKey(conversationId) },
        { throwOnError },
      );
    },
    [conversationId, queryClient],
  );

  const startAssistantStream = useCallback(
    async (queryMessageId: string) => {
      if (activeControllerRef.current !== null) {
        return;
      }

      const controller = new AbortController();
      activeControllerRef.current = controller;

      try {
        const encodedConversationId = encodeURIComponent(conversationId);
        const responsePromise = client.postStream(
          `/conversations/${encodedConversationId}/assistant-response/stream`,
          {
            scope_id: scopeId,
            query_message_id: queryMessageId,
          },
          controller.signal,
        );
        setStreamStateIfMounted({ status: "streaming", draft: "" });
        const response = await responsePromise;

        if (!response.ok || response.body === null) {
          throw new AssistantStreamFailure();
        }

        let sawStart = false;

        for await (const streamEvent of parseAssistantSseStream(response.body)) {
          if (!isMountedRef.current) {
            return;
          }

          if (streamEvent.event === "start") {
            if (sawStart) {
              throw new AssistantStreamFailure();
            }

            sawStart = true;
            setStreamStateIfMounted({ status: "streaming", draft: "" });
            continue;
          }

          if (streamEvent.event === "token") {
            if (!sawStart) {
              throw new AssistantStreamFailure();
            }

            setStreamState((currentState) =>
              currentState.status === "streaming"
                ? {
                    status: "streaming",
                    draft: `${currentState.draft}${streamEvent.data.text}`,
                  }
                : currentState,
            );
            continue;
          }

          if (streamEvent.event === "done") {
            if (!sawStart) {
              throw new AssistantStreamFailure();
            }

            await refetchMessages(false);
            setStreamStateIfMounted({ status: "idle", draft: "" });
            return;
          }

          throw new AssistantStreamFailure();
        }

        throw new AssistantStreamFailure();
      } catch (error) {
        if (controller.signal.aborted || isAbortError(error)) {
          setStreamStateIfMounted({ status: "idle", draft: "" });
          return;
        }

        throw error;
      } finally {
        if (activeControllerRef.current === controller) {
          activeControllerRef.current = null;
        }
      }
    },
    [conversationId, refetchMessages, scopeId, setStreamStateIfMounted],
  );

  const submitMessage = useCallback(
    async (content: string): Promise<boolean> => {
      if (isSubmitInFlightRef.current || activeControllerRef.current !== null) {
        return false;
      }

      isSubmitInFlightRef.current = true;
      setIsSubmittingIfMounted(true);
      setComposerErrorIfMounted(null);
      setStreamStateIfMounted({ status: "idle", draft: "" });

      try {
        const createdMessage = await createMessage.mutateAsync({
          role: "user",
          content,
          token_count: estimateUserMessageTokenCount(content),
        });
        markSubmissionAcceptedIfMounted();

        try {
          await refetchMessages(true);
        } catch {
          setComposerErrorIfMounted("Messages could not be refreshed.");
          return true;
        }

        if (!isMountedRef.current) {
          return true;
        }

        try {
          await startAssistantStream(createdMessage.id);
        } catch {
          setStreamStateIfMounted({
            status: "error",
            queryMessageId: createdMessage.id,
            draft: "",
          });
        }

        return true;
      } catch {
        setComposerErrorIfMounted("Message could not be saved.");
        return false;
      } finally {
        isSubmitInFlightRef.current = false;
        setIsSubmittingIfMounted(false);
      }
    },
    [
      createMessage,
      markSubmissionAcceptedIfMounted,
      refetchMessages,
      setComposerErrorIfMounted,
      setIsSubmittingIfMounted,
      setStreamStateIfMounted,
      startAssistantStream,
    ],
  );

  const retryAssistant = useCallback(async () => {
    if (streamState.status !== "error" || activeControllerRef.current !== null) {
      return;
    }

    setComposerErrorIfMounted(null);

    try {
      await startAssistantStream(streamState.queryMessageId);
    } catch {
      setStreamStateIfMounted({
        status: "error",
        queryMessageId: streamState.queryMessageId,
        draft: "",
      });
    }
  }, [
    setComposerErrorIfMounted,
    setStreamStateIfMounted,
    startAssistantStream,
    streamState,
  ]);

  const stopStreaming = useCallback(() => {
    activeControllerRef.current?.abort();
  }, []);

  useEffect(() => {
    isMountedRef.current = true;

    return () => {
      isMountedRef.current = false;
      activeControllerRef.current?.abort();
    };
  }, []);

  return {
    streamState,
    acceptedSubmissionCount,
    composerError,
    isSubmitting,
    isStreaming: streamState.status === "streaming",
    submitMessage,
    retryAssistant,
    stopStreaming,
  };
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
