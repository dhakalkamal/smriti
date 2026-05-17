import { useMemo } from "react";

import type { MessageResponse } from "../api/useMessages";
import { MessageItem } from "./MessageItem";

interface MessageListProps {
  assistantError: boolean;
  isError: boolean;
  isLoading: boolean;
  isRetrying: boolean;
  messages: MessageResponse[];
  onRetryAssistant: () => void;
}

export function MessageList({
  assistantError,
  isError,
  isLoading,
  isRetrying,
  messages,
  onRetryAssistant,
}: MessageListProps) {
  const sortedMessages = useMemo(
    () => [...messages].sort((left, right) => left.position - right.position),
    [messages],
  );

  if (isLoading) {
    return (
      <div className="flex min-h-0 flex-1 items-center justify-center px-6 py-10">
        <p className="text-sm text-muted-foreground">Loading messages...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex min-h-0 flex-1 items-center justify-center px-6 py-10">
        <p className="text-sm text-danger" role="alert">
          Messages could not be loaded.
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-6 py-6">
      {sortedMessages.length === 0 ? (
        <div className="flex min-h-full items-center justify-center">
          <p className="text-sm text-muted-foreground">
            No messages yet. Start the conversation when you are ready.
          </p>
        </div>
      ) : (
        <ol className="space-y-4" aria-label="Messages">
          {sortedMessages.map((message) => (
            <MessageItem key={message.id} message={message} />
          ))}
        </ol>
      )}
      {assistantError ? (
        <div className="mt-4 rounded-md border border-danger/40 bg-danger/10 px-4 py-3">
          <p className="text-sm font-medium text-danger" role="alert">
            Assistant response could not be created.
          </p>
          <button
            className="mt-3 rounded-md bg-danger px-3 py-2 text-sm font-medium text-white transition hover:bg-danger/90 disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground"
            disabled={isRetrying}
            onClick={onRetryAssistant}
            type="button"
          >
            {isRetrying ? "Retrying..." : "Retry assistant response"}
          </button>
        </div>
      ) : null}
    </div>
  );
}
