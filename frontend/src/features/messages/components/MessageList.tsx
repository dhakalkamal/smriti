import { useMemo } from "react";

import type { MessageResponse } from "../api/useMessages";
import { MessageItem } from "./MessageItem";

interface MessageListProps {
  isError: boolean;
  isLoading: boolean;
  messages: MessageResponse[];
}

export function MessageList({
  isError,
  isLoading,
  messages,
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
    </div>
  );
}
