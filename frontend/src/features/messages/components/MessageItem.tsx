import { useState } from "react";

import { cn } from "../../../lib/cn";
import type { MessageResponse } from "../api/useMessages";
import { MessageRetrievalsPanel } from "./MessageRetrievalsPanel";

interface MessageItemProps {
  message: MessageResponse;
}

export function MessageItem({ message }: MessageItemProps) {
  const [isRetrievalsOpen, setIsRetrievalsOpen] = useState(false);
  const isUserMessage = message.role === "user";
  const isAssistantMessage = message.role === "assistant";
  const label =
    message.role === "user"
      ? "You"
      : message.role === "assistant"
        ? "Assistant"
        : "System";

  return (
    <li className={cn("flex", isUserMessage ? "justify-end" : "justify-start")}>
      <article
        className={cn(
          "max-w-[78%] rounded-lg border px-4 py-3 text-sm leading-6 shadow-sm",
          isUserMessage
            ? "border-accent bg-accent text-accent-foreground"
            : "border-border bg-background text-foreground",
        )}
      >
        <p
          className={cn(
            "mb-1 text-xs font-medium",
            isUserMessage ? "text-accent-foreground/80" : "text-muted-foreground",
          )}
        >
          {label}
        </p>
        <p className="whitespace-pre-wrap break-words">{message.content}</p>
        {isAssistantMessage ? (
          <>
            <button
              aria-expanded={isRetrievalsOpen}
              className="mt-3 text-xs font-medium text-accent transition hover:text-accent/80"
              onClick={() => {
                setIsRetrievalsOpen((currentValue) => !currentValue);
              }}
              type="button"
            >
              {isRetrievalsOpen ? "Hide retrieved memories" : "View retrieved memories"}
            </button>
            <MessageRetrievalsPanel
              conversationId={message.conversation_id}
              isOpen={isRetrievalsOpen}
              messageId={message.id}
            />
          </>
        ) : null}
      </article>
    </li>
  );
}
