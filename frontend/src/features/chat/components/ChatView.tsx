import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { useAssistantResponse } from "../api/useAssistantResponse";
import {
  estimateUserMessageTokenCount,
  useCreateMessage,
} from "../../messages/api/useCreateMessage";
import { messagesQueryKey, useMessages } from "../../messages/api/useMessages";
import { MessageList } from "../../messages/components/MessageList";
import { Composer } from "./Composer";

interface ChatViewProps {
  conversationId: string;
  scopeId: string;
  title: string;
}

interface AssistantFailure {
  queryMessageId: string;
}

export function ChatView({ conversationId, scopeId, title }: ChatViewProps) {
  const queryClient = useQueryClient();
  const messagesQuery = useMessages(conversationId);
  const createMessage = useCreateMessage(conversationId);
  const assistantResponse = useAssistantResponse(conversationId);
  const [composerError, setComposerError] = useState<string | null>(null);
  const [assistantFailure, setAssistantFailure] = useState<AssistantFailure | null>(null);
  const [isChainPending, setIsChainPending] = useState(false);

  async function refetchMessages() {
    await queryClient.invalidateQueries({ queryKey: messagesQueryKey(conversationId) });
  }

  async function handleSubmit(content: string): Promise<boolean> {
    setComposerError(null);
    setAssistantFailure(null);
    setIsChainPending(true);

    try {
      const createdMessage = await createMessage.mutateAsync({
        role: "user",
        content,
        token_count: estimateUserMessageTokenCount(content),
      });

      try {
        await assistantResponse.mutateAsync({
          scope_id: scopeId,
          query_message_id: createdMessage.id,
        });
        await refetchMessages();
        return true;
      } catch {
        setAssistantFailure({ queryMessageId: createdMessage.id });
        await refetchMessages();
        return true;
      }
    } catch {
      setComposerError("Message could not be saved.");
      return false;
    } finally {
      setIsChainPending(false);
    }
  }

  async function handleRetryAssistant() {
    if (assistantFailure === null) {
      return;
    }

    setIsChainPending(true);

    try {
      await assistantResponse.mutateAsync({
        scope_id: scopeId,
        query_message_id: assistantFailure.queryMessageId,
      });
      setAssistantFailure(null);
      await refetchMessages();
    } catch {
      await refetchMessages();
    } finally {
      setIsChainPending(false);
    }
  }

  return (
    <section className="flex min-h-0 flex-1 flex-col" aria-label={title}>
      <header className="border-b border-border bg-background px-6 py-4">
        <h1 className="text-xl font-semibold tracking-normal text-foreground">{title}</h1>
        {isChainPending ? (
          <p className="mt-2 text-sm text-muted-foreground" aria-live="polite">
            Assistant is thinking...
          </p>
        ) : null}
      </header>
      <MessageList
        assistantError={assistantFailure !== null}
        isError={messagesQuery.isError}
        isLoading={messagesQuery.isPending}
        isRetrying={isChainPending}
        messages={messagesQuery.data ?? []}
        onRetryAssistant={() => {
          void handleRetryAssistant();
        }}
      />
      <Composer disabled={isChainPending} errorMessage={composerError} onSubmit={handleSubmit} />
    </section>
  );
}
