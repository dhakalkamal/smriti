import { useState } from "react";

import { useAssistantResponseStream } from "../api/useAssistantResponseStream";
import { useDeleteConversation } from "../../conversations/api/useDeleteConversation";
import { DeleteConversationDialog } from "../../conversations/components/DeleteConversationDialog";
import { useMessages } from "../../messages/api/useMessages";
import { MessageList } from "../../messages/components/MessageList";
import { Composer } from "./Composer";

interface ChatViewProps {
  conversationId: string;
  onDeleted?: () => void;
  scopeId: string;
  title: string;
}

export function ChatView({ conversationId, onDeleted, scopeId, title }: ChatViewProps) {
  const messagesQuery = useMessages(conversationId);
  const assistantStream = useAssistantResponseStream({ conversationId, scopeId });
  const deleteConversation = useDeleteConversation();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

  function openDeleteDialog() {
    if (assistantStream.isStreaming) {
      return;
    }

    deleteConversation.reset();
    setDeleteDialogOpen(true);
  }

  function closeDeleteDialog() {
    if (deleteConversation.isPending) {
      return;
    }

    deleteConversation.reset();
    setDeleteDialogOpen(false);
  }

  function confirmDelete() {
    if (assistantStream.isStreaming || deleteConversation.isPending) {
      return;
    }

    void deleteConversation
      .mutateAsync(conversationId)
      .then(() => {
        setDeleteDialogOpen(false);
        onDeleted?.();
      })
      .catch(() => undefined);
  }

  return (
    <section className="flex min-h-0 flex-1 flex-col overflow-hidden" aria-label={title}>
      <header className="shrink-0 border-b border-border bg-background px-6 py-3">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h1 className="truncate text-xl font-semibold tracking-normal text-foreground">
              {title}
            </h1>
            {assistantStream.isSubmitting ? (
              <p className="mt-2 text-sm text-muted-foreground" aria-live="polite">
                Saving message...
              </p>
            ) : null}
            {assistantStream.isStreaming ? (
              <p className="mt-2 text-sm text-muted-foreground" aria-live="polite">
                Assistant is responding...
              </p>
            ) : null}
          </div>
          <div className="shrink-0 text-right">
            <button
              aria-describedby={
                assistantStream.isStreaming ? "active-delete-streaming-note" : undefined
              }
              className="rounded-md border border-border bg-background px-3 py-2 text-xs font-medium text-muted-foreground transition hover:border-danger/70 hover:bg-danger/10 hover:text-danger disabled:cursor-not-allowed disabled:opacity-60"
              disabled={assistantStream.isStreaming}
              onClick={openDeleteDialog}
              type="button"
            >
              Delete conversation
            </button>
            {assistantStream.isStreaming ? (
              <p
                className="mt-2 max-w-48 text-xs leading-5 text-muted-foreground"
                id="active-delete-streaming-note"
              >
                Deletion is unavailable while the assistant is responding.
              </p>
            ) : null}
          </div>
        </div>
      </header>
      <MessageList
        isError={messagesQuery.isError}
        isLoading={messagesQuery.isPending}
        messages={messagesQuery.data ?? []}
      />
      <AssistantStreamPanel
        draft={assistantStream.streamState.draft}
        isError={assistantStream.streamState.status === "error"}
        isStreaming={assistantStream.isStreaming}
        onRetryAssistant={() => {
          void assistantStream.retryAssistant();
        }}
      />
      <Composer
        acceptedSubmissionCount={assistantStream.acceptedSubmissionCount}
        disabled={assistantStream.isSubmitting}
        errorMessage={assistantStream.composerError}
        isStreaming={assistantStream.isStreaming}
        onStop={assistantStream.stopStreaming}
        onSubmit={assistantStream.submitMessage}
      />
      <DeleteConversationDialog
        errorMessage={
          deleteConversation.isError ? "Conversation could not be deleted." : null
        }
        isPending={deleteConversation.isPending}
        onCancel={closeDeleteDialog}
        onConfirm={confirmDelete}
        open={deleteDialogOpen}
        title={title}
      />
    </section>
  );
}

interface AssistantStreamPanelProps {
  draft: string;
  isError: boolean;
  isStreaming: boolean;
  onRetryAssistant: () => void;
}

function AssistantStreamPanel({
  draft,
  isError,
  isStreaming,
  onRetryAssistant,
}: AssistantStreamPanelProps) {
  if (draft !== "") {
    return (
      <div className="shrink-0 border-t border-border bg-muted/20 px-6 py-3">
        <article className="max-h-48 max-w-[78%] overflow-y-auto rounded-lg border border-border bg-background px-4 py-2.5 text-sm leading-6 text-foreground shadow-sm">
          <p className="mb-1 text-xs font-medium text-muted-foreground">Assistant</p>
          <p className="whitespace-pre-wrap break-words">
            {draft}
            {isStreaming ? (
              <span className="ml-1 text-muted-foreground" aria-hidden="true">
                ...
              </span>
            ) : null}
          </p>
        </article>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="shrink-0 border-t border-border bg-background px-6 py-3">
        <div className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3">
          <p className="text-sm font-medium text-danger" role="alert">
            Assistant failed to respond.
          </p>
          <button
            className="mt-3 rounded-md bg-danger px-3 py-2 text-sm font-medium text-white transition hover:bg-danger/90"
            onClick={onRetryAssistant}
            type="button"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return null;
}
