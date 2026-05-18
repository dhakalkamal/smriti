import { useState } from "react";

import { useDeleteConversation } from "../api/useDeleteConversation";
import type { ConversationResponse } from "../api/useConversations";
import { conversationTitleOrFallback } from "../lib/conversationTitle";
import { DeleteConversationDialog } from "./DeleteConversationDialog";

interface ConversationListProps {
  conversations: ConversationResponse[];
  isError?: boolean;
  isLoading?: boolean;
  onSelectConversation: (conversationId: string) => void;
  selectedConversationId: string | null;
  selectedScopeId: string | null;
}

export function ConversationList({
  conversations,
  isError = false,
  isLoading = false,
  onSelectConversation,
  selectedConversationId,
  selectedScopeId,
}: ConversationListProps) {
  const deleteConversation = useDeleteConversation();
  const [deleteTarget, setDeleteTarget] = useState<{
    id: string;
    title: string | null;
  } | null>(null);
  const filteredConversations =
    selectedScopeId === null
      ? []
      : conversations.filter((conversation) => conversation.scope_id === selectedScopeId);

  function openDeleteDialog(conversation: ConversationResponse) {
    deleteConversation.reset();
    setDeleteTarget({ id: conversation.id, title: conversation.title });
  }

  function closeDeleteDialog() {
    if (deleteConversation.isPending) {
      return;
    }

    deleteConversation.reset();
    setDeleteTarget(null);
  }

  function confirmDelete() {
    if (deleteTarget === null || deleteConversation.isPending) {
      return;
    }

    void deleteConversation
      .mutateAsync(deleteTarget.id)
      .then(() => {
        setDeleteTarget(null);
      })
      .catch(() => undefined);
  }

  if (selectedScopeId === null) {
    return <p className="text-sm text-muted-foreground">Select a scope to see conversations.</p>;
  }

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading conversations...</p>;
  }

  if (isError) {
    return (
      <p className="text-sm text-danger" role="alert">
        Conversations could not be loaded.
      </p>
    );
  }

  if (filteredConversations.length === 0) {
    return <p className="text-sm text-muted-foreground">No conversations in this scope yet.</p>;
  }

  return (
    <>
      <ul className="space-y-2" aria-label="Conversations">
        {filteredConversations.map((conversation) => {
          const isSelected = conversation.id === selectedConversationId;
          const displayTitle = conversationTitleOrFallback(conversation.title);

          return (
            <li className="flex items-stretch gap-2" key={conversation.id}>
              <button
                className={
                  isSelected
                    ? "min-w-0 flex-1 rounded-md border border-accent bg-accent/10 px-3 py-2 text-left text-sm font-medium text-foreground"
                    : "min-w-0 flex-1 rounded-md border border-border bg-background px-3 py-2 text-left text-sm text-foreground transition hover:border-accent/70 hover:bg-muted/40"
                }
                onClick={() => {
                  onSelectConversation(conversation.id);
                }}
                type="button"
              >
                <span className="block truncate">{displayTitle}</span>
              </button>
              {isSelected ? null : (
                <button
                  aria-label={`Delete ${displayTitle}`}
                  className="shrink-0 rounded-md border border-border bg-background px-2.5 py-2 text-xs font-medium text-muted-foreground transition hover:border-danger/70 hover:bg-danger/10 hover:text-danger"
                  onClick={() => {
                    openDeleteDialog(conversation);
                  }}
                  type="button"
                >
                  Delete
                </button>
              )}
            </li>
          );
        })}
      </ul>
      <DeleteConversationDialog
        errorMessage={
          deleteConversation.isError ? "Conversation could not be deleted." : null
        }
        isPending={deleteConversation.isPending}
        onCancel={closeDeleteDialog}
        onConfirm={confirmDelete}
        open={deleteTarget !== null}
        title={deleteTarget?.title ?? null}
      />
    </>
  );
}
