import type { ConversationResponse } from "../api/useConversations";

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
  const filteredConversations =
    selectedScopeId === null
      ? []
      : conversations.filter((conversation) => conversation.scope_id === selectedScopeId);

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
    <ul className="space-y-2" aria-label="Conversations">
      {filteredConversations.map((conversation) => {
        const isSelected = conversation.id === selectedConversationId;

        return (
          <li key={conversation.id}>
            <button
              className={
                isSelected
                  ? "w-full rounded-md border border-accent bg-accent/10 px-3 py-2 text-left text-sm font-medium text-foreground"
                  : "w-full rounded-md border border-border bg-background px-3 py-2 text-left text-sm text-foreground transition hover:border-accent/70 hover:bg-muted/40"
              }
              onClick={() => {
                onSelectConversation(conversation.id);
              }}
              type="button"
            >
              {conversation.title ?? "Untitled conversation"}
            </button>
          </li>
        );
      })}
    </ul>
  );
}
