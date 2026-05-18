import { useState } from "react";

import { ChatEmptyState } from "../features/chat/components/ChatEmptyState";
import { ChatView } from "../features/chat/components/ChatView";
import { ConversationList } from "../features/conversations/components/ConversationList";
import { CreateConversationForm } from "../features/conversations/components/CreateConversationForm";
import { useConversations } from "../features/conversations/api/useConversations";
import { conversationTitleOrFallback } from "../features/conversations/lib/conversationTitle";
import { CreateScopeForm } from "../features/scopes/components/CreateScopeForm";
import { ScopeSelector } from "../features/scopes/components/ScopeSelector";
import { useScopes } from "../features/scopes/api/useScopes";

function ChatPage() {
  const scopesQuery = useScopes();
  const conversationsQuery = useConversations();
  const [selectedScopeId, setSelectedScopeId] = useState<string | null>(null);
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null);

  const scopes = scopesQuery.data ?? [];
  const conversations = conversationsQuery.data ?? [];
  const selectedScope = scopes.find((scope) => scope.id === selectedScopeId) ?? null;
  const conversationsForSelectedScope =
    selectedScopeId === null
      ? []
      : conversations.filter((conversation) => conversation.scope_id === selectedScopeId);
  const selectedConversation =
    conversationsForSelectedScope.find(
      (conversation) => conversation.id === selectedConversationId,
    ) ?? null;

  function handleSelectedScopeIdChange(scopeId: string | null) {
    setSelectedScopeId(scopeId);
    setSelectedConversationId(null);
  }

  return (
    <main className="flex h-dvh min-h-0 flex-col overflow-hidden bg-background text-foreground lg:flex-row">
      <aside className="flex min-h-0 w-full flex-col gap-6 overflow-y-auto border-b border-border bg-muted/30 px-5 py-6 lg:h-dvh lg:max-w-sm lg:shrink-0 lg:border-b-0 lg:border-r">
        <div className="border-l-4 border-accent pl-4">
          <p className="text-sm font-medium uppercase tracking-normal text-muted-foreground">
            Smriti
          </p>
          <h1 className="mt-2 text-2xl font-semibold tracking-normal text-foreground">
            Chat
          </h1>
        </div>

        <ScopeSelector
          isError={scopesQuery.isError}
          isLoading={scopesQuery.isPending}
          onSelectedScopeIdChange={handleSelectedScopeIdChange}
          scopes={scopes}
          selectedScopeId={selectedScopeId}
        />

        <CreateScopeForm
          onCreated={(scope) => {
            setSelectedScopeId(scope.id);
            setSelectedConversationId(null);
          }}
        />

        <section className="min-h-0 flex-1 space-y-4">
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-normal text-muted-foreground">
              Conversations
            </h2>
            {selectedScope === null ? null : (
              <p className="mt-1 text-sm text-muted-foreground">{selectedScope.name}</p>
            )}
          </div>
          <CreateConversationForm
            onCreated={(conversation) => {
              setSelectedConversationId(conversation.id);
            }}
            selectedScopeId={selectedScopeId}
          />
          <ConversationList
            conversations={conversations}
            isError={conversationsQuery.isError}
            isLoading={conversationsQuery.isPending}
            onSelectConversation={setSelectedConversationId}
            selectedConversationId={selectedConversationId}
            selectedScopeId={selectedScopeId}
          />
        </section>
      </aside>

      <section className="flex min-h-0 min-w-0 flex-1 overflow-hidden">
        {renderMainPane({
          conversationsLoading: conversationsQuery.isPending,
          conversationsForSelectedScopeCount: conversationsForSelectedScope.length,
          scopesCount: scopes.length,
          scopesLoading: scopesQuery.isPending,
          selectedConversationId,
          selectedConversationTitle: conversationTitleOrFallback(selectedConversation?.title),
          selectedScopeId,
          onSelectedConversationDeleted: () => {
            setSelectedConversationId(null);
          },
        })}
      </section>
    </main>
  );
}

interface MainPaneState {
  conversationsForSelectedScopeCount: number;
  conversationsLoading: boolean;
  scopesCount: number;
  scopesLoading: boolean;
  selectedConversationId: string | null;
  selectedConversationTitle: string;
  selectedScopeId: string | null;
  onSelectedConversationDeleted: () => void;
}

function renderMainPane({
  conversationsForSelectedScopeCount,
  conversationsLoading,
  onSelectedConversationDeleted,
  scopesCount,
  scopesLoading,
  selectedConversationId,
  selectedConversationTitle,
  selectedScopeId,
}: MainPaneState) {
  if (scopesLoading) {
    return (
      <ChatEmptyState
        description="Loading the local scope list."
        title="Checking your scopes"
      />
    );
  }

  if (scopesCount === 0) {
    return (
      <ChatEmptyState
        description="Create a scope in the sidebar, then start a conversation inside it."
        title="Create your first scope"
      />
    );
  }

  if (selectedScopeId === null) {
    return (
      <ChatEmptyState
        description="Choose a scope from the sidebar to keep retrieval bounded to that memory partition."
        title="Select a scope"
      />
    );
  }

  if (conversationsLoading) {
    return (
      <ChatEmptyState
        description="Loading the conversations for the selected scope."
        title="Checking conversations"
      />
    );
  }

  if (conversationsForSelectedScopeCount === 0) {
    return (
      <ChatEmptyState
        description="Create a conversation in the sidebar to begin chatting in this scope."
        title="Create a conversation"
      />
    );
  }

  if (selectedConversationId === null) {
    return (
      <ChatEmptyState
        description="Pick a conversation from the sidebar, or create a new one."
        title="Select a conversation"
      />
    );
  }

  return (
    <ChatView
      conversationId={selectedConversationId}
      key={selectedConversationId}
      onDeleted={onSelectedConversationDeleted}
      scopeId={selectedScopeId}
      title={selectedConversationTitle}
    />
  );
}

export default ChatPage;
