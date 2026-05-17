import { useState, type FormEvent } from "react";

import { useCreateConversation } from "../api/useCreateConversation";
import type { ConversationResponse } from "../api/useConversations";

interface CreateConversationFormProps {
  onCreated: (conversation: ConversationResponse) => void;
  selectedScopeId: string | null;
}

export function CreateConversationForm({
  onCreated,
  selectedScopeId,
}: CreateConversationFormProps) {
  const [title, setTitle] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);
  const createConversation = useCreateConversation();

  const errorMessage =
    validationError ??
    (createConversation.isError ? "Conversation could not be created." : null);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitConversation();
  }

  async function submitConversation() {
    if (selectedScopeId === null) {
      setValidationError("Select a scope before creating a conversation.");
      return;
    }

    setValidationError(null);

    try {
      const createdConversation = await createConversation.mutateAsync({
        scope_id: selectedScopeId,
        title: title.trim() === "" ? null : title.trim(),
      });
      setTitle("");
      onCreated(createdConversation);
    } catch {
      // The inline mutation error state is rendered below without exposing request data.
    }
  }

  return (
    <form className="space-y-2" onSubmit={handleSubmit}>
      <label className="sr-only" htmlFor="new-conversation-title">
        Conversation title
      </label>
      <input
        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-accent focus:ring-2 focus:ring-accent/20 disabled:cursor-not-allowed disabled:bg-muted/50"
        disabled={selectedScopeId === null || createConversation.isPending}
        id="new-conversation-title"
        onChange={(event) => {
          setTitle(event.target.value);
          setValidationError(null);
        }}
        placeholder="Optional title"
        type="text"
        value={title}
      />
      {errorMessage === null ? null : (
        <p className="text-sm text-danger" role="alert">
          {errorMessage}
        </p>
      )}
      <button
        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm font-medium text-foreground transition hover:border-accent/70 hover:bg-muted/40 disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground"
        disabled={selectedScopeId === null || createConversation.isPending}
        type="submit"
      >
        {createConversation.isPending ? "Creating..." : "+ New conversation"}
      </button>
    </form>
  );
}
