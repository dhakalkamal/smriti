import { useState, type FormEvent } from "react";

import { useCreateScope } from "../api/useCreateScope";
import type { ScopeResponse } from "../api/useScopes";

interface CreateScopeFormProps {
  onCreated: (scope: ScopeResponse) => void;
}

export function CreateScopeForm({ onCreated }: CreateScopeFormProps) {
  const [name, setName] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);
  const createScope = useCreateScope();

  const trimmedName = name.trim();
  const errorMessage =
    validationError ?? (createScope.isError ? "Scope could not be created." : null);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitScope();
  }

  async function submitScope() {
    if (trimmedName === "") {
      setValidationError("Scope name is required.");
      return;
    }

    setValidationError(null);

    try {
      const createdScope = await createScope.mutateAsync({
        name: trimmedName,
        system_prompt: systemPrompt.trim(),
      });
      setName("");
      setSystemPrompt("");
      onCreated(createdScope);
    } catch {
      // The inline mutation error state is rendered below without exposing request data.
    }
  }

  return (
    <form className="space-y-3" onSubmit={handleSubmit}>
      <div className="space-y-1.5">
        <label className="text-sm font-medium text-foreground" htmlFor="new-scope-name">
          New scope
        </label>
        <input
          aria-invalid={errorMessage === null ? undefined : true}
          className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-accent focus:ring-2 focus:ring-accent/20"
          disabled={createScope.isPending}
          id="new-scope-name"
          onChange={(event) => {
            setName(event.target.value);
            setValidationError(null);
          }}
          placeholder="Research Notes"
          type="text"
          value={name}
        />
      </div>
      <div className="space-y-1.5">
        <label className="text-sm font-medium text-foreground" htmlFor="new-scope-prompt">
          System prompt
        </label>
        <textarea
          className="min-h-20 w-full resize-y rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-accent focus:ring-2 focus:ring-accent/20"
          disabled={createScope.isPending}
          id="new-scope-prompt"
          onChange={(event) => {
            setSystemPrompt(event.target.value);
          }}
          placeholder="Optional"
          value={systemPrompt}
        />
      </div>
      {errorMessage === null ? null : (
        <p className="text-sm text-danger" role="alert">
          {errorMessage}
        </p>
      )}
      <button
        className="w-full rounded-md bg-accent px-3 py-2 text-sm font-medium text-accent-foreground transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground"
        disabled={createScope.isPending || trimmedName === ""}
        type="submit"
      >
        {createScope.isPending ? "Creating..." : "Create scope"}
      </button>
    </form>
  );
}
