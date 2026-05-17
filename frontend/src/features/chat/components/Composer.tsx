import { useState, type FormEvent } from "react";

interface ComposerProps {
  disabled: boolean;
  errorMessage: string | null;
  onSubmit: (content: string) => Promise<boolean>;
}

export function Composer({ disabled, errorMessage, onSubmit }: ComposerProps) {
  const [content, setContent] = useState("");
  const trimmedContent = content.trim();

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitContent();
  }

  async function submitContent() {
    if (trimmedContent === "" || disabled) {
      return;
    }

    const shouldClear = await onSubmit(trimmedContent);

    if (shouldClear) {
      setContent("");
    }
  }

  return (
    <form className="border-t border-border bg-background px-6 py-4" onSubmit={handleSubmit}>
      <label className="sr-only" htmlFor="chat-composer">
        Message
      </label>
      <textarea
        className="min-h-28 w-full resize-y rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-accent focus:ring-2 focus:ring-accent/20 disabled:cursor-not-allowed disabled:bg-muted/50"
        disabled={disabled}
        id="chat-composer"
        onChange={(event) => {
          setContent(event.target.value);
        }}
        placeholder="Write a message..."
        value={content}
      />
      {errorMessage === null ? null : (
        <p className="mt-2 text-sm text-danger" role="alert">
          {errorMessage}
        </p>
      )}
      <div className="mt-3 flex justify-end">
        <button
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-foreground transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground"
          disabled={disabled || trimmedContent === ""}
          type="submit"
        >
          Send
        </button>
      </div>
    </form>
  );
}
