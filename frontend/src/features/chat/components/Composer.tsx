import { useEffect, useRef, useState, type FormEvent } from "react";

interface ComposerProps {
  acceptedSubmissionCount: number;
  disabled: boolean;
  errorMessage: string | null;
  isStreaming: boolean;
  onStop: () => void;
  onSubmit: (content: string) => Promise<boolean>;
}

export function Composer({
  acceptedSubmissionCount,
  disabled,
  errorMessage,
  isStreaming,
  onStop,
  onSubmit,
}: ComposerProps) {
  const [content, setContent] = useState("");
  const previousAcceptedSubmissionCountRef = useRef(acceptedSubmissionCount);
  const trimmedContent = content.trim();
  const inputDisabled = disabled || isStreaming;

  useEffect(() => {
    if (acceptedSubmissionCount !== previousAcceptedSubmissionCountRef.current) {
      previousAcceptedSubmissionCountRef.current = acceptedSubmissionCount;
      setContent("");
    }
  }, [acceptedSubmissionCount]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitContent();
  }

  async function submitContent() {
    if (trimmedContent === "" || inputDisabled) {
      return;
    }

    const shouldClear = await onSubmit(trimmedContent);

    if (shouldClear) {
      setContent("");
    }
  }

  return (
    <form className="shrink-0 border-t border-border bg-background px-6 py-3" onSubmit={handleSubmit}>
      <label className="sr-only" htmlFor="chat-composer">
        Message
      </label>
      <textarea
        className="min-h-16 w-full resize-y rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-accent focus:ring-2 focus:ring-accent/20 disabled:cursor-not-allowed disabled:bg-muted/50"
        disabled={inputDisabled}
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
      <div className="mt-2 flex justify-end">
        {isStreaming ? (
          <button
            className="rounded-md border border-danger bg-background px-4 py-2 text-sm font-medium text-danger transition hover:bg-danger/10"
            onClick={onStop}
            type="button"
          >
            Stop
          </button>
        ) : (
          <button
            className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-foreground transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground"
            disabled={disabled || trimmedContent === ""}
            type="submit"
          >
            Send
          </button>
        )}
      </div>
    </form>
  );
}
