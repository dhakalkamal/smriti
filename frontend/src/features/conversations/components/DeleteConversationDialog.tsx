import { useEffect, useId } from "react";

import { conversationTitleOrFallback } from "../lib/conversationTitle";

interface DeleteConversationDialogProps {
  errorMessage: string | null;
  isPending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  open: boolean;
  title: string | null;
}

export function DeleteConversationDialog({
  errorMessage,
  isPending,
  onCancel,
  onConfirm,
  open,
  title,
}: DeleteConversationDialogProps) {
  const titleId = useId();
  const descriptionId = useId();
  const errorId = useId();
  const displayTitle = conversationTitleOrFallback(title);

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape" || isPending) {
        return;
      }

      event.preventDefault();
      onCancel();
    }

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isPending, onCancel, open]);

  if (!open) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 px-4 py-6">
      <section
        aria-describedby={descriptionId}
        aria-labelledby={titleId}
        aria-modal="true"
        className="w-full max-w-md rounded-md border border-border bg-background p-5 shadow-lg"
        role="dialog"
      >
        <h2 className="text-lg font-semibold tracking-normal text-foreground" id={titleId}>
          Delete conversation?
        </h2>
        <p className="mt-3 text-sm leading-6 text-muted-foreground" id={descriptionId}>
          Delete <span className="font-medium text-foreground">{displayTitle}</span> and its
          local memory records.
        </p>
        {errorMessage === null ? null : (
          <p className="mt-3 text-sm text-danger" id={errorId} role="alert">
            {errorMessage}
          </p>
        )}
        <div className="mt-5 flex justify-end gap-2">
          <button
            className="rounded-md border border-border bg-background px-3 py-2 text-sm font-medium text-foreground transition hover:bg-muted/40 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={isPending}
            onClick={onCancel}
            type="button"
          >
            Cancel
          </button>
          <button
            aria-describedby={errorMessage === null ? undefined : errorId}
            className="rounded-md bg-danger px-3 py-2 text-sm font-medium text-white transition hover:bg-danger/90 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={isPending}
            onClick={onConfirm}
            type="button"
          >
            Delete
          </button>
        </div>
      </section>
    </div>
  );
}
