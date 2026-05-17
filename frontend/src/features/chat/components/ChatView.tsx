import { useAssistantResponseStream } from "../api/useAssistantResponseStream";
import { useMessages } from "../../messages/api/useMessages";
import { MessageList } from "../../messages/components/MessageList";
import { Composer } from "./Composer";

interface ChatViewProps {
  conversationId: string;
  scopeId: string;
  title: string;
}

export function ChatView({ conversationId, scopeId, title }: ChatViewProps) {
  const messagesQuery = useMessages(conversationId);
  const assistantStream = useAssistantResponseStream({ conversationId, scopeId });

  return (
    <section className="flex min-h-0 flex-1 flex-col" aria-label={title}>
      <header className="border-b border-border bg-background px-6 py-4">
        <h1 className="text-xl font-semibold tracking-normal text-foreground">{title}</h1>
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
      <div className="border-t border-border bg-muted/20 px-6 py-4">
        <article className="max-w-[78%] rounded-lg border border-border bg-background px-4 py-3 text-sm leading-6 text-foreground shadow-sm">
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
      <div className="border-t border-border bg-background px-6 py-4">
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
