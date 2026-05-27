import { useMemo, useState } from "react";

import { useMessageRetrievals } from "../api/useMessageRetrievals";

interface MessageRetrievalsPanelProps {
  conversationId: string;
  messageId: string;
  isOpen: boolean;
}

const CONTENT_PREVIEW_LENGTH = 220;

export function MessageRetrievalsPanel({
  conversationId,
  isOpen,
  messageId,
}: MessageRetrievalsPanelProps) {
  const [expandedContentIds, setExpandedContentIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const retrievalsQuery = useMessageRetrievals(conversationId, messageId, {
    enabled: isOpen,
  });
  const retrievals = useMemo(
    () =>
      [...(retrievalsQuery.data?.retrievals ?? [])].sort(
        (left, right) => left.rank - right.rank,
      ),
    [retrievalsQuery.data?.retrievals],
  );
  const retryRetrievals = retrievalsQuery.refetch;

  if (!isOpen) {
    return null;
  }

  if (retrievalsQuery.isLoading) {
    return (
      <section className="mt-3 border-t border-border pt-3" aria-label="Retrieved memories">
        <p className="text-sm text-muted-foreground">Loading retrieved memories...</p>
      </section>
    );
  }

  if (retrievalsQuery.isError) {
    return (
      <section className="mt-3 border-t border-border pt-3" aria-label="Retrieved memories">
        <p className="text-sm text-danger" role="alert">
          Retrieved memories could not be loaded.
        </p>
        <button
          className="mt-2 rounded-md border border-border bg-background px-3 py-1.5 text-xs font-medium text-foreground transition hover:bg-muted/40"
          onClick={() => {
            void retryRetrievals();
          }}
          type="button"
        >
          Retry
        </button>
      </section>
    );
  }

  if (retrievals.length === 0) {
    return (
      <section className="mt-3 border-t border-border pt-3" aria-label="Retrieved memories">
        <p className="text-sm text-muted-foreground">No retrieved memories recorded.</p>
      </section>
    );
  }

  return (
    <section className="mt-3 border-t border-border pt-3" aria-label="Retrieved memories">
      <ol className="space-y-3">
        {retrievals.map((retrieval) => {
          const isExpanded = expandedContentIds.has(retrieval.episode.id);
          const isLongContent = retrieval.episode.content.length > CONTENT_PREVIEW_LENGTH;
          const visibleContent =
            isLongContent && !isExpanded
              ? `${retrieval.episode.content.slice(0, CONTENT_PREVIEW_LENGTH).trimEnd()}...`
              : retrieval.episode.content;
          const sourceConversationTitle =
            retrieval.episode.source_conversation_title ?? "Untitled conversation";

          return (
            <li key={`${retrieval.rank.toString()}-${retrieval.episode.id}`}>
              <article
                aria-label={`Retrieved memory rank ${retrieval.rank.toString()}`}
                className="border-l-2 border-accent/60 pl-3"
              >
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
                  <span className="font-medium text-foreground">
                    Rank {retrieval.rank}
                  </span>
                  <span>Similarity {formatScore(retrieval.similarity)}</span>
                  <span>Score {formatScore(retrieval.score)}</span>
                  <span className="rounded border border-border px-1.5 py-0.5 uppercase tracking-normal text-foreground">
                    {retrieval.episode.kind}
                  </span>
                </div>

                <p className="mt-2 whitespace-pre-wrap break-words text-sm text-foreground">
                  {visibleContent}
                </p>
                {isLongContent ? (
                  <button
                    className="mt-1 text-xs font-medium text-accent transition hover:text-accent/80"
                    onClick={() => {
                      setExpandedContentIds((currentIds) => {
                        const nextIds = new Set(currentIds);
                        if (nextIds.has(retrieval.episode.id)) {
                          nextIds.delete(retrieval.episode.id);
                        } else {
                          nextIds.add(retrieval.episode.id);
                        }
                        return nextIds;
                      });
                    }}
                    type="button"
                  >
                    {isExpanded ? "Show less" : "Show more"}
                  </button>
                ) : null}

                <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
                  <span className="rounded border border-border px-1.5 py-0.5 text-foreground">
                    {retrieval.episode.source_scope_name} › {sourceConversationTitle}
                  </span>
                  <time dateTime={retrieval.retrieved_at}>
                    {formatRelativeTime(retrieval.retrieved_at)}
                  </time>
                  <span>{retrieval.scoring_version}</span>
                </div>

                <details className="mt-2 text-xs text-muted-foreground">
                  <summary className="cursor-pointer font-medium text-foreground">
                    Score components
                  </summary>
                  <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1">
                    <ScoreComponent label="Recency" value={retrieval.recency_score} />
                    <ScoreComponent label="Access" value={retrieval.access_score} />
                    <ScoreComponent label="Frequency" value={retrieval.frequency_score} />
                    <ScoreComponent label="Importance" value={retrieval.importance_score} />
                  </dl>
                </details>
              </article>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function ScoreComponent({ label, value }: { label: string; value: number }) {
  return (
    <div className="contents">
      <dt>{label}</dt>
      <dd className="text-foreground">{formatScore(value)}</dd>
    </div>
  );
}

function formatScore(value: number): string {
  return value.toFixed(3);
}

function formatRelativeTime(value: string): string {
  const timestamp = new Date(value).getTime();
  if (Number.isNaN(timestamp)) {
    return value;
  }

  const elapsedSeconds = Math.round((timestamp - Date.now()) / 1000);
  const absoluteSeconds = Math.abs(elapsedSeconds);
  const units = [
    { unit: "year", seconds: 365 * 24 * 60 * 60 },
    { unit: "month", seconds: 30 * 24 * 60 * 60 },
    { unit: "day", seconds: 24 * 60 * 60 },
    { unit: "hour", seconds: 60 * 60 },
    { unit: "minute", seconds: 60 },
  ] as const;
  const selectedUnit =
    units.find((candidate) => absoluteSeconds >= candidate.seconds) ?? units[4];
  const relativeValue =
    selectedUnit.unit === "minute" && absoluteSeconds < selectedUnit.seconds
      ? 0
      : Math.round(elapsedSeconds / selectedUnit.seconds);

  return new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }).format(
    relativeValue,
    selectedUnit.unit,
  );
}
