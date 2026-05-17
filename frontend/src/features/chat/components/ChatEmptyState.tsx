interface ChatEmptyStateProps {
  title: string;
  description: string;
}

export function ChatEmptyState({ description, title }: ChatEmptyStateProps) {
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center px-6 py-10">
      <div className="max-w-md border-l-4 border-accent pl-5">
        <h2 className="text-2xl font-semibold tracking-normal text-foreground">{title}</h2>
        <p className="mt-3 text-sm leading-6 text-muted-foreground">{description}</p>
      </div>
    </div>
  );
}
