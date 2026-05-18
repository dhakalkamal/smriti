export const UNTITLED_CONVERSATION_TITLE = "Untitled conversation";

export function conversationTitleOrFallback(title: string | null | undefined): string {
  const trimmedTitle = title?.trim();

  return trimmedTitle === undefined || trimmedTitle === ""
    ? UNTITLED_CONVERSATION_TITLE
    : trimmedTitle;
}
