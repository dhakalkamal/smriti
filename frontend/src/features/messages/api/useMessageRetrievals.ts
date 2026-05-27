import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "../../../api/client";
import type { components } from "../../../api/types";

export type MessageRetrievalsResponse =
  components["schemas"]["MessageRetrievalsResponse"];

interface UseMessageRetrievalsOptions {
  enabled?: boolean;
}

export function messageRetrievalsQueryKey(conversationId: string, messageId: string) {
  return ["messageRetrievals", "list", conversationId, messageId] as const;
}

export function useMessageRetrievals(
  conversationId: string,
  messageId: string,
  options: UseMessageRetrievalsOptions = {},
) {
  const encodedConversationId = encodeURIComponent(conversationId);
  const encodedMessageId = encodeURIComponent(messageId);

  return useQuery({
    queryKey: messageRetrievalsQueryKey(conversationId, messageId),
    queryFn: () =>
      apiFetch<MessageRetrievalsResponse>(
        `/conversations/${encodedConversationId}/messages/${encodedMessageId}/retrievals`,
      ),
    enabled: options.enabled ?? true,
    staleTime: Infinity,
  });
}
