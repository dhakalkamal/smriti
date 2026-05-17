import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "../../../api/client";
import type { components } from "../../../api/types";

export type MessageResponse = components["schemas"]["MessageResponse"];

export function messagesQueryKey(conversationId: string) {
  return ["messages", "list", conversationId] as const;
}

export function useMessages(conversationId: string) {
  const encodedConversationId = encodeURIComponent(conversationId);

  return useQuery({
    queryKey: messagesQueryKey(conversationId),
    queryFn: () =>
      apiFetch<MessageResponse[]>(`/conversations/${encodedConversationId}/messages`),
  });
}
