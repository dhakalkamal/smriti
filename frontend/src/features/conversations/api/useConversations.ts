import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "../../../api/client";
import type { components } from "../../../api/types";

export type ConversationResponse = components["schemas"]["ConversationResponse"];

export const conversationsQueryKey = ["conversations", "list"] as const;

export function useConversations() {
  return useQuery({
    queryKey: conversationsQueryKey,
    queryFn: () => apiFetch<ConversationResponse[]>("/conversations"),
  });
}
