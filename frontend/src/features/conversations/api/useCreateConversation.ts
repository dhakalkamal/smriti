import { useMutation, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "../../../api/client";
import type { components } from "../../../api/types";
import { conversationsQueryKey, type ConversationResponse } from "./useConversations";

export type CreateConversationBody = components["schemas"]["CreateConversationBody"];

export function useCreateConversation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: CreateConversationBody) =>
      apiFetch<ConversationResponse>("/conversations", {
        method: "POST",
        body,
      }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: conversationsQueryKey });
    },
  });
}
