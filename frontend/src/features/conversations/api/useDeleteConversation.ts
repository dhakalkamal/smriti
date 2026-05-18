import { useMutation, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "../../../api/client";
import { messagesQueryKey } from "../../messages/api/useMessages";
import { conversationsQueryKey } from "./useConversations";

export function useDeleteConversation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (conversationId: string) => {
      const encodedConversationId = encodeURIComponent(conversationId);

      await apiFetch<undefined>(`/conversations/${encodedConversationId}`, {
        method: "DELETE",
      });
    },
    onSuccess: async (_data, conversationId) => {
      queryClient.removeQueries({ queryKey: messagesQueryKey(conversationId) });
      await queryClient.invalidateQueries({ queryKey: conversationsQueryKey });
    },
  });
}
