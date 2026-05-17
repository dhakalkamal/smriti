import { useMutation } from "@tanstack/react-query";

import { apiFetch } from "../../../api/client";
import type { components } from "../../../api/types";

export type CreateMessageBody = components["schemas"]["CreateMessageBody"];
export type CreatedMessageResponse = components["schemas"]["CreatedMessageResponse"];

export function estimateUserMessageTokenCount(content: string): number {
  return Math.max(1, Math.ceil(content.length / 4));
}

export function useCreateMessage(conversationId: string) {
  const encodedConversationId = encodeURIComponent(conversationId);

  return useMutation({
    mutationFn: (body: CreateMessageBody) =>
      apiFetch<CreatedMessageResponse>(`/conversations/${encodedConversationId}/messages`, {
        method: "POST",
        body,
      }),
  });
}
