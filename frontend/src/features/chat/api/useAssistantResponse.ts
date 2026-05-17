import { useMutation } from "@tanstack/react-query";

import { apiFetch } from "../../../api/client";
import type { components } from "../../../api/types";

export type AssistantGenerationResponse = components["schemas"]["AssistantGenerationResponse"];
type CreateAssistantResponseBody = components["schemas"]["CreateAssistantResponseBody"];

export type CreateAssistantResponseRequest = Pick<
  CreateAssistantResponseBody,
  "query_message_id" | "scope_id"
>;

export function useAssistantResponse(conversationId: string) {
  const encodedConversationId = encodeURIComponent(conversationId);

  return useMutation({
    mutationFn: (body: CreateAssistantResponseRequest) =>
      apiFetch<AssistantGenerationResponse>(
        `/conversations/${encodedConversationId}/assistant-response`,
        {
          method: "POST",
          body,
        },
      ),
  });
}
