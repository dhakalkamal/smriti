import { useMutation, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "../../../api/client";
import type { components } from "../../../api/types";
import { scopesQueryKey, type ScopeResponse } from "./useScopes";

export type CreateScopeBody = components["schemas"]["CreateScopeBody"];

export function useCreateScope() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: CreateScopeBody) =>
      apiFetch<ScopeResponse>("/scopes", {
        method: "POST",
        body,
      }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: scopesQueryKey });
    },
  });
}
