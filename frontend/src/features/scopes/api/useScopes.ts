import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "../../../api/client";
import type { components } from "../../../api/types";

export type ScopeResponse = components["schemas"]["ScopeResponse"];

export const scopesQueryKey = ["scopes", "list"] as const;

export function useScopes() {
  return useQuery({
    queryKey: scopesQueryKey,
    queryFn: () => apiFetch<ScopeResponse[]>("/scopes"),
  });
}
