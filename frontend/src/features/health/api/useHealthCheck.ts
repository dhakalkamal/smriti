import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "../../../api/client";
import type { paths } from "../../../api/types";

export type HealthResponse =
  paths["/health"]["get"]["responses"][200]["content"]["application/json"];

export function useHealthCheck() {
  return useQuery({
    queryKey: ["health", "check"],
    queryFn: () => apiFetch<HealthResponse>("/health"),
    staleTime: 30_000,
  });
}
