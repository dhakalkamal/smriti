import { QueryClientProvider, type QueryClient } from "@tanstack/react-query";
import { render, type RenderOptions } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";

import { createQueryClient } from "../lib/queryClient";

interface RenderWithQueryClientResult {
  queryClient: QueryClient;
  renderResult: ReturnType<typeof render>;
}

export function renderWithQueryClient(
  ui: ReactElement,
  options?: RenderOptions,
): RenderWithQueryClientResult {
  const queryClient = createQueryClient();

  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }

  return {
    queryClient,
    renderResult: render(ui, { wrapper: Wrapper, ...options }),
  };
}
