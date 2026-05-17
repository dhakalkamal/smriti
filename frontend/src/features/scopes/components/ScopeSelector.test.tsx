import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ScopeResponse } from "../api/useScopes";
import { ScopeSelector } from "./ScopeSelector";

const scopes = [
  {
    id: "scope-research",
    name: "Research Notes",
    system_prompt: "",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
  {
    id: "scope-family",
    name: "Family Companion",
    system_prompt: "",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
] satisfies ScopeResponse[];

describe("ScopeSelector", () => {
  it("renders a list of scopes", () => {
    render(
      <ScopeSelector
        onSelectedScopeIdChange={vi.fn()}
        scopes={scopes}
        selectedScopeId={null}
      />,
    );

    expect(screen.getByRole("option", { name: "Research Notes" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Family Companion" })).toBeInTheDocument();
  });

  it("reports the selected scope id", () => {
    const handleChange = vi.fn();

    render(
      <ScopeSelector
        onSelectedScopeIdChange={handleChange}
        scopes={scopes}
        selectedScopeId={null}
      />,
    );

    fireEvent.change(screen.getByLabelText("Scope"), {
      target: { value: "scope-family" },
    });

    expect(handleChange).toHaveBeenCalledWith("scope-family");
  });
});
