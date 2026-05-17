import type { ScopeResponse } from "../api/useScopes";

interface ScopeSelectorProps {
  isError?: boolean;
  isLoading?: boolean;
  onSelectedScopeIdChange: (scopeId: string | null) => void;
  scopes: ScopeResponse[];
  selectedScopeId: string | null;
}

export function ScopeSelector({
  isError = false,
  isLoading = false,
  onSelectedScopeIdChange,
  scopes,
  selectedScopeId,
}: ScopeSelectorProps) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-foreground" htmlFor="scope-selector">
        Scope
      </label>
      <select
        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
        disabled={isLoading || scopes.length === 0}
        id="scope-selector"
        onChange={(event) => {
          onSelectedScopeIdChange(event.target.value === "" ? null : event.target.value);
        }}
        value={selectedScopeId ?? ""}
      >
        <option value="">{isLoading ? "Loading scopes..." : "Select a scope"}</option>
        {scopes.map((scope) => (
          <option key={scope.id} value={scope.id}>
            {scope.name}
          </option>
        ))}
      </select>
      {isError ? (
        <p className="text-sm text-danger" role="alert">
          Scopes could not be loaded.
        </p>
      ) : null}
      {!isLoading && scopes.length === 0 ? (
        <p className="text-sm text-muted-foreground">Create the first scope to begin.</p>
      ) : null}
    </div>
  );
}
