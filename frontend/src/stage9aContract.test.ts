import { describe, expect, it } from "vitest";

const sourceFiles = import.meta.glob<string>(
  [
    "./**/*.{json,ts,tsx}",
    "!./api/openapi.json",
    "!./api/types.ts",
    "!./test/**",
    "!./**/*.test.ts",
    "!./**/*.test.tsx",
  ],
  {
    eager: true,
    import: "default",
    query: "?raw",
  },
);

const forbiddenPatterns = [
  // Stage 9b introduces this path; superseded by stage9bContract.test.ts.
  "EventSource",
  "localStorage",
  "sessionStorage",
  "URLSearchParams",
  "console.",
] as const;

describe("Stage 9a contract", () => {
  it("does not introduce forbidden Stage 9a app-code patterns", () => {
    const matches = findForbiddenMatches();

    expect(matches, matches.join("\n")).toEqual([]);
  });
});

function findForbiddenMatches(): string[] {
  return Object.entries(sourceFiles).flatMap(([sourcePath, source]) =>
    source.split("\n").flatMap((line, lineIndex) =>
      forbiddenPatterns
        .filter((pattern) => line.includes(pattern))
        .map(
          (pattern) =>
            `${toRepoPath(sourcePath)}:${(lineIndex + 1).toString()}: ${pattern}: ${line.trim()}`,
        ),
    ),
  );
}

function toRepoPath(sourcePath: string): string {
  return `frontend/src/${sourcePath.replace(/^\.\//, "")}`;
}
