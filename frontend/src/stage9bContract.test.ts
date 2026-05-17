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
  "EventSource",
  "localStorage",
  "sessionStorage",
  "URLSearchParams",
  "console.log",
  "console.info",
  "console.warn",
  "console.error",
  "console.debug",
] as const;

describe("Stage 9b contract", () => {
  it("does not introduce forbidden Stage 9b app-code patterns", () => {
    const matches = [
      ...findForbiddenMatches(),
      ...findDirectFetchMatches(),
      ...findMessagesSetQueryDataMatches(),
      ...findInlineSseLiteralMatches(),
    ];

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

function findDirectFetchMatches(): string[] {
  return Object.entries(sourceFiles).flatMap(([sourcePath, source]) => {
    const repoPath = toRepoPath(sourcePath);

    if (repoPath === "frontend/src/api/client.ts") {
      return [];
    }

    return source
      .split("\n")
      .flatMap((line, lineIndex) =>
        line.includes("fetch(")
          ? [`${repoPath}:${(lineIndex + 1).toString()}: fetch(: ${line.trim()}`]
          : [],
      );
  });
}

function findMessagesSetQueryDataMatches(): string[] {
  return Object.entries(sourceFiles).flatMap(([sourcePath, source]) =>
    source.split("\n").flatMap((line, lineIndex) =>
      line.includes("setQueryData") && line.includes("messages")
        ? [
            `${toRepoPath(sourcePath)}:${(lineIndex + 1).toString()}: setQueryData messages: ${line.trim()}`,
          ]
        : [],
    ),
  );
}

function findInlineSseLiteralMatches(): string[] {
  return Object.entries(sourceFiles).flatMap(([sourcePath, source]) => {
    const repoPath = toRepoPath(sourcePath);

    if (
      repoPath === "frontend/src/lib/sse.ts" ||
      repoPath === "frontend/src/features/chat/api/useAssistantResponseStream.ts"
    ) {
      return [];
    }

    return source.split("\n").flatMap((line, lineIndex) => {
      const hasSseLiteral =
        line.includes('"data:') ||
        line.includes("'data:") ||
        line.includes("`data:") ||
        line.includes('"event:') ||
        line.includes("'event:") ||
        line.includes("`event:");

      return hasSseLiteral
        ? [`${repoPath}:${(lineIndex + 1).toString()}: SSE literal: ${line.trim()}`]
        : [];
    });
  });
}

function toRepoPath(sourcePath: string): string {
  return `frontend/src/${sourcePath.replace(/^\.\//, "")}`;
}
