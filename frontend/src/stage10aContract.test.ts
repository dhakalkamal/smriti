import { describe, expect, it } from "vitest";

const sourceFiles = import.meta.glob<string>(
  [
    "./**/*.{css,json,ts,tsx}",
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

interface LinePattern {
  label: string;
  pattern: RegExp;
}

interface ContractMatch {
  label: string;
  line: string;
  lineNumber: number;
  repoPath: string;
  subsection: "§15.1" | "§15.2" | "§15.3";
}

const stage15Point1ForbiddenPatterns = [
  { label: "localStorage", pattern: /\blocalStorage\b/u },
  { label: "sessionStorage", pattern: /\bsessionStorage\b/u },
  { label: "URLSearchParams", pattern: /\bURLSearchParams\b/u },
  { label: "deleted_at", pattern: /\bdeleted_at\b/u },
  { label: "is_deleted", pattern: /\bis_deleted\b/u },
  { label: "archived_at", pattern: /\barchived_at\b/u },
] as const satisfies readonly LinePattern[];

const stage15Point2ForbiddenPatterns = [
  { label: "console.log", pattern: /\bconsole\.log\b/u },
  { label: "console.info", pattern: /\bconsole\.info\b/u },
  { label: "console.warn", pattern: /\bconsole\.warn\b/u },
  { label: "console.error", pattern: /\bconsole\.error\b/u },
  { label: "console.debug", pattern: /\bconsole\.debug\b/u },
] as const satisfies readonly LinePattern[];

const deleteDialogGuardPatterns = [
  { label: "MessageResponse", pattern: /\bMessageResponse\b/u },
  {
    label: "message content field",
    pattern:
      /(?:\.\s*content\b|\[\s*["']content["']\s*\]|\bcontent\??\s*:|\bcontent\s*[;=]|\{\s*content\s*(?:[,}]))/u,
  },
  {
    label: "messages query key",
    pattern: /\bmessagesQueryKey\b|\[\s*["']messages["']\s*,\s*["']list["']/u,
  },
] as const satisfies readonly LinePattern[];

describe("Stage 10a contract", () => {
  it("does not introduce forbidden Stage 10a app-code patterns", () => {
    const matches = [
      ...findStage15Point1Matches(),
      ...findStage15Point2Matches(),
      ...findDeleteDialogGuardMatches(),
    ].map(formatMatch);

    expect(matches, matches.join("\n")).toEqual([]);
  });
});

function findStage15Point1Matches(): ContractMatch[] {
  return Object.entries(sourceFiles).flatMap(([sourcePath, source]) => {
    const repoPath = toRepoPath(sourcePath);
    const patternMatches = findLineMatches(
      repoPath,
      source,
      "§15.1",
      stage15Point1ForbiddenPatterns,
    );

    if (repoPath === "frontend/src/api/client.ts") {
      return patternMatches;
    }

    return [
      ...patternMatches,
      ...findLineMatches(repoPath, source, "§15.1", [
        { label: "direct fetch(", pattern: /\bfetch\s*\(/u },
      ]),
    ];
  });
}

function findStage15Point2Matches(): ContractMatch[] {
  return Object.entries(sourceFiles).flatMap(([sourcePath, source]) => {
    const repoPath = toRepoPath(sourcePath);

    if (!isDeletionOrChatCodePath(repoPath)) {
      return [];
    }

    return findLineMatches(repoPath, source, "§15.2", stage15Point2ForbiddenPatterns);
  });
}

function findDeleteDialogGuardMatches(): ContractMatch[] {
  return findDeleteDialogGuardSourcePaths().flatMap((sourcePath) =>
    findLineMatches(
      toRepoPath(sourcePath),
      sourceFiles[sourcePath],
      "§15.3",
      deleteDialogGuardPatterns,
    ),
  );
}

function findDeleteDialogGuardSourcePaths(): string[] {
  const guardSourcePaths = new Set(
    Object.keys(sourceFiles).filter(isDeleteConversationTsxSourcePath),
  );

  for (const sourcePath of Array.from(guardSourcePaths)) {
    const source = sourceFiles[sourcePath];

    for (const importSpecifier of findImportSpecifiers(source)) {
      const importedSourcePath = resolveRelativeImport(sourcePath, importSpecifier);

      if (
        importedSourcePath !== null &&
        toRepoPath(importedSourcePath).startsWith(
          "frontend/src/features/conversations/components/",
        )
      ) {
        guardSourcePaths.add(importedSourcePath);
      }
    }
  }

  return Array.from(guardSourcePaths).sort();
}

function findLineMatches(
  repoPath: string,
  source: string,
  subsection: ContractMatch["subsection"],
  patterns: readonly LinePattern[],
): ContractMatch[] {
  return source.split("\n").flatMap((line, lineIndex) =>
    patterns
      .filter(({ pattern }) => pattern.test(line))
      .map(({ label }) => ({
        label,
        line,
        lineNumber: lineIndex + 1,
        repoPath,
        subsection,
      })),
  );
}

function findImportSpecifiers(source: string): string[] {
  const fromImports = Array.from(
    source.matchAll(/\b(?:import|export)\s+(?:type\s+)?[^"']*?\s+from\s+["']([^"']+)["']/gu),
    (match) => match[1],
  );
  const sideEffectImports = Array.from(
    source.matchAll(/\bimport\s+["']([^"']+)["']/gu),
    (match) => match[1],
  );

  return [...fromImports, ...sideEffectImports];
}

function resolveRelativeImport(sourcePath: string, importSpecifier: string): string | null {
  if (!importSpecifier.startsWith(".")) {
    return null;
  }

  const sourceDirectory = sourcePath.replace(/\/[^/]+$/u, "");
  const normalizedPath = normalizeSourcePath(`${sourceDirectory}/${importSpecifier}`);
  const candidates = [
    `${normalizedPath}.tsx`,
    `${normalizedPath}.ts`,
    `${normalizedPath}.json`,
    `${normalizedPath}/index.tsx`,
    `${normalizedPath}/index.ts`,
  ];

  return candidates.find((candidate) => candidate in sourceFiles) ?? null;
}

function normalizeSourcePath(path: string): string {
  const normalizedParts: string[] = [];

  for (const part of path.replace(/^\.\//u, "").split("/")) {
    if (part === "" || part === ".") {
      continue;
    }

    if (part === "..") {
      normalizedParts.pop();
      continue;
    }

    normalizedParts.push(part);
  }

  return `./${normalizedParts.join("/")}`;
}

function isDeletionOrChatCodePath(repoPath: string): boolean {
  return (
    repoPath.startsWith("frontend/src/features/conversations/") ||
    repoPath.startsWith("frontend/src/features/chat/") ||
    repoPath.startsWith("frontend/src/features/messages/") ||
    repoPath === "frontend/src/pages/ChatPage.tsx"
  );
}

function isDeleteConversationTsxSourcePath(sourcePath: string): boolean {
  const repoPath = toRepoPath(sourcePath);
  const fileName = repoPath.split("/").at(-1) ?? "";

  return (
    repoPath.startsWith("frontend/src/features/conversations/") &&
    fileName.includes("DeleteConversation") &&
    fileName.endsWith(".tsx")
  );
}

function formatMatch(match: ContractMatch): string {
  return `${match.repoPath}:${match.lineNumber.toString()}: ${match.subsection} ${match.label}: ${match.line.trim()}`;
}

function toRepoPath(sourcePath: string): string {
  return `frontend/src/${sourcePath.replace(/^\.\//u, "")}`;
}
