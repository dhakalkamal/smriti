# Stage 8: Frontend Foundation

This document defines the Stage 8 contract before frontend implementation begins.

Stage 8 creates the smallest local-only React foundation needed for later product UI work. It is a
tooling and contract stage, not a chat-product stage.

## 1. Purpose

Stage 8 establishes the frontend project boundary for Smriti:

- React 18 + Vite + TypeScript strict mode
- pnpm-managed frontend tooling
- Tailwind v3 design foundation
- generated OpenAPI types from the local FastAPI app
- a thin API client boundary
- TanStack Query provider setup
- local-only dev defaults and guardrails
- a minimal smoke-render path

The goal is to make later frontend stages boring in the best way: typed API access, local-only
network behavior, and validation commands are already pinned before the chat UI is built.

## 2. Scope

Stage 8 includes:

- create `frontend/` with the Vite React TypeScript scaffold
- configure pnpm scripts for frontend validation
- configure TypeScript with `"strict": true` and no relaxations
- configure Vite to bind `127.0.0.1:5173`
- default `VITE_API_BASE_URL` to `http://127.0.0.1:8000`
- configure Tailwind v3 base files and design-token conventions
- configure ESLint, Prettier, Vitest, and React Testing Library
- configure TanStack Query at the app boundary
- add `scripts/export_openapi.py` as the backend-side OpenAPI exporter
- add `pnpm generate:api` to export `frontend/src/api/openapi.json` offline and generate
  `frontend/src/api/types.ts`
- commit generated `frontend/src/api/types.ts`
- add a thin `frontend/src/api/client.ts` fetch wrapper
- add localhost-only API base URL validation
- add one minimal app shell and one frontend smoke test

Stage 8 should not build real product workflows. It should leave the application ready for Stage 9
without pre-creating feature code that Stage 9 has not contracted.

## 3. Non-Goals

Stage 8 explicitly excludes:

- chat UI
- SSE hook implementation
- assistant-response UI
- scope management UI
- routing and React Router
- Zustand or any other client-state library
- React Hook Form
- MSW
- shadcn/ui copied components
- backend route, schema, service, startup, or lifespan behavior changes
- stale generated OpenAPI type CI enforcement
- Playwright or end-to-end tests
- service workers or PWA behavior
- telemetry
- analytics
- remote assets, remote fonts, remote scripts, CDN imports, or external iframes

The only backend-side file in Stage 8 is the offline OpenAPI exporter script. It must not change
backend runtime behavior.

## 4. Locked Frontend Decisions

- Package manager: pnpm
- Framework: React 18
- Bundler/dev server: Vite
- Language: TypeScript strict mode
- Styling: Tailwind v3
- Server state: TanStack Query
- Client state: `useState` and `useReducer` only
- API types: `openapi-typescript`, generated from FastAPI OpenAPI JSON
- Tests: Vitest + React Testing Library
- Router: deferred
- shadcn/ui: zero copied components in Stage 8

Do not introduce Redux, Zustand, MobX, Recoil, Jotai, React Router, React Hook Form, MSW, MUI,
Chakra UI, Ant Design, Mantine, CSS-in-JS, socket.io, WebSocket libraries, analytics SDKs, Google
Fonts, CDN assets, or remote assets.

## 5. OpenAPI Generation Workflow

Stage 8 locks offline-capable OpenAPI generation.

The backend-side exporter lives at:

```text
scripts/export_openapi.py
```

The exporter imports the FastAPI app factory, calls `create_app().openapi()`, and writes JSON to:

```text
frontend/src/api/openapi.json
```

The exporter must not enter the FastAPI lifespan. It must not initialize the database pool, connect
to Postgres, call Ollama, bind sockets, or require any live service.

Before this contract was written, the backend was inspected and the OpenAPI path was verified with:

```bash
PYTHONPATH=src uv run python -c "from smriti.api import create_app; app = create_app(); schema = app.openapi(); print(schema['info']['title'])"
```

That command completed without starting Postgres, Ollama, or the FastAPI server. The exporter itself
uses the canonical project import path through `uv run`; it must not add `sys.path` or `PYTHONPATH`
workarounds for editable-install environment issues.

The Stage 8 frontend command should be:

```bash
pnpm generate:api
```

That command should run the exporter from the repo root and then run `openapi-typescript` against
the exported JSON:

```bash
uv run python scripts/export_openapi.py
openapi-typescript frontend/src/api/openapi.json -o frontend/src/api/types.ts
```

The pnpm script must call the exporter through `uv run python scripts/export_openapi.py`. It must
not set `PYTHONPATH`, call `.venv/bin/python`, or fetch `/openapi.json` over HTTP by default.

If `uv run python scripts/export_openapi.py` cannot import `smriti` because the local editable
install is broken, treat that as an environment/bootstrap issue under `AGENTS.md`; inspect and fix
the editable install or macOS filesystem flags before changing packaging or backend startup
boundaries.

`frontend/src/api/types.ts` is committed and never hand-edited. Stage 8 does not add CI enforcement
for stale generated types.

Fallback rule: if a future implementation discovers that `create_app().openapi()` begins requiring
Postgres, Ollama, or any live service, Stage 8 must stop and either fall back to online generation
from `http://127.0.0.1:8000/openapi.json` or defer offline generation until a backend contract
adjusts startup/import boundaries. Stage 8 must not silently change backend startup behavior to make
offline export work.

## 6. shadcn/ui Scope

Stage 8 adds zero copied shadcn/ui components.

Tailwind configuration, CSS variables, and design conventions may be prepared. Do not add `Button`,
`Card`, `Dialog`, `Input`, or any other copied shadcn component until Stage 9 or later actually
needs it.

## 7. Smoke Test Scope

The current backend GET route inventory under `src/smriti/api/` is:

- `GET /health`
- `GET /scopes`
- `GET /conversations`
- `GET /conversations/{conversation_id}/messages`

There is no root or ping endpoint. `GET /health` is the only no-side-effect health-style endpoint.
The other GET routes are domain endpoints and must not be used for Stage 8 smoke testing.

Stage 8 smoke behavior:

- If the backend is available, the app shell may perform one reachability check against
  `GET /health`.
- The frontend smoke test may cover rendering the app shell and the health-check state.
- If `GET /health` is not available in a future backend shape, Stage 8 smoke must be frontend-only
  and backend reachability is deferred.
- Do not add a backend endpoint in Stage 8.
- Do not use scopes, conversations, messages, retrieval, or assistant routes for Stage 8 smoke
  testing.

## 8. Vite Project Structure

Stage 8 should create only the structure it needs:

```text
frontend/
├── package.json
├── pnpm-lock.yaml
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── tailwind.config.ts
├── postcss.config.js
├── eslint.config.js
├── .prettierrc
├── index.html
├── public/
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── index.css
    ├── api/
    │   ├── openapi.json
    │   ├── types.ts
    │   └── client.ts
    ├── components/
    │   └── ui/
    ├── features/
    ├── hooks/
    ├── lib/
    └── pages/
```

Do not pre-create feature modules with placeholder product code. Empty structural folders are
acceptable only when required by tooling or documented repo layout.

## 9. Local Dev Workflow

Backend remains the local FastAPI service on:

```text
http://127.0.0.1:8000
```

Frontend dev server runs on:

```text
http://127.0.0.1:5173
```

Vite must bind `127.0.0.1`, never `0.0.0.0`. The default frontend API base URL is
`http://127.0.0.1:8000`. Any `VITE_API_BASE_URL` override must still be localhost.

The expected Stage 8 workflow is:

```bash
cd frontend
pnpm install
pnpm generate:api
pnpm dev
```

Stage 8 may document that the backend needs to be running only for live reachability checks. API type
generation must work offline.

## 10. API Client Boundary

`frontend/src/api/client.ts` is a thin fetch wrapper:

- owns base URL resolution
- rejects non-localhost API URLs
- sets JSON request/response behavior
- does not cache
- does not retry
- does not know domain workflows

Components must not call `fetch` directly. Feature API hooks should use the client boundary.

## 11. TanStack Query Organization

TanStack Query owns server-state caching and invalidation.

Stage 8 only wires the provider and may include a health/reachability hook if the smoke path uses
`GET /health`.

Later feature API hooks should live under:

```text
src/features/<feature>/api/
```

Query key convention:

```text
["feature", "operation", ...params]
```

Assistant response generation is not a query. The non-streaming assistant response is a mutation.
The streaming assistant response is a stream action owned by a custom hook.

## 12. SSE Integration Architecture

Stage 8 does not implement SSE hooks.

Stage 9 should implement SSE behind feature-local custom hooks, not raw stream handling in
components. The stream layer must handle only the backend's locked events:

- `start`
- `token`
- `done`
- `error`

Partial assistant text is transient UI state. It must not be inserted into the TanStack Query cache
as a persisted message. The cache may be updated only after the backend sends `done` with the
persisted assistant message, or by refetching messages after successful completion. Disconnects,
cancellations, and stream errors must clear transient state and persist nothing client-side.

## 13. Routing

Routing is deferred.

Stage 8 should use a single app shell. Stage 9 may continue with internal state. React Router should
be introduced only by a future contract that needs durable URLs, deep links, or multi-page
navigation.

## 14. Feature, Page, Component, and Hook Organization

Use the organization defined in `FRONTEND.md`:

- `pages/` for top-level screens
- `features/<feature>/` for domain UI and feature API hooks
- `components/ui/` for copied shadcn primitives, starting with none in Stage 8
- `components/` for shared non-domain components
- `hooks/` for reusable non-domain hooks
- `lib/` for utilities such as class composition and local env validation
- `api/` for generated OpenAPI artifacts and the thin client boundary

Avoid feature folders until a stage needs real feature code.

## 15. Structural Privacy Guarantees

Frontend code must preserve local-only behavior structurally:

- no non-localhost network calls
- no CDN scripts, stylesheets, fonts, or images
- no Google Fonts
- no analytics or telemetry
- no external iframes
- no tracking pixels
- no remote assets
- no secrets in frontend env vars
- no browser storage of conversation contents in Stage 8
- no service workers
- no hidden background sync
- no message content in logs

All assets must be local if assets are introduced later.

## 16. Testing Strategy

Stage 8 tests:

- app shell smoke test
- API base URL localhost guard tests
- optional health reachability UI behavior test using a fetch stub

Stage 8 validation commands:

```bash
pnpm typecheck
pnpm lint
pnpm test
pnpm build
```

Backend tests are not required for a frontend-only Stage 8 implementation, but the OpenAPI exporter
script should be importable and should not require live services.

Later stages:

- Stage 9 adds chat rendering, composer behavior, assistant mutation, and SSE parser/hook tests.
- Stage 10 adds scope selection, scope isolation UI, and mutation invalidation tests.
- Playwright remains deferred until a future contract.
- Snapshot tests remain excluded unless explicitly justified.

## 17. Contract Discipline

Frontend contracts mirror backend contract discipline:

- generated types come from backend OpenAPI, not hand-written response shapes
- frontend code may not invent backend capabilities
- mutations must respect append-only message semantics
- scope boundaries must remain visible in UI state and API calls
- assistant streaming must mirror backend persistence semantics
- implementation follows the approved stage contract
- deviations require a contract update before code changes

Stage 8 is complete when the minimal local frontend foundation exists, generated API types are
committed, the app renders a shell locally, localhost guardrails are in place, and all frontend
validation commands pass.

Additional exit criteria:

- `uv run python scripts/export_openapi.py --output frontend/src/api/openapi.json` runs
  successfully in the local environment and produces a valid JSON file.
- `pnpm generate:api` runs the full export-and-generate workflow successfully and produces:
  - `frontend/src/api/openapi.json`
  - `frontend/src/api/types.ts`
- `frontend/src/api/types.ts` is committed.
