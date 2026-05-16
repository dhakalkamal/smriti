# FRONTEND.md

Frontend conventions for the Smriti React application under `frontend/`.

## Read AGENTS.md first

This document specializes the project-wide rules in `AGENTS.md` for
frontend work. It does not replace them.

The privacy invariants, localhost-only architecture, "no remote
assets", "no telemetry", "no CDN imports", "no analytics SDKs", and
contract-first stage discipline in `AGENTS.md` apply equally to
frontend code.

If anything in this document appears to relax an `AGENTS.md` rule,
`AGENTS.md` wins. When in doubt, the more restrictive rule wins.

## Tooling

Locked decisions for the frontend stack:

- Package manager: pnpm
- Bundler and dev server: Vite
- Language: TypeScript with `"strict": true`, no relaxations
- UI framework: React 18
- Styling: Tailwind v3
- UI components: shadcn/ui, components copied into the codebase,
  not imported as a runtime component library
- Server state: TanStack Query (`@tanstack/react-query`)
- Client state: `useState` and `useReducer` only. Introducing
  Zustand (or any other client-state library) requires a
  contract update.
- Forms: controlled components first. React Hook Form may be
  introduced by contract update when forms become complex.
- API types: `openapi-typescript`, generated from the backend's
  OpenAPI schema, types-only, no runtime
- Tests: Vitest + React Testing Library
- API mocking in tests: simple mocked hooks or fetch stubs in
  early stages. MSW is the preferred tool for non-trivial
  API-interaction tests once they emerge; introduce by contract
  update.
- E2E (deferred): Playwright
- Lint: ESLint with TypeScript and React plugins
- Format: Prettier

### Do not introduce

- Redux, MobX, Recoil, Jotai
- Zustand without a contract update
- Material UI, Chakra UI, Ant Design, Mantine
- Styled Components, Emotion, or any CSS-in-JS library
- socket.io or any WebSocket library
- React Hook Form without a contract update
- MSW without a contract update
- Any analytics SDK
- Any CDN-loaded font, script, image, or stylesheet
- Google Fonts
- React Router beyond what a future stage explicitly introduces
- Snapshot-testing patterns (`expect(...).toMatchSnapshot()`)

If a use case appears to require something on this list, propose
a contract update first. Do not silently introduce one of these.

## Repository layout

The frontend lives in `frontend/` at the repository root:

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

Rules:

- Frontend tests live alongside the code they test:
  `*.test.ts` or `*.test.tsx`
- No separate `frontend/tests/` directory
- `src/api/types.ts` is committed, not gitignored

## Code style

### TypeScript

- `strict: true`, no relaxations
- No `any`. If `unknown` is insufficient, explain why.
- Prefer `type` for unions and aliases
- Prefer `interface` for extensible object shapes
- Use generated API types directly from `src/api/types.ts`
- Do not redeclare backend response shapes
- Use discriminated unions for state machines:

```ts
type RequestState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: Error };
```

### React

- Functional components only
- Hooks for state and effects
- Shared logic goes into custom hooks under `src/hooks/`
- `PascalCase` for components
- `camelCase` for hooks/utilities
- One component per file unless tightly coupled
- Props typed as `<ComponentName>Props`
- Default exports for page-level components
- Named exports for everything else
- Avoid `useEffect` for derived state

### Styling

- Tailwind utility classes first
- No feature-level CSS files unless explicitly justified
- `index.css` holds Tailwind directives and CSS variables
- shadcn components for primitives
- Design tokens configured in `tailwind.config.ts`
- Use a `cn()` helper for class composition
- No CSS-in-JS
- Avoid inline `style={{}}`

### Server state

- All server state goes through TanStack Query
- Manual `fetch` calls in components are forbidden
- Wrap API access in hooks under:
  `src/features/<feature>/api/`
- Query key convention:
  `["feature", "operation", ...params]`
- Mutations use `useMutation`
- SSE consumption goes through custom hooks, not raw
  `EventSource` in components

### Client state

- Local state: `useState` / `useReducer`
- Shared server-derived state stays in TanStack Query
- Shared client-only UI state may use lifted state or
  React context
- Zustand requires a contract update

### Accessibility

- Interactive elements must be keyboard navigable
- Form controls must have labels
- Prefer semantic HTML before ARIA roles
- Color is never the only signal
- Include `eslint-plugin-jsx-a11y` in frontend linting

## Frontend-backend contract

- Backend OpenAPI schema is the source of truth
- Regenerate types via `pnpm generate:api`
- Never hand-edit `src/api/types.ts`
- `src/api/client.ts` is a thin fetch wrapper
- Default API base URL:
  `http://127.0.0.1:8000`
- No retries or caching inside the fetch wrapper
- TanStack Query owns caching behavior
- SSE uses:
  `text/event-stream`
  with `start`, `token`, `done`, `error`

## Testing requirements

- Every page has at least one smoke test
- Components with logic have behavior tests
- Hooks with non-trivial logic have tests
- API mocking starts simple
- Introduce MSW later if needed
- Tests run via `pnpm test`
- `pnpm test --coverage` available
- No snapshot tests without explicit justification

## Environment configuration

- Vite binds `127.0.0.1:5173`
- Never bind `0.0.0.0`
- `VITE_API_BASE_URL` defaults to
  `http://127.0.0.1:8000`
- Frontend env vars must start with `VITE_`
- No frontend env var may contain secrets

## Localhost-only enforcement

Frontend code must never communicate with non-localhost
services.

Specifically:

- No external `fetch`
- No CDN `<script>`
- No CDN `<link>`
- No tracking pixels
- No external `<iframe>`

If an asset is needed, vendor it locally under `public/`.

## Stage discipline

Frontend stages follow the same contract-first workflow as
backend stages.

- Each frontend stage gets a contract document
- Contracts pin locked decisions before implementation
- Implementation is proposed first, approved second,
  executed third
- Agents must not silently deviate from locked decisions
- Validation after implementation:
  `pnpm typecheck`
  `pnpm lint`
  `pnpm test`
  `pnpm build`

## Commit conventions

- Frontend commits must pass:
  - `pnpm typecheck`
  - `pnpm lint`
  - `pnpm test`
- Frontend-only commits do not require backend tests
- Backend-only commits do not require frontend tests
- Cross-stack commits must pass both suites

## Not specified yet

Deferred to later stage contracts:

- Internationalization
- Authentication UI
- Advanced theming
- Animation framework
- Error boundary granularity
- Service workers / PWA
- Bundle-size budgets

Do not invent conventions for these without a contract update.