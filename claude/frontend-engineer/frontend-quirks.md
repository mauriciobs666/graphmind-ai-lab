# Front-end framework/tooling quirks — this lab's React/TS storefront stack

> **Live-verified knowledge base for `frontend-engineer`.** Facts confirmed by hands-on work
> against this lab's actual stack — React 18 + TypeScript (`tsc -b`, strict-ish `tsconfig`) +
> Vite + TanStack Query v5 (`@tanstack/react-query`) + `react-i18next`/`i18next` + Tailwind —
> built while implementing `salesperson/` (the storefront SPA, `docs/plans/salesperson-ui.md`).
> Most are generic library/language facts, not project-specific — they'll bite again the next
> time this stack (or a piece of it) is used anywhere in the lab. `frontend-engineer.md` (the
> always-on prompt) points here and stays lean; this file holds the perishable, growing fact
> list. **Re-verify an entry before relying on it if the underlying library's major version has
> since changed.** Each entry tied to a specific npm package names the pinned version
> (`salesperson/package.json`) it was verified against; the two entries that aren't
> library-version-specific — a browser-engine layout behavior, and a project testing technique —
> say so instead rather than carry a fabricated version.

---

## TypeScript & build tooling

- **`UseQueryResult`/`UseMutationResult` (TanStack Query v5) are discriminated unions, not plain
  object types.** `interface X extends UseQueryResult<T,E> { extra: … }` fails `tsc` with
  **TS2312** ("can only extend an object type or intersection of object types with statically
  known members"). Use a type alias with an intersection instead:
  `type X = UseQueryResult<T,E> & { extra: … }`. Verified against `@tanstack/react-query@^5.102.8`
  and `typescript@~6.0.2` in `salesperson/src/api/hooks.ts` (`UseShopStateResult`,
  `UseMessagesResult`, `UsePostMessageResult`, and 7 more) — 10 interfaces converted, confirmed
  by a clean `tsc -b`.

- **A Vite/TS scaffold has no `resolveJsonModule` by default.** `import en from "./en.json"`
  type-checks fine under Vite/esbuild at *runtime* (bundler resolution doesn't care), but fails
  `tsc -b` with **TS2307** ("cannot find module") until `"resolveJsonModule": true` is added to
  `tsconfig.app.json`'s `compilerOptions`. Verified against `typescript@~6.0.2` and `vite@^8.2.2`.
  Already set in `salesperson/tsconfig.app.json` (added for the `src/locales/{en,pt-BR,es}.json`
  i18n bundles) — if a *new* Vite/TS component is scaffolded elsewhere in the lab and needs to
  import JSON directly, expect to hit this and add the flag again; nothing about the default
  scaffold carries it forward.

## TanStack Query v5 — testing

- **`invalidateQueries()` only triggers a refetch for a query with an active observer (a mounted
  `useQuery`).** An invalidated query with no mounted observer is just marked stale — no network
  call fires. A hook-level test asserting a re-read/refetch after a mutation's `onError`/`onSuccess`
  calls `invalidateQueries()` must **also mount the query hook being invalidated** in the same test
  component, not just the mutation hook — otherwise the assertion never sees the follow-up
  request and either false-passes (asserting absence) or fails confusingly (asserting presence).
  Verified against `@tanstack/react-query@^5.102.8` in `salesperson/src/api/hooks.test.tsx`'s C4
  network-effect tests (`usePresenterResetAll`/`useResetMine`/`useAdvanceOrder`).

## Testing — React Testing Library / Vitest

- **A heading and a button/CTA inside it sharing identical `t()`-translated copy makes
  `screen.getByText`/`getByRole` ambiguous** (multiple-match failure) in RTL tests. Split heading
  vs. action copy into distinct strings before writing assertions — e.g.
  `ResetControl.tsx`'s heading "Session controls" vs. its button "Reset my session", not both
  "Reset everyone". Verified against `@testing-library/react@^16.3.3`. Bit
  `salesperson/src/views/PresenterResetAllControl.tsx` when both were initially localized to
  "Reset everyone"/"Redefinir todos"; fixed by renaming the heading key.

- **A "seed a no-op placeholder into a sibling step's file, hand off ownership" pattern** (used to
  let a downstream implementation step land with zero file-ownership conflicts —
  `docs/plans/salesperson-ui.md` §5.0's seed rows) **is safe for the seeding step's *files*, but
  not automatically for its *tests*.** The seeding step's own tests can go red once the real
  content lands, if they assert on the placeholder's **literal text** (`getByText('The catalog
  will appear here.')`) or on a **global, URL-unscoped mock call count/absence**
  (`expect(fetchMock).not.toHaveBeenCalled()` / `toHaveBeenCalledTimes(1)` with no URL filter) —
  because the newly-real sibling content now mounts its own real data-fetching hooks the
  placeholder never had, and a global assertion has no way to distinguish "my endpoint fired" from
  "some other panel's endpoint fired." **Write the seeding step's tests to assert against the
  panel's own real, production output** (e.g. its actual "Loading …" copy) or **scope any mock
  assertion to the specific URL/endpoint under test**, from the start — don't wait for the
  downstream step to expose the gap. A project testing technique, not tied to any library
  version. Verified twice on this exact pattern in `salesperson/`:
  S12b's four seed placeholders (`views/{Cart,Order,Profile,Catalog}Panel.tsx`) broke
  `layout/Shell.test.tsx` (literal placeholder text) and 8/9 tests in
  `components/sheets/ResetControl.test.tsx` (global `fetchMock` call-count) once S14 replaced them
  with real panels — fixed in place per `falkor-chat/docs/HISTORY.md`'s 2026-09-14 entry
  ("fix cross-cutting test breakage S14's landing exposed").

## CSS / layout

- **In a `flex flex-col` ancestor chain, a `display:block` intermediate `div` sized via
  `flex: 1 1 0%` (Tailwind `flex-1`) does NOT give its own percentage-height children (`h-full`)
  a definite size in Chromium** — the child silently collapses to its content height instead of
  the parent's real (flex-grown) pixel height, even though the intermediate div's own
  `getBoundingClientRect()` reports the correct larger size. Fix: size the child with an explicit
  computed height (`h-[calc(100dvh-<known-fixed-sibling-height>)]`) instead of a percentage. A
  Chromium layout-engine behavior, not an npm-versioned library fact — no build/version was
  recorded at verification time. Verified via a live Playwright probe (`getBoundingClientRect` +
  `getComputedStyle`) against
  `salesperson/src/layout/Shell.tsx`'s `<div className="flex-1"><Outlet/></div>` — a child using
  `h-full` measured 177px against the wrapper's actual 787px.

## i18next

- **i18next instance methods (`i18n.t`, `i18n.changeLanguage`) are not pre-bound.** Extracting
  `const t = i18n.t` and calling it standalone throws, because the method reads `this.translator`
  internally (`node_modules/i18next/dist/cjs/i18next.js`: `t(...args) { return
  this.translator?.translate(...args); }` — an unbound prototype method, no arrow-fn/constructor
  bind). Verified against `i18next@^26.4.1`/`react-i18next@^17.0.13`. Either call as `i18n.t(...)`
  directly, use `i18n.t.bind(i18n)`, or — the pattern this
  codebase uses — **thread an already-bound `TFunction`** (from `useTranslation()`, which does
  return a bound `t`) into any plain non-React function that needs to translate outside the React
  tree (`salesperson/src/components/message/composerNotice.ts`'s `composerNoticeFor(..., t:
  TFunction)`, kept plain for cheap mutation testing). Verified via
  `salesperson/src/components/message/composerNotice.test.ts`.
