# S12b — Mobile layout shell + AC-5 reset control (`salesperson/src/layout/**`, `salesperson/src/components/sheets/**`, `salesperson/tests/e2e/**`, `salesperson/playwright.config.ts`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** S12b (salesperson-ui)

## Scope & verdict

Reviewed: `salesperson/src/layout/{Shell,Header,icons,SheetContext,sessionBridge,injectBridge}.tsx`
(+ tests), `salesperson/src/components/sheets/{BottomSheet,CartSheet,OrderSheet,CatalogSheet,
ProfileSheet,ResetControl}.tsx` (+ tests), the four `salesperson/src/views/{Cart,Order,Profile,
Catalog}Panel.tsx` seed placeholders, `salesperson/playwright.config.ts`,
`salesperson/tests/e2e/mobile-shell.spec.ts` — against `docs/plans/salesperson-ui.md` (v1.34)
§5.0's S12b row, §5.1's S12b row and done-condition, AC-4, AC-5, and §4.8/§5.3 C7 (the
reset-then-language-step contract `ResetControl` implements). `falkor-chat/docs/reviews/
salesperson-ui-s12a.md` read for house style and, more substantively, for the exact shape of the
S12a-delivered `useResetMine()`/`resolveErrorAction` machinery this unit sits next to.
`salesperson/AGENTS.md` read for file ownership (its `routes.tsx` row changed under me mid-review
to a v1.35 shape unrelated to S12b's own row — noted, not otherwise investigated, out of scope).

Independently re-verified rather than taken on report: full `npx vitest run` (**117/117 green**,
10 files, matching `teco`'s figure exactly, toolchain provisioned per `salesperson/AGENTS.md`'s
`~/.local/node/current/bin` prepend). Did not re-run `tsc -b`/`build`/Playwright — `teco`'s report
already reproduced those and I found no reason to doubt them. Read `App.tsx`, `layout/Shell.tsx`,
`layout/sessionBridge.ts`, `layout/injectBridge.tsx`, `session/SessionContext.tsx`, `api/dispatch.ts`
and the relevant slice of `api/hooks.ts` in full, tracing the actual render tree and the actual
control flow rather than the delegate's or brief's paraphrase of either.

**Verdict: needs changes.** One blocker: `ResetControl`'s own error handling silently narrows
§5.3's C1–C14 contract for `POST /shop/api/reset` down to "503 gets a named message, everything
else gets one generic message" — a real behavioral regression against a plan clause that
explicitly presupposes full compliance ("the client-side rule is §5.3 C7, **built by S12a — S12b
renders it**"), and avoidable without touching `App.tsx` because the classifier it skips
(`resolveErrorAction`) is a plain function. One major on the architectural question the brief
centers on: the `sessionBridge`/`spliceIntoGrandchild` workaround is sound and well-tested for
what it does today, but the root cause — `LayoutShell` sitting outside **all three** of
`SessionContext`, TanStack Query and the router, not just the one the brief names for navigation —
will hit S14's cart/order/catalog panels on essentially every hook they need, and should be fixed
in `App.tsx` before S14 is dispatched, not discovered mid-unit. Two minors round it out. The mobile
chrome, the sheet/`BottomSheet` accessibility contract, the seed placeholders and the Playwright
suite are otherwise well-built and genuinely tested.

**CPG:** considered, not relevant — `salesperson/` has no loaded CPG (`cpg_salesperson` does not
exist among this instance's graphs), and the ~15 newly-authored files under review are small
enough that direct reading is faster and more precise than building one would have been.

## Findings

### Blocker — `ResetControl`'s error handling drops C3/C4/C6b/C13 for `POST /shop/api/reset`, reachable without touching `App.tsx`

`ResetControl.tsx:50-57`'s `catch` block has exactly two branches: `err.status === 503` gets
"nothing was reset, try again"; **every other status — 401, 504, 409, anything else — gets the
same generic "Could not reset your session. Please try again."**, with the UI simply returning to
the confirm step. Compare `api/hooks.ts:317-324`'s `useResetMine()`, which the plan explicitly
frames this row around ("the client-side rule is §5.3 C7, built by S12a — S12b renders it"): it
runs every error through `dispatch('resetMine', error, 'user')` → `resolveErrorAction()`
(`api/dispatch.ts`), which — for this exact route — additionally has to handle a 401 (**C3**:
clear the participant credential, return to join), a 504 (**C4**: re-read `/state` rather than
report "nothing changed" — this is §4.8's F8, the multi-revision timeout-ambiguity contract), a
409 `unscoped_participant` (**C6b**: a distinct alarm, not busy and not success) and anything
unruled (**C13**: loud, not swallowed). `ResetControl` implements none of these four for its own
call site; only C9's 503 case is hand-rolled inline.

This is not merely undertested — it is **wrong behavior**, traced end-to-end rather than inferred:
on a 401 (plausible in practice — a presenter `reset-all` racing the participant's own reset
invalidates every participant token per §4.8's "reset everyone" row), the participant is left
stuck on a confirm dialog that will keep failing the same way, when the spec says they should be
bounced to join. On a 504 (the scenario §4.8's F8/C4 machinery exists specifically to get right),
a reset that may well have **committed** server-side is reported as a plain failure with no
re-read, inviting a redundant second reset rather than the "unknown, go check" treatment the rest
of the client gives every other writing route.

**Why this was avoidable without an `App.tsx` edit:** `resolveErrorAction` (`api/dispatch.ts`) is
a pure function with no React dependency — it does not require `useResetMine()`'s hook wrapper,
only `useErrorEffects`'s *side effects* do (context clear + `useNavigate`). Of those,
`clearParticipant`'s effect is itself just `session/storage.ts`'s plain `clearParticipantSession()`
(already imported one line below `loadParticipantSession`/`participantAuthHeader` in
`ResetControl.tsx:22`) plus `router.navigate(APP_PATHS.participant, { replace: true })` — the same
imperative call `ResetControl` already uses on success. C4's exact mechanism (`queryClient.
invalidateQueries`) *is* blocked by the same provider-unreachability the major below covers, but
C3's clear+redirect and C6b's distinct copy are not.

**Fix:** import `resolveErrorAction` in `ResetControl.tsx`, call it as `resolveErrorAction
('resetMine', err, 'user')` in the `catch` block, and switch on `.kind`: `clearParticipant` →
`clearParticipantSession()` + `router.navigate(APP_PATHS.join or wherever join lives, { replace:
true })`; `reread`/`nothingChangedRetry` (503/504) → keep the confirm step, but distinguish the
504 copy from the 503 copy (e.g. "we couldn't confirm whether that worked — check your cart/order
before trying again") rather than the current identical-looking generic message; `unscopedAlarm`
→ its own copy, no implied retry; `unhandled` → a visible "unexpected response" message, never the
silent generic fallback. Add a `ResetControl.test.tsx` case per branch mirroring the existing 503
test's shape (mock `fetch` to return each status/body, assert the resulting UI state/credential),
which is also what proves the fix rather than merely asserting the delegate's intent.

### Major — the root cause (`LayoutShell` outside `SessionContext`/TanStack Query/the router) reaches beyond navigation and will hit essentially every S14 hook

The brief's framing names three unreachable families — `useSession()`, `useQuery()`/
`useMutation()`, `useNavigate()`/`useLocation()` — and reading `App.tsx`/`layout/Shell.tsx`
confirms the render tree makes all three genuinely siblings of `LayoutShell`'s `{children}`, not
descendants: `<LayoutShell>` wraps `<QueryClientProvider><SessionProvider><RouterProvider/>
</SessionProvider></QueryClientProvider>` as `children`, and `Shell.tsx` renders `<Header/>` and
all four `*Sheet` components as *siblings* of `{withBridge}` (the wrapped `children`), not nested
inside it. So it isn't only `useResetMine()`'s navigation half that's unreachable from
`components/sheets/**` — `useShopState()`, `useCatalog()`, `useAdvanceOrder()` (all `useQuery`/
`useMutation` under TanStack Query) are equally unreachable, because `QueryClientProvider` itself
sits inside the same excluded subtree.

§5.0's v1.34 seed row commits S14's four panels to mount **inside these same sheets**
(`components/sheets/{Cart,Order,Catalog,Profile}Sheet.tsx`), and §2.4's parity table (cart running
total, order status chip, catalog grid, profile card) is unbuildable from static placeholder
content — every one of those needs `useShopState()`/`useCatalog()` at minimum, and the order
panel's `cancel`/`fulfill`/`deliver` actions need `useAdvanceOrder()`, which itself needs
`useNavigate` transitively (via `useErrorEffects`) for its own error dispatch. That is at least
three more bespoke bridges in `S14`'s shape (one for `useSession`, already half-built; one for
`QueryClient` reads/mutations; one for navigation), each isolated and no-op-degrading like this
one, or S14 cannot deliver its scope without an unauthorized `App.tsx` edit.

The delegate names the clean fix correctly and defers it appropriately (reorder `App.tsx` so the
three providers wrap `<I18nProvider>`/`<LayoutShell>` rather than the reverse, mirroring the
`routes.tsx` v1.34 precedent) and does not touch `App.tsx`, which is the right call for this unit.
**Call:** this is not a defect in what S12b built — the bridge is a contained, tested, gracefully-
degrading interim for the one place it was actually needed this unit — but the root cause should
not wait for S14 to rediscover it three more times. Recommend `teco` route a plan amendment to
`architect` **before S14 is dispatched**, weighing the reorder against accepting the bridge
pattern as precedent (the bridge pattern's cost compounds — a `QueryClient` bridge is materially
harder to make safe than a single setter, since query results are keyed/cached data rather than
one primitive value) — same shape as the `routes.tsx` ownership gap `architect` closed as plan
v1.34 off `salesperson-ui-s12a.md`'s major.

### Minor — `BottomSheet`'s focus-on-open is implemented but not asserted by any test

`BottomSheet.tsx:30` calls `panelRef.current?.focus()` synchronously when `open` flips true, and
the panel carries `tabIndex={-1}` to make it focusable — a real implementation, not just markup.
But `BottomSheet.test.tsx`'s four cases cover role/name, the close button, the backdrop and
Escape; none asserts `document.activeElement` after open. Given the accessibility contract is
explicitly called out in this review's brief and the other three legs (role/aria-modal, Escape,
backdrop) are each tested, the fourth leg is the one gap. **Fix:** add `expect(document.
activeElement).toBe(screen.getByRole('dialog'))` (or query by the panel's `data-testid`) to the
existing "renders as an accessible, labelled dialog when open" test.

### Minor — a successful reset via `ResetControl` does not invalidate `useShopState()`'s cache, so stale cart/order data can render briefly post-reset

`ResetControl` sits outside `QueryClientProvider` (same root cause as the major above), so unlike
`useResetMine()` — which would invalidate `queryKeys.state(participantId)` — it has no cache to
invalidate. Because `customerId == participantId` survives "reset mine" (§4.8), the query key is
unchanged, so `useShopState()`'s cache keeps serving pre-reset cart/order data until the next 2 s
poll tick (§5.3 C8) organically overwrites it. Low real-world impact — the immediate post-reset
render is the language step, not a cart/order view, and the staleness window is bounded by the
poll cadence — but worth naming alongside the major, since it is the same architectural gap
manifesting a second way. No action required beyond what the major already recommends.

## What's solid

- **The `sessionBridge`/`injectBridge` mechanism is sound for what it does.** `spliceIntoGrandchild`
  makes exactly one structural assumption (`Children.only` on two levels), states it in a comment,
  unit-tests both the happy path and two distinct no-op-degrade shapes (`injectBridge.test.tsx`),
  and the singleton setter in `sessionBridge.ts` guards its own cleanup (`if (currentSetter ===
  setPendingLanguageStep) currentSetter = null`) — traced against React 19 `StrictMode`'s dev-only
  double-invoke (`main.tsx` does wrap in `StrictMode`) and against `useState`'s guaranteed-stable
  setter identity: the mount→cleanup→remount cycle converges correctly, and nothing here is
  exercised only by inference — `Shell.test.tsx` mounts the *real* provider stack (not a stand-in)
  specifically so a regression in the splice assumption would show up there too.
- **The reset-control happy path is tested on rendered state, not the fetch**, per the plan's own
  bar for this exact class of defect (`ResetControl.test.tsx`'s third case) — `pendingLanguageStep`
  flips, the credential survives, the sheet closes, matching §4.8/C7 precisely.
- **The four S14 seed placeholders are genuinely inert** — no hooks, no data, one line of static
  copy each, matching `i18n/Provider.tsx`'s and `layout/Shell.tsx`'s own established pass-through
  style, and wired into their sheets with both a unit test (`Shell.test.tsx`) and an e2e test
  (`mobile-shell.spec.ts`) confirming the wiring rather than just the file's existence.
- **The Playwright suite is well-targeted, not tautological** — real overflow checks
  (`scrollWidth > clientWidth`) at both named viewports, a full open/close/overflow loop over all
  four sheets, and the AC-5 flow asserted on the rendered language-step heading and dialog
  visibility rather than on the intercepted request.
- **No new dependency was added** for icon-heavy header/sheet chrome — inline SVGs
  (`layout/icons.tsx`), consistent with `salesperson/AGENTS.md`'s minimal-footprint framing.

## Open questions

- None that block a verdict here. The plan-amendment question the major raises (reorder `App.tsx`
  now vs. accept the bridge pattern as precedent) is `architect`'s/`teco`'s call, not one this
  review needs the human to resolve before landing S12b's own verdict.

## Pass 2 — 2026-09-13

**Scope of this pass.** A fresh `frontend-engineer` run addressed the blocker and the first minor.
Changed files: `ResetControl.tsx`, `ResetControl.test.tsx`, `BottomSheet.test.tsx`. The major
(`App.tsx` provider nesting) and the second minor (post-reset cache staleness) are explicitly out
of scope — both routed to a separate `architect` unit (`app-composition`) already in flight — and
are not re-investigated here. Re-verified independently rather than taken on report: full `npx
vitest run` (**121/121 green**, 10 files, matching `teco`'s figure), `npx tsc -b` (clean, exit 0).

**Verdict: approve.** The blocker is genuinely closed against C3/C4/C6b/C13, confirmed by two of
my own reverted mutations (not ones `teco` had already run), and the minor is fixed exactly as
recommended. No new findings. The major and the second minor stay open, tracked in the
in-flight `app-composition` unit rather than this document.

### Disposition of Pass 1 findings

- **Blocker (`ResetControl`'s error handling dropped C3/C4/C6b/C13) — fixed, independently
  re-verified.** `ResetControl.tsx:65-107` now runs every non-`ApiError` catch case through the
  old generic message and every `ApiError` through `resolveErrorAction('resetMine', err, 'user')`,
  switching on `.kind`: `clearParticipant` (401/C3) calls `clearParticipantSession()` and
  navigates to join; `reread` (504/C4-F8) gets its own "couldn't confirm whether that worked"
  copy, textually distinct from the 503 copy; `unscopedAlarm` (409/C6b) gets its own copy with no
  retry language; `unhandled` (C13) names the status in a visible message; an unreachable
  `default` stays loud rather than silently swallowing a future classification change. I
  independently mutated **two** branches `teco`'s own report did not name as its check (`teco`
  re-mutated `unscopedAlarm`): reverted `clearParticipant` to the old stay-on-confirm-step/generic-
  message behavior → the new 401 test's `localStorage` assertion reddened
  (`ResetControl.test.tsx:156`, expected `null`, received the still-present session JSON); reverted
  `reread` to the same generic copy → the new 504 test's alert-text assertion reddened
  (`ResetControl.test.tsx:173`). Both restored via file copy, confirmed byte-identical (`diff`
  clean against my pre-mutation backup) and the full suite green again before moving on. The four
  new test cases (401/504/409-unscoped/500-unhandled) each assert a real effect — actual
  `localStorage` state, an actual `router.navigate` spy call, or alert copy with an explicit
  negative assertion against a sibling branch's copy — not just that some error path was taken,
  which is the right altitude for this defect class per this same document's Pass-1 evidence.
- **Minor (`BottomSheet` focus-on-open untested) — fixed exactly as recommended.**
  `BottomSheet.test.tsx:27` adds `expect(document.activeElement).toBe(dialog)` to the existing
  accessible-dialog test. Not independently re-mutated this pass (a one-line assertion addition
  against an unchanged, already-correct implementation — the fix and its own rationale are
  legible directly from the diff, and the addition is exactly what Pass 1 asked for).

### Findings carried forward, unresolved

- **Major — root cause of the `App.tsx` provider-nesting gap.** Unchanged since Pass 1; being
  handled by the separate `app-composition` `architect` unit per `teco`'s message. Not re-checked
  this pass.
- **Minor — post-reset `useShopState()` cache staleness.** Unchanged since Pass 1; deferred to the
  same `app-composition` unit. Not re-checked this pass.

## Pass 3 — 2026-09-14

**Scope of this pass.** The consolidated closing-gate review of S12b's full, final state:
plan v1.36 §4.11 (`0f99b63`), `App.tsx` (providers reordered, `LayoutShell` no longer rendered
here), `routes.tsx` (the three routes nested under one new pathless layout route,
`element: <LayoutShell />`), `Shell.tsx` (`LayoutShell` takes no props, renders `<Outlet/>`,
splice code dropped), the deletion of `layout/sessionBridge.ts` / `layout/injectBridge.tsx` /
`layout/injectBridge.test.tsx`, `ResetControl.tsx`'s rewrite onto `useResetMine()`, both rebuilt
test files (`ResetControl.test.tsx`, `Shell.test.tsx`), and `salesperson/AGENTS.md`'s extended
ownership rows — plus a full re-check that the rest of S12b's original scope (mobile shell,
seed placeholders, `BottomSheet`, Playwright suite) still holds under the new composition, not
just the delta since Pass 2.

**Verdict: approve with suggestions.** The blocker, the major and both minors from Pass 1/2 are
now genuinely closed — independently re-verified, not taken on report, including two of my own
reverted mutations against code `teco`'s own checks didn't cover. One new minor surfaces, found
specifically by checking the landed fixes against each other rather than each in isolation (a
transient stale-error-copy regression introduced by the `useResetMine()` rewrite, reproduced by
execution, not inferred) — cosmetic and self-correcting, not a reason to hold the gate, but worth
fixing before it's forgotten.

**Independent verification this pass:** full `npx vitest run` — **118/118 green**, 9 files (down
from Pass 2's 10/121, exactly accounted for by `injectBridge.test.tsx`'s 3 tests going with the
file it tested); `npx tsc -b` clean; `./build.sh` clean, bundle emitted with the `/shop/` prefix
intact. `grep`-confirmed no surviving reference to `sessionBridge`/`injectBridge`/
`spliceIntoGrandchild` anywhere in `src/`/`tests/` except one explanatory comment in `Shell.tsx`
naming the deleted mechanism for context. `git status`/hashes of every file this row does *not*
list as changed (`BottomSheet.{tsx,test.tsx}`, `CartSheet.tsx`, `OrderSheet.tsx`,
`CatalogSheet.tsx`, `ProfileSheet.tsx`, `Header.tsx`, `icons.tsx`, `SheetContext.tsx`, all four
seed placeholders, `playwright.config.ts`, `mobile-shell.spec.ts`) read directly and confirmed
unmodified — the full original S12b scope wasn't silently touched by this revision. Playwright's
10/10 (both viewports, AC-5 end to end) is `teco`'s own independently-run figure from a live
`vite preview`; not re-run this pass — no reason to doubt it, and the full unit-test suite
already re-covers the same component wiring at a faster loop.

### (1) Does this close the Major and second Minor, and does the resolution match what I'd have recommended?

**Yes to both, cleanly.** I traced `App.tsx` → `routes.tsx` → `Shell.tsx`'s actual render tree
directly rather than the plan's own description of it: `LayoutShell` is now the router's top-level
pathless layout route (`{ element: <LayoutShell />, children: [...] }`), rendered *by*
`RouterProvider` as part of its own matched-route output — a genuine descendant of
`SessionProvider`/`QueryClientProvider` (which now wrap `RouterProvider` directly, no `children`
slot needed) *and* inside the router's own `NavigationContext`/`LocationContext`, since those are
established once around `RouterProvider`'s whole rendered output, not per leaf route. This is
materially better than the provider-order-swap-only alternative I'd floated as one option in
Pass 1's major (which the plan's own §4.11 also traces and rejects, correctly, on the same
grounds I would have: `RouterProviderProps` has no `children`, so a bare swap fixes
session/query reachability but leaves `useNavigate()`/`useLocation()` exactly as unreachable as
before) — the pathless-layout-route mechanism is the standard react-router pattern for exactly
this shape, not a bespoke fix. The second minor (no post-reset cache invalidation) is closed as a
consequence: `useResetMine()`'s own `onError` already calls `queryClient.invalidateQueries(...)`
on a `reread`/`state` action (`api/hooks.ts:322-324`), and `ResetControl` now goes through that
hook rather than working around it.

### (2) Interaction check across the landed fixes

Traced the full chain end to end (`App.tsx` → `routes.tsx` → `Shell.tsx` → `ResetControl.tsx` →
`api/hooks.ts`) rather than checking each file against its own finding in isolation. Two things
worth naming:

- **No leftover references, no dead imports, no double-navigation.** `ResetControl.tsx` no longer
  imports `router`/`APP_PATHS`/`resolveErrorAction` directly (`useResetMine()` owns all of that
  internally) — confirmed by reading the full import list, not just the diff. On a 401,
  `useErrorEffects`' own navigate (`api/hooks.ts:70-72`, conditioned on not already being on
  `/shop/presenter`) and `ResetControl`'s new `useEffect`-driven `onReset?.()` (closing the
  profile sheet) fire independently and don't conflict — proved this is genuinely load-bearing,
  not just plausible-looking, with a mutation not in `teco`'s own table: reverted the `useEffect`
  to a no-op (`if (false) onReset?.()`), reran the suite, and exactly the "clears the participant
  credential and closes the sheet on a 401" test reddened (`dialog` still present); restored via
  `cp`, `diff`-confirmed byte-identical, full suite green again (118/118) and `tsc -b` clean.
- **New minor found by the cross-check, not by either fix's own test suite** — see below.

### (3) New — Minor: a stale error message from a prior failed attempt can render during a same-session retry's pending window

`ResetControl.tsx:75` derives `error` as `errorMessageFor(resetMine.action)` on every render;
`resetMine.action` (`api/hooks.ts:310,315,322`) is a `useState` set only inside `useResetMine()`'s
own `onSuccess` (→ `null`) or `onError` (→ the new `ErrorAction`) — nothing clears it when a new
`mutate()` call *starts*. Pass 2's now-superseded inline version guarded this explicitly
(`setError(null)` at the top of `handleConfirm()`, before the `try`); the v1.36 rewrite dropped
that guard along with the rest of the hand-rolled dispatch, and nothing replaced it. Reproduced by
execution, not inferred: a scratch probe (409 first attempt → alert shows "no longer scoped to
this store" → click "Yes, reset" again without closing the sheet, second `fetch` deliberately slow
→ the alert with the **first** attempt's text is still on screen while the second request is
in flight) asserted true and passed; removed after confirming, not left in the tree. Self-correcting
once the second request settles (the next render picks up the new `action`), so this is cosmetic —
a participant retrying after any error (409/504/503/500 alike) can see stale, possibly misleading
copy for the duration of the retry — not a blocker, but a real regression relative to what Pass 2
shipped for the same interaction. **Fix, scoped entirely to `ResetControl.tsx` (no `api/hooks.ts`
edit, no ownership crossing):** `const error = resetMine.isPending ? null : errorMessageFor(resetMine.action);`
— suppresses the stale copy for exactly the pending window, with no state-management change
needed in the S12a-owned hook. A regression test: mock a first failing response, confirm the alert,
click "Yes, reset" again with a deliberately slow second `fetch`, assert `screen.queryByRole('alert')`
is null immediately after the second click and before the second response resolves.

### (4) Full original S12b scope, re-checked under the new composition

Mobile shell (sticky header, four icon buttons, safe-area classes), the `BottomSheet` a11y
contract (role/aria-modal/focus-on-open/Escape/backdrop — Pass 2's fix intact, re-read
unchanged), the four S14 seed placeholders and their sheet wiring, and the Playwright suite's own
assertions are all **unchanged files**, confirmed by direct re-read against Pass 1/2's record
rather than assumed stable because the diff summary didn't name them. `Shell.test.tsx`'s rebuilt
harness mounts `LayoutShell` as a real layout route (matching `App.tsx` exactly) and its four
cases — header+routed-content render, sheet open/pressed-state, close-without-affecting-routed-
content, all-four-placeholders-wired — all still pass unmodified in substance (only the mounting
harness changed to match the new composition). AC-4's no-horizontal-overflow and AC-5's full
reset-confirm-to-language-step flow are covered end to end by `teco`'s independently-run
Playwright pass (10/10) and by the unit suite's own equivalent coverage; nothing in this pass
found either to have regressed.

### Disposition of Pass 1/2 findings

- **Blocker (dropped C3/C4/C6b/C13) — closed via the v1.36 architectural fix, independently
  re-verified.** Superseded rather than patched: `ResetControl` now delegates entirely to
  `useResetMine()`, which already implements the full dispatch. Re-verified by reading
  `ResetControl.tsx` in full (not diffed against Pass 2's version, since the whole approach
  changed) and by two of my own reverted mutations (§2 above and the Pass-2-style 401/504 checks,
  re-run against the new shape) — both genuinely load-bearing.
- **Major (provider-nesting root cause) — closed, matches the recommended remedy.** See (1) above.
- **Minor (`BottomSheet` focus-on-open untested) — unaffected by this pass, still fixed** (Pass 2
  disposition stands; `BottomSheet.test.tsx` untouched by v1.36).
- **Minor (post-reset cache staleness) — closed as a consequence of the major's fix.** See (1)
  above.
