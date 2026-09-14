# S12a — Session + API client + routing (`salesperson/src/{api,session}/**`, `routes.tsx`, shared entry files)

> **Status:** active · **Owner:** `analyst` · **Tracks:** S12a (salesperson-ui)

## Scope & verdict

Reviewed: `salesperson/src/session/**`, `salesperson/src/api/**`, `salesperson/src/routes.tsx`,
`salesperson/src/routePaths.ts`, the two S12b/S12c placeholder slots
(`src/layout/Shell.tsx`, `src/i18n/Provider.tsx`), and the rewritten shared entry files
(`App.tsx`, `main.tsx`, `index.css`), against the S12a row in `docs/plans/salesperson-ui.md`
§5.1 and, in full, §5.3 ("The client's credential & session contract" — C1–C14) and §5.2 (the
`/shop/api` surface). `salesperson/AGENTS.md` read for file ownership and toolchain. Verified
independently rather than taken on report: full `npx tsc -b` (clean), full `npx vitest run`
(85/85 green, both before and after every mutation below), `main.tsx` diffed byte-for-byte
against the `be1ef30` scaffold commit (identical, confirming the "unchanged" claim rather than
just accepting it), `git status`/`git diff --stat` re-checked against the delivered file list,
and every one of C1–C14's dispatch-level tests read line-by-line against §5.3's prose. Three
reverted mutations run against the actual source (edited, `npx vitest run`, diffed back against
a pre-mutation copy, restored, re-verified byte-identical and green) rather than reasoned about
statically — one required by the brief (a rule the delegate did not report mutating), two
volunteered because reading the wiring surfaced a specific, checkable doubt. `routes.tsx`'s
integration-seam design judged against §5.0's file-ownership table, read in full rather than
skimmed.

**Verdict: needs changes.** Two blockers, both variations on the same theme: the delivered test
suite verifies §5.3's rules as a *pure classification function* thoroughly and well, but does not
verify, at the point where the rule actually has an effect (the TanStack Query wiring in
`hooks.ts`), that the classification is *acted on* correctly — which is exactly the failure shape
the plan spent eight review passes and this section's own C13 essay warning about ("a global 401
handler and a wrong re-read endpoint each produce the right rendered outcome for the wrong
reason"). Both are proven with a reverted mutation, not asserted from reading. One further major
(a genuine file-ownership gap for S12d/S13/S14) and one minor round out the findings. The
architecture, the pure-dispatch test suite, and the session/storage layer are otherwise
well-grounded, and `main.tsx` is confirmed genuinely untouched.

**CPG:** considered, not relevant — `salesperson/` is a fresh, S5-scaffolded TypeScript tree with
no CPG loaded (`cpg_salesperson` does not exist among this instance's graphs, and none of the
loaded `cpg_*` graphs cover a component that includes it); the review was done by direct reading
of the ~14 small, freshly-authored files under review, which is faster and more precise for this
size than building a graph would have been.

## Findings

### Blocker — C4's re-read is verified only as a pure classification, never as the network request the rule exists to produce; proven via reverted mutation

`dispatch.test.ts`'s C4 block (`src/api/dispatch.test.ts:79-118`) thoroughly enumerates all five
writing routes against `resolveErrorAction`, a pure function operating on a plain `ApiError`
object — no `fetch`, no React. But the *consumer* of that classification — each mutation hook's
`onError` branch in `hooks.ts`, which is what actually performs the re-read — is exercised by no
test anywhere in the suite. I proved this is load-bearing, not incidental, by mutating
`usePresenterResetAll`'s `onError` (`hooks.ts:459`) from `resolved.via.endpoint ===
'presenterParticipants'` to `resolved.via.endpoint === 'state'` — i.e., exactly the historical
defect shape §5.3 C4/C5 exists to prevent, reproduced one layer up from where `dispatch.test.ts`
looks. Full suite: **85/85 still green.** Restored and re-verified byte-identical
(`diff` clean) and green before moving on. The same gap holds for `resetMine` and
`orderAdvance` (their `onError` branches are equally untested at the network level), and for
`messagesPost`'s reconciliation — `reconcilePostMessageFailure` is unit-tested as a pure function
(`hooks.test.tsx:74-95`) but the mutation's actual `onError` path (the `Promise.all([getMessages,
getState])` call at `hooks.ts:211-214`) is never driven through a mocked `fetch` to confirm it
hits `GET /shop/api/messages` **and** `GET /shop/api/state`, nor is `resetMine`'s or
`orderAdvance`'s.

This is not a stylistic gap — it is the specific, twice-stated done-condition for this exact
pair of rules: *"For C3 and C4 alike, assert the intercepted request and the stored credentials,
never the rendered outcome — a global 401 handler and a wrong re-read endpoint each produce the
right rendered outcome for the wrong reason, which is exactly how both defects survived four
review passes"* (`docs/plans/salesperson-ui.md`, S12a row). C3 *is* asserted this way
(`hooks.test.tsx:97-147` drives real navigation/session state through a mounted router). C4 is
not, for any of its five routes.

**Fix:** for each of `resetMine`, `orderAdvance`, `presenterResetAll` and `messagesPost`, add a
`hooks.test.tsx` case that mounts the relevant hook with a mocked global `fetch`, returns a `504`
on the write, and asserts the **next** `fetchMock` call's URL matches the rule's endpoint exactly
(`/shop/api/state`, `/shop/api/state`, `/shop/api/presenter/participants`,
`/shop/api/messages` + `/shop/api/state` respectively) — mirroring the call-inspection pattern
`hooks.test.tsx:189-236` (C8) and `:238-334` (C12) already use for other rules on this same
surface. That pattern is proven to work in this codebase; it is simply not applied to C4.

### Blocker — C8's "one shared constant, not two literals" is not actually asserted; proven via reverted mutation

The plan's own prescribed proof for C8 is explicit: *"asserted by changing that constant in the
test and observing both intervals move, so two literals fail"* (S12a row). The delivered test
(`hooks.test.tsx:189-236`) does something weaker: it imports `POLL_INTERVAL_MS` and uses its
*current value* to compute how long to advance fake timers, then checks both `/state` and
`/messages` fired together. It never varies the constant, so it cannot distinguish "both hooks
import the same symbol" from "both hooks happen to hard-code the same number today." I proved
this by editing `useMessages`'s `refetchInterval` from `POLL_INTERVAL_MS` to a hard-coded literal
`2_000` (`hooks.ts:167`) — precisely the regression C8 exists to catch — and re-ran the full
suite: **85/85 still green.** Restored and re-verified byte-identical and green.

**Fix:** `vi.mock('./polling', () => ({ POLL_INTERVAL_MS: <different value> }))` (or an
equivalent override) in a dedicated C8 test, then assert both `/state` and `/messages` refetch
at the *mocked* interval rather than at `2_000` — that is what actually fails when one hook
regresses to a private literal. The existing test is still useful (it does prove the two ticks
are aligned) and can stay alongside it.

### Major — `routes.tsx`'s file ownership has no path for S12d/S13/S14 to mount their real views without an unauthorized edit

`routes.tsx` ships, correctly per its own row, with minimal inline placeholder `JoinScreen` /
`ChatScreen` / `PresenterKeyScreen` / `PresenterRoster` components — reasonable as an integration
seam to prove the live round-trip and exercise every hook, and explicitly framed as such in the
file's own header comment. The problem is downstream: §5.0's file-ownership table
(`docs/plans/salesperson-ui.md:1061`, and the file column of the S12a/S12b/S12c/S12d/S13/S14 rows)
assigns `src/routes.tsx` to **S12a only** — none of S12b, S12c, S12d, S13 or S14's rows list
`routes.tsx` among their owned paths, unlike the three shared entry files, which at least get an
explicit "no later step edits them" *and* an explicit slot mechanism (`I18nProvider`/
`LayoutShell`) so S12b/S12c never need to touch them. `routes.tsx` has no equivalent slot: S13's
`views/Chat*`, S14's `views/{Cart,Order,Profile,Catalog}*` and S12d's `views/Presenter*` can only
reach the mounted app by S13/S14/S12d editing `routes.tsx`'s JSX directly — an edit the plan's own
table does not authorize any of those three rows to make. Today this is latent (S12a's
placeholders work fine standalone); it becomes live the moment S12d, S13 or S14 starts, i.e.
immediately once this gate opens.

This does not reflect on S12a's own correctness — it built exactly what its row specifies — but
it is a real coordination gap on the five units this review is gating, which is exactly the kind
of scope/ownership question the brief asked me to make a call on rather than leave open. **Call:**
flag as major rather than block S12a on it, since S12a's own deliverable is sound; but it should
be resolved (by `teco`/`architect`) before S12d/S13/S14 are dispatched, not discovered mid-unit.
Two reasonable repairs: (a) amend §5.0 to grant S12d/S13/S14 a narrow, additive edit right on
`routes.tsx` (swap one placeholder's JSX for a real import, nothing else), or (b) have `routes.tsx`
import its three screens from a small per-screen registry file that S12d/S13/S14 already own,
so `routes.tsx` itself never needs a post-S12a edit.

### Minor — `presenterSession`'s classification licenses a response the server contract says cannot occur

`PRESENTER_ROUTES` (`dispatch.ts:80-84`) includes `presenterSession`, so a `401` on that route —
which §5.2's classification says cannot happen (`presenterSession` carries no credential and
touches no `get_presenter` dependency) — falls through to `clearPresenter` rather than to C13's
`unhandled` path. Harmless today because the server cannot produce it, but it quietly narrows
C13's "anything unruled is loud" guarantee for this one cell: if a future server regression ever
did emit a `401` there, the client would silently clear a credential rather than surface the
unrecognized response C13 is designed to make loud. Suggest special-casing `presenterSession`
ahead of the generic `PRESENTER_ROUTES.has(route)` check (as its `403` already is) so only its own
named case is handled and every other status on that route — including a `401` — falls through to
`unhandled`.

## What's solid

- **The pure-dispatch architecture and its test suite are genuinely strong.** `resolveErrorAction`
  keying on `route` before `status` structurally forecloses the single-global-401-handler defect
  (C1); `dispatch.test.ts`'s C2/C3/C4/C9/C10/C11/C13/C14 blocks enumerate every route/field/source
  cell §5.3 names, not a sample of them, and I independently re-verified this by reading every
  cell against the plan's own tables rather than trusting the delegate's naming.
- **Mutation-testing credibility holds up on what it covers.** I reverted-mutated C9's
  poll/user split (a rule the delegate did not report touching) and confirmed it goes red
  (2 failures, restored clean) — the classification layer's tests are real, not decorative.
- **`main.tsx` is confirmed byte-identical** to the `be1ef30` scaffold commit — checked by diff,
  not by trusting the claim.
- **Storage/session layer is well-tested, including the failure path**: `storage.test.ts`
  exercises a throwing `localStorage` (private-mode Safari shape) and confirms graceful
  degradation rather than a crash — a case easy to skip and not skipped.
- **C3's "must not navigate the presenter away" nuance is tested with a real router**
  (`hooks.test.tsx:97-147`), which is the harder and more valuable test than the pure-dispatch
  equivalent — this is the one C-rule where the plan's "assert the intercepted request/effect"
  bar is actually met end-to-end.
- **`react-router-dom` addition is within the sanctioned exception** (`salesperson/AGENTS.md`'s
  "a router library is deliberately not chosen... S12a picks and installs one"), and
  `package.json`/`package-lock.json` show only that one addition.
- **The `App.tsx` slot composition is correct and pass-through today**, confirmed by reading
  (not just running): `I18nProvider` and `LayoutShell` wrap `QueryClientProvider`/
  `SessionProvider`/`RouterProvider` with no-op bodies, matching the "S12b/S12c mount without
  editing App.tsx" contract.

## Open questions

- None that block a decision here — the `routes.tsx` ownership gap (major, above) is a question
  for whoever sequences S12d/S13/S14 next, not one this review needs the human to resolve before
  landing a verdict on S12a itself.

## Pass 2 — 2026-09-13

**Scope of this pass.** A fresh `frontend-engineer` run addressed both Pass-1 blockers and the
minor. Changed files: `dispatch.ts`, `dispatch.test.ts`, `hooks.test.tsx`; `hooks.ts`'s net diff
is reported empty. Re-verified independently rather than taken on report — see per-finding
disposition below. The major (`routes.tsx` ownership) is explicitly out of scope for this pass,
routed to `architect` and landed separately as plan v1.34; not re-checked here.

**Verdict: approve.** Both blockers are fixed and the fixes hold up under my own reverted
mutation testing (three mutations run this pass, all independently, all confirmed red then
restored clean); the minor is fixed exactly as recommended. No new findings.

### Disposition of Pass 1 findings

- **Blocker 1 (C4 network-effect gap) — fixed, independently re-verified.** `hooks.ts` confirmed
  byte-identical to my Pass-1 backup (`diff` clean) before checking anything else — the reported
  "net diff empty" claim is not just accepted, it's checked. Four new `hooks.test.tsx` blocks
  (`resetMine`, `orderAdvance`, `presenterResetAll`, `messagesPost`) mock `fetch`, force a `504`
  on the write, and assert the logged URL of the *next* call(s) via a new `fetchCallLog` helper —
  exactly the fix I recommended, and exactly the C8/C12 call-inspection pattern already in this
  file. I independently mutated `usePresenterResetAll`'s `onError` endpoint check back to
  Pass-1's bug (`'presenterParticipants'` → `'state'`) — a rule the coordinator's report also
  named, re-derived rather than trusted — and confirmed the new test goes red (`expected 2 to be
  greater than 2`), then restored `hooks.ts` byte-identical and reconfirmed 92/92 green. Did not
  independently re-mutate the `resetMine`/`orderAdvance`/`messagesPost` cases; read their test
  bodies in full and they mirror the same proven pattern against the same `hooks.ts` code I
  diffed as unchanged, which is sufficient given the one I did mutate is the same code shape.
- **Blocker 2 (C8 shared-constant gap) — fixed, independently re-verified.** `vi.mock('./polling',
  () => ({ POLL_INTERVAL_MS: 5_000 }))` at module scope, plus a self-check
  (`expect(POLL_INTERVAL_MS).toBe(5_000)`) that documents why the mock must actually take effect
  for the surrounding test to mean anything. I independently re-ran my own Pass-1 mutation
  (hard-coding `useMessages`'s `refetchInterval` to a literal `2_000`) against the *fixed* suite
  and confirmed both the original C8 test and the new one go red (`expected 4 to be 2`), then
  restored and reconfirmed 92/92 green. This is the same mutation I ran in Pass 1 (where it was
  silently green) — re-running it here is what actually proves the fix, not just reading the new
  test's intent.
- **Minor (`presenterSession`'s 401 licensed by `PRESENTER_ROUTES` membership) — fixed exactly as
  recommended.** `presenterSession` is now special-cased ahead of the generic
  `PRESENTER_ROUTES.has(route)` branch (`dispatch.ts:119-134`); only its own `403` maps to
  `presenterKeyRejected`, everything else (including a 401) falls through to `unhandled`. New
  `dispatch.test.ts` case asserts exactly that. I independently reverted the special-case to
  Pass-1's shape and confirmed the new test goes red, then restored clean.

**Independent verification this pass:** `git status --porcelain salesperson/` shows only the
expected S12a file set (no stray mutation artifacts); `npx tsc -b` clean; `npx vitest run` 92/92
green, both as delivered and after each of my three reverted mutations was restored.
