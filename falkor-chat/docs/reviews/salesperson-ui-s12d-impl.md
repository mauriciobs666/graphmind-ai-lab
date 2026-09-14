# Review: salesperson-ui S12d (Presenter view) — implementation

> **Status:** active · **Owner:** `analyst` · **Tracks:** S12d (`docs/plans/salesperson-ui.md` §5.1, v1.38)

## Scope & verdict

Reviewed the uncommitted `salesperson` working-tree diff implementing S12d (the presenter view)
against `docs/plans/salesperson-ui.md` §5.1's S12d row (v1.38, read in full), §5.2's roster
projection and `reset-all` response shape, and §5.3's C1–C14 client contract (read in full, with
particular attention to C2, C4, C9, C11 and C13 per the brief). Also checked file-ownership
conformance against `salesperson/AGENTS.md`'s ownership table and test quality/coverage against the
delivered `*.test.tsx`/`presenter.spec.ts` files. Files in scope: `salesperson/src/routes.tsx` (diff
only), `salesperson/src/views/Presenter{KeyScreen,Roster,ResetAllControl}.{tsx,test.tsx}`,
`salesperson/tests/e2e/presenter.spec.ts`, and the three `salesperson/src/locales/*.json` diffs.
`salesperson/src/api/hooks.ts`/`dispatch.ts` were read for context (they are S12a-owned, already
committed at `7ab7a97` and prior, and out of this unit's licensed scope) but not re-reviewed as a
subject in their own right.

**CPG:** not applicable — `cpg_salesperson` does not exist in this FalkorDB instance (confirmed by
prior reviews in this coordination); this is a front-end SPA review grounded directly in source
reading, not a graph-backed task.

Independently reproduced: `npx tsc -b` clean; `npx vitest run` → **237/237 passed, 26 files**,
matching both the implementer's and `teco`'s reports. `git diff --stat` matches the file list in the
brief. Locale key-parity (27 `presenter.*` leaf keys per locale) re-derived and confirmed. The
`routes.tsx` diff is exactly the licensed narrow swap — two new imports replacing two inline
placeholders and their now-dead hook imports; `ParticipantRoute`'s switch logic and everything else
in the file is untouched.

**Verdict: needs changes** (three Major findings, no blockers — all three are additive fixes inside
files this unit already owns, none touch a shared entry file or another step's subtree).

## Findings

### Major — `PresenterRoster` folds C13's "unhandled" rule into C9's staleness copy; a genuinely
unmapped response never renders as C13 requires

`salesperson/src/views/PresenterRoster.tsx:28-35` branches only on `roster.isError` (a plain
boolean), never on `roster.action?.kind`. Both a documented `503` (C9 — "nothing changed") and an
undocumented, structurally-impossible-per-spec response (C13 — "any unruled `(route, response)`...
renders an explicit 'unhandled response' failure... naming the route and the status") render the
same `presenter.roster.loadError`/`staleNotice` copy ("Couldn't load the participant roster.
Retrying automatically…" / "Showing the last known roster — reconnecting…"). Neither string names a
route or a status. `salesperson/src/locales/en.json` has no `presenter.roster.error.unhandled` key
at all (checked all three locales — none do), unlike the sibling components in this same diff:
`PresenterKeyScreen.tsx:64-68` and `PresenterResetAllControl.tsx:33-35` both explicitly branch on
`action?.kind === 'unhandled'` and render a distinct, status-interpolated message.

Verified live: `PresenterRoster.test.tsx:64-71` fires a bare `500` — which `dispatch.ts`'s
`resolveErrorAction` resolves to `{kind:'unhandled', route, status}` (no branch matches 500: not
422/401/403/503/504/409, falls to the final `return {kind:'unhandled', ...}`) — and the test asserts
the *generic* loadError copy, i.e. it locks in the wrong behavior rather than catching it.

Why it matters: §5.3 devotes its single longest passage to C13 precisely because "a rule matches and
is wrong" is the residual class that survived eight review passes and one live incident (D-2/C14);
the plan explicitly names this exact case — "any unmapped response — ... C13 is the proof it does
not [exist]" — as the row S8's completeness table stakes on the map being total. If S8's map is ever
wrong for `GET /shop/api/presenter/participants`, this is the one view with no client-side tripwire
for it; an operator sees "reconnecting…" instead of the diagnostic surface C13 exists to provide.

Suggested fix: give `PresenterRoster` the same `unhandled` branch its two siblings already have — a
new `presenter.roster.error.unhandled` key (status-interpolated, mirroring the other two files) and
an explicit `roster.action?.kind === 'unhandled'` check rendered ahead of the generic
`loadError`/`staleNotice` fallback. Replace the existing `500` case in
`PresenterRoster.test.tsx:64-71` with an assertion on the new distinct copy (route/status named), and
add a second case that still exercises the true C9 `503` path (`graph_unavailable`/
`graph_read_timeout`) rendering the staleness copy, so the two are asserted as genuinely different
outcomes rather than one test standing in for both.

### Major — `PresenterKeyScreen` shows a stale rejection message through a second, pending submit

`salesperson/src/views/PresenterKeyScreen.tsx` has no `login.isPending ? null : ...` masking around
its error derivation (`keyRejected`/`fieldError`/`unhandled`, lines 24-36) — unlike
`PresenterResetAllControl.tsx:57-58` in this same diff, whose comment names the exact defect class
("A retry in flight clears the previous attempt's error/result immediately rather than leaving it
rendered until the new response lands (mirrors `ResetControl.tsx`'s own analyst-flagged fix)"), and
unlike `salesperson/src/components/sheets/ResetControl.tsx:82`, the file that established the
pattern after an earlier analyst finding.

Reproduced live (throwaway test, not committed): submit a wrong key → "That key was not accepted."
renders; submit again while the second request is still in flight → the stale rejection text is
still on screen (`getByRole('alert')` still returns the old node) even though the button already
reads "Checking…". `login.action` (from `usePresenterLogin` in the already-committed `hooks.ts`) is
only cleared in the mutation's own `onSuccess`/`onError`, never at `mutate()`-call time, so nothing
clears it until the new response lands — exactly the shape the codebase already named and fixed
twice elsewhere.

Why it matters: this is a real, user-visible correctness bug (a rejected-key error appears to still
be in effect during a submission that may well succeed), it is the identical defect class a review
already caught and fixed twice in sibling files of this same tree, and the fix requires no grant
beyond this unit's own file — no `hooks.ts` edit needed.

Suggested fix: `const error = login.isPending ? null : { keyRejected, fieldError, unhandled }` (or
equivalently gate each derived boolean on `!login.isPending`), mirroring
`PresenterResetAllControl.tsx:57`. Add a test exercising two submits in a row (reject, then a second
attempt left pending) asserting the alert is absent during the pending window — the throwaway
reproduction above is a ready template.

### Major — the reset-all `_one` (singular, count = 1) plural forms are exercised by no test in any
locale for `success`, and only incidentally for `incomplete.heading`

Independently reproduced `teco`'s mutation: replaced `en.json`'s `presenter.resetAll.success_one`
and `presenter.resetAll.incomplete.heading_one` with garbage strings, reran the full suite —
**237/237 stayed green**. Restored from a `test -s`-verified backup, confirmed `diff -q`
byte-identical, reran clean. My own read of every `*.test.tsx` confirms why: the only test that
drives a `count === 1` reset-all response through `success` uses `clearedParticipants: 3`
(`PresenterResetAllControl.test.tsx:70`, `_other`); the only `count === 1` case anywhere is the
locale-switch test's `incomplete` banner (`unresolved: ['p-1']`, `PresenterResetAllControl.test.tsx:
159-177`), which happens to exercise pt-BR's `incomplete.heading_one` — but exercises no locale's
`success_one`, and no other locale's `incomplete.heading_one`.

This is my own judgment, not a restatement of `teco`'s: I rate it **Major**, not minor. A one-
participant `reset-all` (`clearedParticipants: 1`, no `incomplete`) is an entirely ordinary outcome
of the most common demo shape (one presenter, one participant) — not an edge case reachable only
under contrived conditions — and i18next's plural-suffix resolution provides no safety net here: if
`success_one` is missing or garbled in a shipped locale bundle, the participant-facing English is
either raw (`{{count}} participant cleared.` never resolving) or i18next's own fallback behavior
applies silently, and nothing in this suite would notice either way.

Suggested fix: for `success`, add one `count: 1` case per locale actually asserted on rendered text
(at minimum en, since the other two already get incidental `count`-varying coverage through the
i18n-switch tests — but note that coverage is for `incomplete.heading`, not `success`, so `success`
needs its own `count: 1` case regardless of locale). For `incomplete.heading`, add the missing en/es
`count: 1` cases; pt-BR's is already covered via the existing locale-switch test. A coverage probe
that generalizes rather than enumerates: for every `t()` call site in this diff that passes `count`,
assert against both `count: 1` and `count: 2+` in at least one locale — the axis being tested is
"singular vs. plural form resolves", not a fixed pair of numbers, so a third grammatical number (not
applicable to en/es/pt-BR, but the shape of the gap generalizes) would be equally invisible today.

## What's solid

- The four-key roster projection is correctly scoped: `participantId` used only as row key,
  `joinedAt` unused, and the negative-control fixture (a participant carrying `cartItemCount`/
  `orderStatus` in the raw response) is present in both the unit test and the e2e spec, asserting
  none of it leaks into the DOM — a genuinely strong test for the contract's actual point.
- C2's two-meanings-one-action split (bad key vs. unconfigured key) is correctly left
  undifferentiated on the client, matching §5.3's explicit instruction that the client "cannot and
  must not distinguish" them.
- C4's `504` re-read wiring (`presenterParticipants` invalidation) and C9's user-vs-poll split are
  correctly delegated to and already covered by the already-committed `hooks.ts`/`hooks.test.tsx`
  (S12a); this unit's own components correctly consume `.action`/`.data` without re-implementing
  dispatch logic.
- The `incomplete`/`unresolved` rendering is unambiguous: the incomplete banner and the clean-sweep
  message are mutually exclusive in both the component's JSX and the tests (`queryByRole('status')`
  absent on the incomplete path and vice versa).
- File-ownership and `routes.tsx`/locale-namespace grants are followed exactly — no edit outside the
  licensed set, confirmed via `git diff --stat` against every unlicensed path (`src/i18n/**`,
  `views/Chat*`, `layout/**`, `components/**`, `AGENTS.md`, `docs/HISTORY.md`, `package.json`).
- i18n interpolation placeholders (`{{status}}`, `{{count}}`) are spelled consistently across all
  three locales and all six call sites — checked directly, and confirmed live: mutating one call
  site's argument key (`status` → `Status`) breaks the one test that asserts on the interpolated
  value, so this axis is genuinely covered where it's exercised.

## Open questions

None — all three findings are actionable within this unit's own licensed files, and none changes
scope, touches another step's subtree, or requires a stakeholder decision.

## Pass 2

**Verdict: approve.** All three Pass 1 Majors are fixed and independently re-verified (own
mutation, own restore, own re-run — not a re-statement of the delegate's or `teco`'s report).
Reproduced `npx tsc -b` clean and `npx vitest run` → **243/243 passed, 26 files** (up from 237),
matching both reports exactly. `git diff --stat` on the three locale files is additive-only
(46 lines each); `presenter.*` leaf-key parity re-derived at **28/28/28** across en/es/pt-BR (up
from 27, the one addition being `presenter.roster.error.unhandled`). No interaction found between
the three fixes, nor with anything Pass 1's "What's solid" credited — each touches a disjoint code
path (`PresenterRoster.tsx`'s render branch, `PresenterKeyScreen.tsx`'s derived-state gate, three
locale files' leaf values) and the full suite is green with all three applied together.

- **Finding 1 (`PresenterRoster` C13/C9 conflation) — fixed.** Read the new `unhandled` branch
  (`PresenterRoster.tsx:33-41,59-69`) and the rewritten test file in full. Own mutation: set
  `const unhandled = null;`, ran `PresenterRoster.test.tsx` — exactly 1 failed (the new C13 test) /
  7 passed; restored from backup, `diff -q` byte-identical, full suite reconfirmed 243/243. The old
  bare-500 test that used to lock in the wrong (generic) copy is gone, replaced by a dedicated C13
  test plus two separate documented-503 (C9) tests — the two rules are now asserted as distinct
  outcomes, exactly as suggested.
- **Finding 2 (`PresenterKeyScreen` stale-error flash) — fixed.** Read `PresenterKeyScreen.tsx:28`
  (`login.isPending ? null : login.action`) and the new retry test in full. Own mutation: reverted
  the gate to `const action = login.action;`, ran `PresenterKeyScreen.test.tsx` — exactly 1 failed
  (the new retry test) / 5 passed; restored, `diff -q` byte-identical, file re-run clean at 6/6. The
  new test's shape matches the throwaway reproduction from Pass 1 (reject, retry while the second
  request is held open, assert the alert is gone the instant `isPending` flips).
- **Finding 3 (untested `_one` singular forms) — fixed to my Pass 1 minimum bar.** `success_one` now
  has an exact-text assertion in en; `incomplete.heading_one` now has one in en and es (pt-BR's
  incidental coverage from the locale-switch test is unchanged and now also asserted on exact text).
  Own mutation: corrupted `en.json`'s `success_one` and `incomplete.heading_one`, and `es.json`'s
  `incomplete.heading_one`, in one pass; ran `PresenterResetAllControl.test.tsx` — exactly 3 failed
  / 7 passed, one failure per corrupted key, no collateral failures; restored, `diff -q`
  byte-identical on both files, full suite reconfirmed 243/243. Residual, not a blocker: `success_one`
  is still untested in es/pt-BR — Pass 1 explicitly accepted "at minimum en" as satisfying this
  finding, so this is a non-blocking suggestion, not a re-opened finding: extending the same
  `count: 1` treatment to `success_one` in the other two locales for symmetry with `incomplete.heading`'s
  now-three-locale coverage.

No genuinely new findings. One non-blocking suggestion (above, folded into Finding 3's disposition
rather than listed separately, since it's a narrower residual of the same finding, not a new one).
