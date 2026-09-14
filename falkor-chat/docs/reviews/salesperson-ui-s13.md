# Review: salesperson-ui S13 (Chat view)

> **Status:** active · **Owner:** `analyst` · **Tracks:** S13 (`docs/plans/salesperson-ui.md` §5.1)

## Scope & verdict

Reviewed the delivered S13 unit — `salesperson/src/views/ChatView.tsx`,
`salesperson/src/views/ChatView.test.tsx`, `salesperson/src/components/message/**`
(`Composer.tsx`, `DeadTurnNotice.tsx`, `MessageBubble.tsx`, `Transcript.tsx`, `TurnIndicator.tsx`,
`composerNotice.ts`, `optimistic.ts`, each with its own test file), and `routes.tsx`'s diff (the
`ChatScreen`→`ChatView` swap only) — against `docs/plans/salesperson-ui.md` §5.1's S13 row, §5.2
(the chat behavior contract, incl. *The join greeting*, *The queue position*, *The dead-turn
signal*), §5.3 (C3, C4, C6a, C6b, C9, C11, C13, C14, as rendered — dispatch itself is S12a's and
already gated), and `salesperson/AGENTS.md`'s file-ownership table. Baseline is the uncommitted
working tree; out of scope, per the brief, and not flagged below: S14's already-accepted work, and
the in-flight `S12b-testfix` unit's concurrent edits to `layout/Shell.test.tsx`/
`components/sheets/ResetControl*`.

Independently re-ran `npx vitest run` (178/178 green) and `npx tsc -b` (clean, both matching the
brief's expectation). Read `git diff -- salesperson/src/routes.tsx` directly (100 lines: 3
insertions, 97 deletions) and confirmed it is exactly the narrow, additive swap the plan's §5.0
`routes.tsx` row and S13's own row license — the inline `ChatScreen` placeholder (including its
`useResetMine()`-backed Reset button) replaced by `import { ChatView } from './views/ChatView'`,
nothing else touched. The Reset control is not lost in that swap: it now lives in
`components/sheets/ResetControl.tsx`/`ProfileSheet.tsx` (S12b's subtree, confirmed by reading
those files), so its removal from `routes.tsx` is a relocation the plan already sequenced, not a
regression. `git status --porcelain salesperson/` shows nothing touched outside S13's ownership
grant (`src/views/Chat*`, `src/components/message/**`, plus the one `routes.tsx` line) once the
concurrent S12b-testfix/S14 files are set aside.

**The welcome-turn item — confirmed, not rediscovered.** (a) §5.2's *The join greeting* section
(`docs/plans/salesperson-ui.md:1326-1353`) and §5.3's credentials table (`:1475-1495`, "the
`POST /shop/api/session` response **minus `welcome`**, which is a one-shot greeting rather than
session state") name `welcome` as explicit, named UX — the plan's own words, not an inference —
and S13's own row (`:1254`) commits to rendering it ("never a greeting the client composes"). The
follow-up's premise is sound. (b) `ChatView.tsx`'s delivered code (`views/ChatView.tsx`,
`routes.tsx`) contains **no** reference to `welcome`/`data.welcome` anywhere — `grep -rn "welcome"
src/` turns up only the type declaration (`api/endpoints.ts:37`), a docstring explanation
(`ChatView.tsx:13-22`), a storage test asserting it is *never* persisted
(`session/storage.test.ts:51`), and an unrelated i18n-fallback comment. Nothing silently half-
renders it (e.g. no stray placeholder text, no broken conditional) — the gap is exactly "not built
yet," not a worse, disguised partial build. Mid-review, `docs/plans/salesperson-ui.md` landed
v1.37 (§4.12), granting S13 a narrow, additive edit on `session/SessionContext.tsx`/`api/hooks.ts`
to close this gap — confirming the follow-up is already in motion, and consistent with (a)/(b)
above; the code under review here predates and is unaffected by that grant.

**Verdict: approve with suggestions.** One major (a reachable, previously uncovered branch,
confirmed live via a controlled reproduction — not a suspected/defensive one), and one
minor/open question about a cross-cutting gap that is not S13-specific. No blockers.

**CPG:** not applicable — `cpg_salesperson` does not exist in this FalkorDB instance and was
judged, independently by `teco`, not worth building for this component's size; grounding was done
by direct source/diff reading, the same posture `falkor-chat/docs/reviews/salesperson-ui-s14.md`
used.

## Findings

### Major — `composerNotice.ts`'s `reread`-default branch is untested, and — unlike S14's
### `isOrderLine` precedent — it is genuinely reachable in normal operation, confirmed by a live reproduction

`composerNotice.ts:48-64`'s `case 'reread':` has an inner `switch (reconciliation)` whose
`default:` (line 62-63, text `"We couldn't confirm your message went through — checking…"`) has no
test anywhere: `composerNotice.test.ts`'s three `reread` cases (lines 31-48) only ever pass
`'nothingCommitted'`, `'turnRunning'` or `'turnLost'`, never `null` or an unrecognized value, and
`ChatView.test.tsx` never exercises a `504` on `POST /shop/api/messages` at all. I confirmed the
gap independently: backed up `composerNotice.ts`, mutated the `default` case's return, ran
`npx vitest run` (178/178 still green), restored byte-identical (`diff -q` clean), reconfirmed
green.

I then traced `api/hooks.ts`'s `usePostMessage()` (`hooks.ts:186-224`) to judge reachability rather
than assume it. `onError` (`:205-221`) calls `setAction(resolved)` **synchronously**, then — only
for `resolved.kind === 'reread'` — `await`s `Promise.all([getMessages(...), getState(...)])`
before calling `setReconciliation(...)`. Those two state updates are separated by a real network
round trip, not by a synchronous batch, so React renders at least once in between: with
`action.kind === 'reread'` already set and `reconciliation` still `null` (its initial value, and
the only value it can hold before this code path's own `setReconciliation` call ever fires). That
`null` is not one of the three named `PostMessageReconciliation` values, so it falls to `default`.

This is not a hypothetical — I reproduced it. I added a temporary probe test to `ChatView.test.tsx`
(backed up first, restored byte-identical and reconfirmed 178/178 afterward — `diff -q` clean)
driving a real `504` on `POST /shop/api/messages` with the subsequent `getMessages`/`getState`
reread gated behind a manually-released promise, and asserted the "…checking…" copy renders
*before* the gate is released. **It rendered** — the intermediate state is real, occurs on every
`504`-triggered reread in the running app, and is currently unverified by any test in the suite.
This is the opposite conclusion from `falkor-chat/docs/reviews/salesperson-ui-s14.md`'s
`isOrderLine` precedent (that gap was confirmed *unreachable* given the server's own filtering,
which is why it was rated Minor) — here the branch fires in ordinary degraded-network operation,
which is why this one is Major rather than Minor. The behavior itself is not wrong (the copy reads
sensibly as an in-flight "checking" state), so this is a coverage gap with real regression risk,
not a live bug.

**Suggested improvement:** add one `ChatView.test.tsx` case mirroring the reproduction above (a
`504` on the post, gated `getMessages`/`getState` responses) asserting the "…checking…" text
appears while the gate is held and is gone once released — this also documents, as a test, that
the intermediate state is intentional. A cheaper, complementary addition:
`composerNotice.test.ts` should drive `reconciliation: null` explicitly against a `reread` action
(the actual reachable value, not an invented fourth string) rather than leaving the `default`
branch to whatever the switch falls through to untested.

### Minor / open question — the SPA's own chrome is not yet routed through `react-i18next` anywhere
### outside the join screen's language chooser — cross-cutting, not S13-specific

§4.5 states plainly: "The UI's **own** chrome is localised independently with `react-i18next` (one
JSON bundle per locale)." The only populated bundle key today is `join.languageLabel`/
`join.languageHint` (`src/locales/en.json`), consumed by `i18n/LanguageChooser.tsx` — S12c's own
file. Every other user-facing string I read in this unit's scope — `Composer.tsx`'s
`"Type a message…"`/`"Send"`, `Transcript.tsx`'s `"No messages yet — say hello."`,
`TurnIndicator.tsx`'s `"Thinking…"`/`"…ahead of you."`, `DeadTurnNotice.tsx`'s literal recovery
copy, and every string `composerNotice.ts` returns — is a hardcoded English literal, not a `t()`
call. This is not unique to S13: a spot check of `views/OrderPanel.tsx` and
`components/sheets/*.tsx` (S14/S12b's own subtrees) shows the identical pattern, and
`falkor-chat/docs/reviews/salesperson-ui-s14.md` did not flag it either.

I am not rating this a blocker or major for S13 specifically because no step's row in §5.1 —
S13's included — names "route this view's copy through `t()`" as a done-condition; §4.5's chrome
sentence is stated once at the design-decision level with no step assigned to execute it, which
reads as a gap in the plan's own delegation rather than a defect in any one delivered unit. Given
the same shape already surfaced once this coordination (the welcome-turn ownership gap, resolved
by a small follow-up rather than by re-opening S13), this looks like the same kind of
plan/ownership hole rather than new scope for S13 to absorb unilaterally. **Suggested handling:**
route to `architect`/`teco` as a small ownership/scope question — does full-chrome i18n get its
own follow-up unit (mirroring `S13-welcome-ownership`'s shape), or was §4.5's "own chrome is
localised" always meant to cover only the pieces already wired (join's language step), with FR-3's
localization commitment resting entirely on the server-side reply language? I did not find the
answer in the plan text, which is why this is an open question rather than a finding with a fix
attached.

## What's solid

- **File-ownership discipline held exactly.** The `routes.tsx` diff is the licensed one-line swap
  and nothing more (verified by reading the diff, not the report); the dropped Reset button is
  confirmed relocated to S12b's `ResetControl.tsx`/`ProfileSheet.tsx`, not lost.
- **The welcome-turn gap is handled the right way — documented, not hidden or half-built.**
  `ChatView.tsx`'s own docstring (lines 12-22) names the exact blocking file-ownership fact
  (`session/SessionContext.tsx`/`api/hooks.ts` are S12a's, no S13 arrow existed at delivery time)
  rather than working around it with an unauthorized edit or a silent placeholder.
- **C6a's two independent gates are kept genuinely independent, and tested that way.**
  `ChatView.tsx:38-41`'s `composerDisabled` reads only `turn.state`; `DeadTurnNotice`'s `visible`
  reads only `deadTurnNotice` (`turn.lastTurn === 'failed'`) — `ChatView.test.tsx`'s dead-turn test
  (`:158-197`) asserts the input is *not* disabled while the notice is showing, the harder and more
  specific assertion than just checking the notice renders.
- **`TurnIndicator`'s `queuePosition === 0` branch is tested at both the unit level
  (`TurnIndicator.test.tsx:24-28`) and through the full `ChatView` wiring
  (`ChatView.test.tsx:138-156`)** — the exact mutation shape (`queuePosition && …`) the plan's own
  §5.2 text calls out as the bug this guards against is covered twice, not assumed once.
  `no-markup` (agent-emitted text) is likewise tested at both the `MessageBubble` unit level and
  through the full `ChatView` render.
- **The autoscroll-when-at-bottom logic is tested for both directions** — pulled down when the
  viewer was at the bottom, *not* yanked down when they had scrolled up — via a real `scroll`
  listener driven through `fireEvent.scroll`, not just asserted from reading the component.
- **The optimistic-send design has no second source of truth.** `withOptimisticRow` renders the
  mutation's own already-server-assigned row rather than a client-fabricated id, and stops the
  moment the poll catches up by `msgId` — confirmed by reading `optimistic.ts` and its test file,
  which covers the append, the dedupe, and the no-op cases explicitly.

## Open questions

- The i18n-chrome gap above: does it get its own follow-up unit, or was full-chrome localization
  never actually in scope for any delivered step? Routed to `architect`/`teco`, not blocking S13's
  acceptance.
