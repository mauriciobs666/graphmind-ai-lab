# Review: salesperson-ui S17 (Chrome i18n sweep) — implementation

> **Status:** active · **Owner:** `analyst` · **Tracks:** S17 (`docs/plans/salesperson-ui.md` §4.13/§5.0/§5.1, v1.39)

## Scope & verdict

Reviewed the uncommitted `salesperson` working-tree diff implementing S17 (the chrome-i18n sweep)
against `docs/plans/salesperson-ui.md` §4.13 (v1.39, read in full) and the two-pass design gate
`falkor-chat/docs/reviews/salesperson-ui-s17.md`. Baseline: `git diff`/`git status` in the working
tree (35 files: 18 source + 16 test + 3 locale + 1 new test file — `salesperson/src/layout/
Header.test.tsx`), read file-by-file in full, not sampled. Cross-checked against
`salesperson/AGENTS.md`'s file-ownership table and `docs/plans/salesperson-ui2-coordination.md`'s
ledger row for this unit.

**Verdict: approve.** Every one of the 17 files §4.13's tables name is swept correctly; the two
design-gate Blocker-fix keys (`chat.notice.messageNotSent`, `chat.transcript.label`) are wired and
independently tested, one of them end-to-end through the real component tree, not just the unit
that originally missed it. All three locale bundles carry an identical 77-key set (verified by
flattening and diffing them directly, not by trusting `i18n/locales.test.ts`'s own report). The
cognate-exception design (`cart.total`/`order.total`/`layout.header.brand`) is not only correctly
scoped to Layer-1-only per the plan, several of its tests go further and positively assert the
untranslated value survives a `pt-BR` switch — verified for real by an independent mutation (see
Findings). File ownership is respected exactly: nothing outside the five granted locations was
touched, `routes.tsx`/`views/Presenter*`/`src/i18n/**`/`package.json` all show zero diff. `npx tsc
-b` and `npx vitest run` reproduced clean (23 files / 219 tests), matching both the implementer's
and teco's reports.

**CPG:** considered, not relevant — `cpg_salesperson` does not exist in this FalkorDB instance (per
the brief and the design review's own posture); this is a source-tree review with the actual
17-file diff to ground against directly, not a graph-shaped question.

## Findings

### None blocking, none major

No blockers or majors found. Two minor/informational notes below, neither gating.

### Minor — a handful of dynamic branches remain untested, correctly disclosed as pre-existing

`CartPanel.tsx`'s `cart.staleNotice`, `OrderPanel.tsx`'s `order.staleNotice`/`order.error.
staleOrderRefresh`/`order.error.reread`, and `ProfilePanel.tsx`'s `profile.staleNotice` have no
test driving their branch (a background-refetch race, or a 404/409/504-specific advance response).
I confirmed this is genuinely pre-existing, not introduced or worsened by S17: `git show
HEAD:salesperson/src/{views/CartPanel.test.tsx,views/OrderPanel.test.tsx,views/ProfilePanel.test.tsx}`
grepped for `stale`/`reconnecting`/`staleOrderRefresh`/`reread` returns zero hits in all three —
these branches were untested before the sweep too. §4.13 never commits S17 to adding behavioural
harnesses (explicitly "no behavioural or structural change"), so this is not a gap this unit owns
to close; each affected key is covered by `i18n/locales.test.ts`'s key-coverage check (Layer 1) and
is a direct, literal `t(...)` call at its production site — the implementer's own comments in each
test file say exactly this, and it checks out. Notably, `OrderPanel.tsx`'s `order.error.unhandled`
branch, which was *also* untested pre-sweep (confirmed: no `Unexpected response`/`unhandled`/`500`
hit in `HEAD`'s copy), got a **new** test in this diff (`OrderPanel.test.tsx`'s "routes the
threaded errorMessageFor() unhandled-status copy" case) — the implementer closed one of these gaps
opportunistically rather than leaving it. **No action needed**; noting only so a future S17-shaped
sweep doesn't need to re-derive that this was checked.

### Nit — `CartPanel.tsx`'s formatting of two short returns is slightly inconsistent with its neighbors

`CartPanel.tsx`'s loading/empty branches now wrap a single short `<p>…</p>` across three lines
(`return (\n  <p>…</p>\n);`) where `ProfilePanel.tsx`'s equivalent branch does the same but
`CartPanel.tsx`'s own `cart.empty` branch stays a one-liner (`return <p>…</p>;`). Purely cosmetic,
consistent with the project having no wired formatter gate (`salesperson/AGENTS.md`: oxlint is not
wired into the build) — not worth a follow-up commit on its own.

## Independent verification performed

- **Reproduced clean build/test**: `npx tsc -b` (0 errors) and `npx vitest run` → 23 files / 219
  tests passed, matching both the implementer's and teco's reports exactly.
- **Locale key-parity, computed directly** (not trusting `locales.test.ts`'s own pass/fail): flattened
  all three bundles in Python — 77 leaf keys each, zero missing/extra in either direction.
- **Independent mutation 1 — cognate-exception content, not just presence** (outside both teco's
  and the implementer's own mutation tables, per the brief's specific ask): mutated `pt-BR.json`'s
  `layout.header.brand` from `"Storefront"` to `"Loja Virtual"`. `Header.test.tsx`'s own
  `findByText('Storefront')` assertion (written to prove the cognate stays untranslated) reddened
  correctly — 1 of 7 tests failed in `Header.test.tsx`/`Shell.test.tsx`. This is a genuine positive
  finding: the implementer's tests assert the cognate's *exact value*, not merely that the key
  exists, so a regression in a "should stay untranslated" key would **not** go unnoticed, despite
  the plan only requiring Layer-1 (key-presence) proof for these keys. Restored via `cp` from a
  `test -s`-verified backup, confirmed `diff -q` byte-identical, full suite re-confirmed 219/219.
- **Independent mutation 2 — interpolation-key case mismatch** (an axis neither teco's nor the
  implementer's own tables touched): mutated `pt-BR.json`'s `order.idLabel` from `"Pedido nº
  {{orderId}}"` to `"Pedido nº {{orderid}}"` (i18next interpolation is case-sensitive).
  `OrderPanel.test.tsx` reddened correctly — the `t('order.idLabel', { orderId: order.orderId })`
  call site passes `orderId`, not `orderid`, so the placeholder went unresolved and `screen.
  getByText('Pedido nº ord-1')` failed to find a match. Restored and re-verified byte-identical,
  full suite green.
- **File-ownership scope**: `git diff --stat -- salesperson/src/i18n salesperson/src/routes.tsx
  salesperson/src/views/Presenter\* salesperson/package.json salesperson/AGENTS.md
  docs/HISTORY.md` returns empty for every one — confirmed zero touch outside the five granted
  locations (`src/layout/**`, `src/components/sheets/**`, `src/locales/**`, `src/views/Chat*` +
  `src/components/message/**`, `src/views/{Cart,Order,Profile,Catalog}*`).
- **File count**: all 17 files §4.13's tables name appear in the diff (6 `chat.*` + 7 `layout.*` +
  4 panel files), plus the 3 locale files and the one new `Header.test.tsx` — matching the design
  review's own independently-confirmed "seventeen, not thirteen" count.
- **Design-gate defect closure, read directly against the diff, not inferred from the plan text
  alone**: `composerNotice.ts:53`'s `nothingCommitted` branch now calls `t('chat.notice.
  messageNotSent')`, value matches exactly, and is exercised twice — once in
  `composerNotice.test.ts`'s new pt-BR pass, once end-to-end in `ChatView.test.tsx`'s new
  "threads t into composerNoticeFor()" test (a 504 → `reread`/`nothingCommitted` sequence rendered
  through the real component tree). `Transcript.tsx:53`'s `aria-label="Transcript"` now calls
  `t('chat.transcript.label')`, tested in both `Transcript.test.tsx` and `Shell.test.tsx`'s (via
  `ChatView.test.tsx`) assertions.
- **`composerNotice.ts`'s threaded-parameter design**: `composerNoticeFor(action, reconciliation,
  t)`'s signature matches §4.13 exactly, including the `import type { TFunction } from 'i18next'`
  line; `ChatView.tsx` is confirmed the only production call site
  (`grep -rn composerNoticeFor salesperson/src`).

## What's solid

- **Every one of the 17 tabled files matches its design-table English value character-for-character**
  — verified by direct diff reading, not sampled: `Composer.tsx`, `Transcript.tsx`,
  `TurnIndicator.tsx` (including the `_one`/`_other` plural pair, both forms exercised in its test),
  `DeadTurnNotice.tsx`, `MessageBubble.tsx`, `composerNotice.ts`, `Header.tsx`, `BottomSheet.tsx`,
  the four `*Sheet.tsx` wrappers, `ResetControl.tsx`, `CartPanel.tsx`, `OrderPanel.tsx` (incl. the
  `t(\`order.status.${order.status}\`, statusLabel(order.status))` default-value shorthand, wired
  exactly as designed), `ProfilePanel.tsx`, `CatalogPanel.tsx`.
- **No structural or behavioural change** anywhere in the swept files — every diff hunk is a
  string-for-`t()`-call substitution plus the `useTranslation()` import/hook-call, confirmed by
  reading every hunk; `TurnIndicator.tsx`'s `text` const and branch structure are untouched, as the
  plan's "no structural change" promise requires.
- **Locale-switch test coverage (§4.13 Layer 2) is thorough and, in several places, stricter than
  the plan's own minimum** — every swept file gets a dedicated pt-BR-switch test with the
  negative-literal assertion, and the cognate keys get a positive content assertion the plan didn't
  strictly require (see Independent mutation 1, above).
- **`ChatView.test.tsx`'s new `useTranslation()`-import obligation** (the design review's Minor,
  closed in v1.39's plan text) is honored — the import is present, and the file goes further than a
  bare import: it adds a genuine composed-chrome end-to-end test exercising `Composer`+`Transcript`+
  the threaded `composerNoticeFor` together under `pt-BR`.
- **`git status` after review is clean of any residue from my own mutation testing** — both mutated
  files were restored byte-identical and reconfirmed via `diff -q` before moving on.

## Open questions

None. The two Minor/Nit items above are informational, not blocking, and match precedent already
set elsewhere in this coordination (S14's own non-blocking untested-branch findings were logged,
not re-dispatched).
