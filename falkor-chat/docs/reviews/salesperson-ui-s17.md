# Review: salesperson-ui S17 (Chrome i18n sweep) — plan

> **Status:** active · **Owner:** `analyst` · **Tracks:** S17 (`docs/plans/salesperson-ui.md` §4.13/§5.0/§5.1, v1.38)

## Scope & verdict

Reviewed the uncommitted v1.37 → v1.38 amendment to `docs/plans/salesperson-ui.md` (new §4.13,
designing step **S17** — sweeping S12b's/S13's/S14's hardcoded English chrome through
`react-i18next`) plus the matching `salesperson/AGENTS.md` ownership-table diff. This is a **plan**
review, not an implementation review — S17 has not been dispatched. Baseline: `git diff --
docs/plans/salesperson-ui.md` and `git diff -- salesperson/AGENTS.md` in the working tree
(uncommitted, ~330 + 20 lines), read in full, plus `docs/plans/salesperson-ui2-coordination.md`'s
`## RESUME HERE` section for context. Verified by direct reading of all 17 source files §4.13
names (not sampled), by running the plan's own prescribed "one-time static residual check" grep
verbatim against the pre-sweep tree, by tracing every production call site of `composerNoticeFor`,
and by reading the installed `i18next@26.4.1`'s plural-resolver source.

**Verdict: needs changes.** Two Blocker-class completeness gaps in the plan's own "derived, not
sampled" string inventory (one of which defeats the plan's own stated safety net for finding such
gaps), plus one Major gap in the test design's false-fail exposure. The design's other four axes
(namespacing, `composerNotice.ts`'s `t`-threading, plural-suffix handling, file-ownership/
sequencing) are sound and verified.

**CPG:** not applicable — `cpg_salesperson` does not exist in this FalkorDB instance (confirmed by
`teco` for the S13 review; same posture holds here), and this is a design-document review with a
concrete source tree to ground against directly, not a graph-shaped question.

## Findings

### Blocker — `composerNotice.ts`'s `nothingCommitted` branch is missing from the key table

`composerNotice.ts:48-64`'s `reread` switch has **four** text-bearing branches
(`nothingCommitted`, `turnRunning`, `turnLost`, the `default`), each already exercised by its own
test in `composerNotice.test.ts` (lines 31-35 name it explicitly: `"C4 — reread + nothingCommitted
reads as safe to retry"`, asserting `/not sent/i` against the literal `'Your message was not sent.
Please try again.'`). §4.13's `chat.notice.*` table (diff lines 98-106) lists only **three** of the
four — `sentAwaitingReply` (turnRunning), `replyUnconfirmed` (turnLost), `checkingDelivery`
(default) — and has no row at all for `nothingCommitted`'s text. This directly falsifies §4.13's
own claim, repeated three times, that the table was "confirmed by direct reading of all thirteen
files" / "derived by reading every one of the thirteen files, not sampled." Because both of S17's
completeness gates (Layer 1's key-coverage test and Layer 2's per-component locale-switch tests)
are keyed off this same table, this branch will ship permanently hardcoded and **no layer of the
plan's own test design will ever flag it** — not a coverage gap that surfaces later, a silent one.
**Fix:** add a `chat.notice.messageNotSent` (or similar) row citing `composerNotice.ts:53` and
`composerNotice.test.ts:31-35` before S17 dispatches.

### Blocker — the plan's own "real detector, not a vacuous one" residual-check regex catches 4 of ~40 table entries

§4.13/S17's row claims the JSX-text-child grep (`>[ \t]*[A-Za-z][^<{}]*[A-Za-z][ \t]*<`) is
"confirmed by this section's own reading to catch every string in the tables above when run
against the pre-sweep tree, so it is a real detector, not a vacuous one." I ran it verbatim (GNU
grep, `-nE`) against all 17 named files. It matched exactly **4** of the roughly 40 JSX-text-child
entries the tables list: `<span>Total</span>` (×2, `CartPanel.tsx:65`/`OrderPanel.tsx:188`) and
`ProfilePanel.tsx`'s two `<dt>` labels. Every multi-line JSX text block — which is most of the
actual sentences, since this codebase's Prettier-style formatting puts long text on its own line,
separate from the surrounding tags — is invisible to a line-based grep with no multiline flag.
Even single-line cases fail once the text ends in punctuation: the pattern's tail,
`[A-Za-z][ \t]*<`, requires a **letter** immediately (optionally followed by whitespace) before
`<`, so `>Your cart is empty.<` (ending in `.`) never matches. See the Appendix for the exact
commands and full output. The prop-oriented half of the same check
(`aria-label=`/`placeholder=`/`title=`) works correctly and is not in question. **Fix:** either
repair the regex (e.g. `grep -Pzo` for real multiline matching, and relax or drop the trailing
`[A-Za-z]` requirement to tolerate closing punctuation/ellipses), or drop the "confirmed... real
detector" claim and state plainly that Layer 2's committed tests are the only reliable completeness
gate — telling a future implementer this hand-run check is a working backstop when it demonstrably
is not is worse than not having the claim at all.

### Blocker — `Transcript.tsx`'s `aria-label="Transcript"` is missing from the key table

`components/message/Transcript.tsx:53` carries `aria-label="Transcript"` on the scrollable `<ul>`
— a genuine, language-bearing accessible name, structurally identical to `layout.header.nav`'s
`aria-label="Shop"` (which the table *does* capture). §4.13's `chat.*` table names only
`chat.transcript.empty` for this file. This is a second concrete hole in the "derived by reading
every one of the thirteen files, not sampled" claim — found via the working (prop-based) half of
the plan's own residual grep, which correctly flags it once run, but the table itself never named a
key for it. **Fix:** add `chat.transcript.label` (or similar) citing `Transcript.tsx:53`.

### Major — the locale-switch negative assertion has an unaddressed false-fail mode for cognates/brand names

Layer 2's per-component test pattern (`queryByText(<English literal>).not.toBeInTheDocument()`
under `'pt-BR'`) is applied uniformly to "every one of the thirteen files" with no stated
exception. `cart.total`/`order.total`'s English value is `"Total"` — which is also the correct,
faithful Portuguese *and* Spanish word for the same concept (a genuine cross-language business-
receipt cognate); `layout.header.brand`'s value `"Storefront"` is a brand name a translator would
plausibly and correctly leave unchanged in all three locales. §4.13 states plainly that "pt-BR/es
values are S17's own translation work, not specified here" — meaning nobody has yet committed to
values that are guaranteed to differ from the English literal. A delegate who translates
`cart.total` *correctly* to pt-BR as `"Total"` will see the prescribed test fail against fully
correct, properly-`t()`-wired code, since the literal is still genuinely on screen (rendered from
the pt-BR bundle, not the `fallbackLng` path the plan already reasons about correctly for the
*missing-translation* case). This is exactly the "check keyed on something the unit invented"
failure shape from the other direction: not a check too weak to catch a real bug, but one that can
fail on **correct** code, inviting a delegate to "fix" it by picking a needlessly non-cognate
pt-BR/es wording just to satisfy the test. **Fix:** name `cart.total`/`order.total`/
`layout.header.brand` (and instruct implementers to flag any other candidate cognate they hit) as
verified by Layer 1 (key-coverage / call-site presence) only, not by the disappearing-literal
assertion.

### Minor — "thirteen files" undercounts the table's own contents by 4

§4.13/S17's row says "thirteen files" six times across the diff. The tables actually name **17**
distinct files: 6 in `chat.*` (`Composer.tsx`, `Transcript.tsx`, `TurnIndicator.tsx`,
`DeadTurnNotice.tsx`, `MessageBubble.tsx`, `composerNotice.ts`), 7 in `layout.*`
(`Header.tsx`, `BottomSheet.tsx`, `CartSheet.tsx`, `CatalogSheet.tsx`, `OrderSheet.tsx`,
`ProfileSheet.tsx`, `ResetControl.tsx`), 4 in `cart/order/profile/catalog`
(`CartPanel.tsx`, `OrderPanel.tsx`, `ProfilePanel.tsx`, `CatalogPanel.tsx`). Not itself evidence of
missing files (I independently confirmed all 17 listed files are the correct, complete set of
chrome-bearing files in the five subtrees — `Shell.tsx`/`icons.tsx`/`SheetContext.tsx` correctly
carry no user-facing text and are correctly left out), but it undercuts confidence in "not sampled,
derived by reading," and the residual-check instruction literally tells a future reader to check
the wrong file count. **Fix:** correct the count throughout.

### Minor — `ChatView.tsx`/`ChatView.test.tsx`'s new `useTranslation()` obligation isn't named alongside the tables

§4.13's `composerNotice.ts` section commits `ChatView.tsx` to `const { t } = useTranslation();` —
a new `react-i18next` call site — but `ChatView.tsx` never appears as a row in any of the three key
tables (it owns no key of its own, only threads `t` through). The generic Layer 2 sentence ("every
test file that renders a component newly calling `useTranslation()`/`t()`... needs the import
added") technically covers `ChatView.test.tsx` (confirmed: it currently has no `i18n`/
`useTranslation` import at all), but a delegate skimming the per-file tables for "what needs a test
update" could miss a file that never appears in any of them. **Fix:** one explicit sentence in the
`composerNotice.ts` subsection naming `ChatView.test.tsx` under the same import rule.

## What's solid

- **File-ownership/sequencing (review item 6):** the three subtrees are genuinely disjoint
  (confirmed by direct directory listing — no overlap among `layout/**`+`components/sheets/**`,
  `views/Chat*`+`components/message/**`, `views/{Cart,Order,Profile,Catalog}*`), so "one unit, not
  three" is correctly driven by the shared `locales/*.json` files alone, as claimed. The stated
  dependency order — after S12b, S13 (incl. its still-in-flight fix-back and the still-undispatched
  welcome-turn follow-up), S14; before S15 — is consistent with `git log` (S13's only commit is
  `acd0413`) and with the coordination doc's `## RESUME HERE`.
- **`i18next` plural-suffix claim (review item 3), verified against the installed package:**
  `node_modules/i18next/dist/cjs/i18next.js:1088-1093`'s `PluralResolver.getRule` calls
  `new Intl.PluralRules(cleanedCode, {type})` — genuine per-locale CLDR data, not a hardcoded
  English binary rule. `en`/`pt-BR`/`es` all resolve to the same two-category `{one, other}` CLDR
  class, so the `_one`/`_other` suffix pair the plan specifies is sufficient for all three locales
  without further design work; the brief's worry that it might assume English's binary system
  applies universally does not hold here, though only because none of these three locales needs a
  third category.
- **`composerNotice.ts`'s `t`-threading design (review item 2):** verified `ChatView.tsx` really is
  the only production call site (`grep -rn composerNoticeFor salesperson/src`), the
  rejected-ambient-singleton alternative is reasoned correctly against the file's own stated
  mutation-testing-cheapness constraint (`composerNotice.ts:1-9`), and `TFunction` is a real,
  importable type from the installed `i18next@26.4.1` (`node_modules/i18next/index.d.ts:19`).
- **S12d's and `routes.tsx`'s scope exclusions (review item 4):** S12d's swept row is a concrete,
  binding commitment ("from its first line of code," naming the `presenter.*` namespace under
  §4.13's own convention), not vague future intent; R13's reversal trigger for `routes.tsx` is
  actionable.
- **String-table accuracy where checked:** every English value cited for `CartPanel.tsx`,
  `OrderPanel.tsx`, `ProfilePanel.tsx`, `CatalogPanel.tsx`, `ResetControl.tsx`, `Header.tsx`,
  `BottomSheet.tsx`, the four `*Sheet.tsx` wrappers, `Composer.tsx`, `TurnIndicator.tsx`,
  `DeadTurnNotice.tsx`, and `MessageBubble.tsx` was checked string-for-string against the source
  and matches exactly; the stated exclusions (`EM_DASH`, server-sourced product/order/category
  data) are correctly drawn.

## Open questions

None blocking — the three completeness gaps and the false-fail test-design gap are concrete enough
to fix without further stakeholder input; they're architect's to close in a v1.39 pass before S17
dispatches.

## Pass 2 — 2026-09-14

Scope: v1.38 → v1.39 (`git diff -- docs/plans/salesperson-ui.md`, now diffed against the committed
v1.37 baseline — v1.38 never landed as its own commit). Verified all six of `architect`'s claimed
fixes independently against the current diff and the same source tree used in Pass 1.

**Verdict: approve with suggestions.** All three Blockers and the Major are genuinely closed,
correctly and at the root cause in two cases, not just the symptom. One new Minor surfaced from
teco's own follow-up questions — not blocking.

**Disposition of Pass 1 findings:**

1. **Blocker (`composerNotice.ts`'s `nothingCommitted` missing) — fixed.** `chat.notice.messageNotSent`
   added, value `"Your message was not sent. Please try again."` matches `composerNotice.ts:53`
   exactly, citation to `composerNotice.test.ts:31-35` correct.
2. **Blocker (residual-check regex catches 4/~40) — fixed, verified independently.** See "New
   checks" below — re-ran the corrected pattern myself rather than trusting the reported count.
3. **Blocker (`Transcript.tsx`'s `aria-label="Transcript"` missing) — fixed.** `chat.transcript.label`
   added, citing `Transcript.tsx:53`, value correct.
4. **Major (cognate/brand-name false-fail) — fixed.** New exception clause names the exact two keys
   I flagged (`cart.total`/`order.total` = `"Total"`, `layout.header.brand` = `"Storefront"`),
   correctly scopes them to Layer-1-only verification, and generalizes with a "flag any other
   cognate the same way" clause. See "New checks" below for the one residual note on this clause.
5. **Minor ("thirteen" undercounts by 4) — fixed, at the root.** `grep -n thirteen` on the current
   file returns only one hit, and it's unrelated pre-existing text (line ~1871, the C1-C14 rule
   count, nothing to do with §4.13). All six i18n-sweep-related sites now read "seventeen." Beyond
   the find-and-replace, §5.1's S17 row now **cites** §4.13's test design instead of restating it —
   mirroring the plan's own pre-existing S12a-row precedent (§5.3 cited, not restated) — so the file
   count exists in exactly one place going forward, closing the "two places, one drifts" root cause
   rather than only the symptom I originally flagged.
6. **Minor (`ChatView.test.tsx`'s import obligation unnamed) — fixed.** New sentence names it
   explicitly, correctly states it today has no `i18n`/`useTranslation` import (re-confirmed), and
   correctly explains why it never appears as a table row (owns no key of its own).

**New checks, per teco's specific asks:**

- **Over-matching / false positives in the corrected regex** (`grep -Pzo '>[^<>{}]*[A-Za-z][^<>{}]*<'`):
  I ran it myself, per file, against all 17 files, counting matches by NUL-record (not naive
  line-counting — see the Appendix for why that matters: a careless count gives 85, not the real
  number). **Result: 35 raw matches, 34 genuine chrome strings and exactly one false positive** — a
  multi-line comment-bleed in `CatalogPanel.tsx`'s file-header comment (a stray `>`/`<` inside
  backtick-quoted code spanning three `//` lines). This is a real instance of exactly what v1.39's
  own new text warns about ("it can produce false-positive noise from a nearby comment") — confirmed
  accurate, and easy for a human doing the now-explicit "hand-eyeballed sanity pass" to recognize as
  comment noise (the matched text visibly contains prose about "the manifest builder," not UI copy).
  Also tested: running the check against multiple files in **one** invocation (rather than a
  per-file loop) does **not** bleed matches across file boundaries — GNU grep resets its buffer per
  input file even under `-z`; confirmed directly. No new correctness risk found.
- **Disclosure accuracy/completeness** (does the plan honestly state what the check still can't
  see?): the two named blind-spot examples (`TurnIndicator.tsx`'s `firstInLine`/`aheadOfYou_*` pair,
  `composerNotice.ts`'s values) are verified real — both produced **zero** matches for those specific
  strings in my run. More importantly, the plan's *general* rule ("a string assigned to a JS
  variable/ternary/template before reaching `{…}`... is structurally invisible to this pattern
  regardless of fix") is not just accurate for the two named cases — I independently found it also,
  correctly, predicts two **further**, unnamed blind spots: `ResetControl.tsx`'s four
  `errorMessageFor()` branches plus its `{resetMine.isPending ? 'Resetting…' : 'Yes, reset'}` ternary,
  and `OrderPanel.tsx`'s three `errorMessageFor()` branches — all already correctly keyed in the
  table (verified in Pass 1), just also invisible to Layer 3, exactly as the stated general rule
  predicts. The disclosure is accurate and, as a *rule* rather than an exhaustive list, sufficient —
  no gap found here.
- **Does the cognate exception weaken coverage for anything not on the list?** The carve-out is
  textually scoped to two named keys plus an open "flag it the same way" clause, with no explicit
  instruction that a future claimed cognate needs independent verification before being exempted
  from the negative assertion. In practice this is low-risk — S17's own implementation will get the
  usual independent review gate, which would naturally scrutinize any newly-claimed cognate the same
  way it scrutinizes everything else in the diff — but the plan text doesn't say so, so it's a thin
  spot rather than a closed one. **Minor, not blocking**: worth one sentence in a future pass, or
  simply a note the S17 gate reviewer carries by convention; not worth reopening v1.39 for.
- **Does the cited-not-restated S17 row still stand alone?** Yes — it mirrors the plan's own
  pre-existing S12a-row citation precedent (§5.3, C1-C14), same document, and explicitly flags by
  name that "named cognate exceptions" exist even without restating them, so a reader isn't left to
  guess that detail is there. No information lost relative to the old restated version.

## Appendix — Pass 2: corrected regex re-verification

Per-file match counts, `/usr/bin/grep -Pzo '>[^<>{}]*[A-Za-z][^<>{}]*<' <file>`, counted by NUL
record (`| grep -zac ''`) — **not** by piping through `tr '\0' '\n' | wc -l`, which over-counts:
a single real match commonly spans an embedded newline (JSX text on its own line between tags), so
naive newline-counting on the converted stream gave 85 on a first attempt, and neither figure
matches teco's independently-reported 62. I could not reproduce 62 with any counting method I
tried; it doesn't change the verdict (the check is explicitly a hand-run, non-gating aid — see
§4.13's own "not a pass/fail gate" language), but the discrepancy is unresolved and worth a note if
this number is ever cited as a fact rather than an illustration.

```
components/message/Composer.tsx: 1     components/sheets/BottomSheet.tsx: 0
components/message/Transcript.tsx: 1    components/sheets/CartSheet.tsx: 0
components/message/TurnIndicator.tsx: 1 components/sheets/CatalogSheet.tsx: 0
components/message/DeadTurnNotice.tsx: 1 components/sheets/OrderSheet.tsx: 0
components/message/MessageBubble.tsx: 0 components/sheets/ProfileSheet.tsx: 0
components/message/composerNotice.ts: 0 components/sheets/ResetControl.tsx: 5
layout/Header.tsx: 1                    views/CartPanel.tsx: 5
                                         views/OrderPanel.tsx: 10
                                         views/ProfilePanel.tsx: 5
                                         views/CatalogPanel.tsx: 5
TOTAL: 35
```

The one false positive (`views/CatalogPanel.tsx`, inside its own file-header comment, not JSX):

```
>` or `null` — the manifest builder
// (`falkor-chat` §4.7) has already resolved presence/absence server-side, so
// this file's only job is to render exactly one of the two card shapes per
// product, never an `<
```

## Appendix — Pass 1: residual-check regex verification

Commands run (GNU grep, not the environment's `ugrep`-backed `grep` shell function, which mishandles
`-E`+`-G` and either silently returns wrong results or errors "conflicting matchers specified"):

```
cd salesperson/src
/usr/bin/grep -nE '>[ \t]*[A-Za-z][^<{}]*[A-Za-z][ \t]*<' \
  components/message/Composer.tsx components/message/Transcript.tsx \
  components/message/TurnIndicator.tsx components/message/DeadTurnNotice.tsx \
  components/message/MessageBubble.tsx components/message/composerNotice.ts \
  layout/Header.tsx components/sheets/BottomSheet.tsx components/sheets/CartSheet.tsx \
  components/sheets/CatalogSheet.tsx components/sheets/OrderSheet.tsx \
  components/sheets/ProfileSheet.tsx components/sheets/ResetControl.tsx \
  views/CartPanel.tsx views/OrderPanel.tsx views/ProfilePanel.tsx views/CatalogPanel.tsx
```

Output (all 4 matches, out of ~40 JSX-text-child table rows):

```
views/CartPanel.tsx:65:        <span>Total</span>
views/OrderPanel.tsx:188:        <span>Total</span>
views/ProfilePanel.tsx:42:        <dt className="font-medium text-slate-500 dark:text-slate-400">Name</dt>
views/ProfilePanel.tsx:44:        <dt className="font-medium text-slate-500 dark:text-slate-400">Delivery address</dt>
```

The prop half of the same check, for comparison (works correctly, 10 real matches including the
un-tabled `Transcript.tsx` aria-label):

```
components/message/Composer.tsx:65:          placeholder="Type a message…"
components/message/Composer.tsx:72:          aria-label="Send"
components/message/Transcript.tsx:53:      aria-label="Transcript"
layout/Header.tsx:30:        <nav aria-label="Shop" className="flex items-center gap-1">
components/sheets/BottomSheet.tsx:48:        aria-label="Close"
components/sheets/BottomSheet.tsx:70:            aria-label="Close"
components/sheets/CartSheet.tsx:8:    <BottomSheet open={openSheet === 'cart'} title="Cart" onClose={close}>
components/sheets/CatalogSheet.tsx:8:    <BottomSheet open={openSheet === 'catalog'} title="Catalog" onClose={close}>
components/sheets/OrderSheet.tsx:8:    <BottomSheet open={openSheet === 'order'} title="Order status" onClose={close}>
components/sheets/ProfileSheet.tsx:9:    <BottomSheet open={openSheet === 'profile'} title="Profile" onClose={close}>
```
