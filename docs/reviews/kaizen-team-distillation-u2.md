# Kaizen-team distillation — U2 (`frontend-engineer`, 8-entry inbox)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (`docs/plans/kaizen-team-distillation-coordination.md`, unit U2)

## Scope & verdict

Reviewed the uncommitted working-tree diff produced by `cobb` distilling all 8
`frontend-engineer`-produced raw `:KaizenEntry` nodes in the shared `kaizen_team` FalkorDB graph
(per `skills/agent-maintenance/SKILL.md` §5), against: (1) the actual repo code each promoted/
discarded claim cites (file paths, line content, instance counts), (2) the receiving files'
existing content for placement/duplication and cross-consistency, (3) `falkor-chat/docs/reviews/
salesperson-ui-s12b.md` and `docs/plans/salesperson-ui.md` §4.11 for the one discard, and (4) an
independent re-query of `kaizen_team`. Six changed files plus one new one:
`claude/frontend-engineer/frontend-quirks.md` (new), `claude/frontend-engineer/frontend-engineer.md`,
`claude/frontend-engineer/kaizen/history.md`, `claude/frontend-engineer/kaizen/plan.md`,
`claude/README.md`, `claude/AGENTS.md`. I did not evaluate
`docs/plans/kaizen-team-distillation-coordination.md` itself, only used it for scope confirmation.

**Verdict: approve with suggestions.** No blocker, no major. All 7 promoted facts are true and
correctly cited against the live tree; the 1 discard is correctly verified as fixed and already
documented elsewhere; the KB-vs-project-docs override is sound; the catalog/roster pointers agree
with each other and with the graph state is confirmed empty. Two minor accuracy gaps in the new
KB file and two nits on pattern-fidelity and placement — none block promotion.

**CPG: not applicable — this unit edits only agent/skill prompts and knowledge-base markdown, no
code-level component.**

## Findings

### Minor — `frontend-quirks.md`'s TS2312 entry miscounts the converted type aliases (9 claimed, 10 actual)

`claude/frontend-engineer/frontend-quirks.md:19-25` says `salesperson/src/api/hooks.ts` has
"`UseShopStateResult`, `UseMessagesResult`, `UsePostMessageResult`, and 6 more) — 9 interfaces
converted." Actual count in the file: 10 exported type aliases matching
`UseQueryResult<...> & {...}` / `UseMutationResult<...> & {...}` — `UseShopStateResult`,
`UseMessagesResult`, `UsePostMessageResult`, `UseCatalogResult`, `UseAdvanceOrderResult`,
`UseResetMineResult`, `UseJoinResult`, `UsePresenterLoginResult`,
`UsePresenterParticipantsResult`, `UsePresenterResetAllResult` (verified by `grep -n "^export type
Use.*Result" salesperson/src/api/hooks.ts`). The underlying technical fact (TS2312 on
`interface extends`, fixed via a type-alias intersection) is correct and demonstrated throughout
the file — only the instance count is off by one. Suggested improvement: change "9 interfaces
converted" to "10" and "and 6 more" to "and 7 more" in `frontend-quirks.md:25`.

### Minor — the KB's "each entry names the version it was verified against" promise is unmet for most of the 7 entries

`frontend-quirks.md:10-11`'s header commits to a re-verification discipline: "**Re-verify an
entry before relying on it if the underlying library's major version has since changed** (each
entry names the version it was verified against)." In practice only the TS2312 entry names one
explicitly (`@tanstack/react-query v5`); the `resolveJsonModule` entry, both RTL/Vitest entries,
the CSS/Chromium entry, and the i18next entry cite no library version at all, even though
`salesperson/package.json` pins them precisely (`i18next ^26.4.1`, `vite ^8.2.2`, `typescript
~6.0.2`, `@testing-library/react ^16.3.3`, `vitest ^4.1.11`). Without a stated version, a future
reader has no way to tell whether a "major version" bump has happened, defeating the stated
re-verify trigger. Suggested improvement: add the pinned version to each entry (or, if some
entries genuinely don't hinge on a single dependency's version — e.g. the seed-placeholder
pattern, which is a lab convention, not a library fact — narrow the header's blanket claim to say
so explicitly, the way `falkordb-quirks.md`'s own header scopes its promise to the pinned engine
build).

### Nit — `frontend-quirks.md`'s "see README.md for the catalog entry" pointer line isn't part of the pattern it claims to mirror

`frontend-quirks.md:13` ("`frontend-engineer.md` points here; see [`README.md`](../README.md) for
the catalog entry.") has no counterpart in `claude/graph-dba/falkordb-quirks.md` (the file this KB
explicitly models itself on) nor in the other two KBs created by this same distillation exercise's
sibling units, `claude/analyst/review-techniques.md` and `claude/qa-engineer/qa-testing-techniques.md`
— none of the three add a README backlink. Harmless and not incorrect, just an unannounced
addition to a pattern described as mirrored. No action required unless a future pass wants the KB
shape to converge exactly.

### Nit — the new blockquote in `frontend-engineer.md` sits next to the wrong section

`claude/frontend-engineer/frontend-engineer.md:53-57`'s new KB pointer is placed immediately after
the unrelated "Python-native UIs" / Streamlit subsection (`:50-51`) rather than adjacent to
"JavaScript / TypeScript & frameworks" (`:32-36`), which is what the KB's content (React/TS/Vite/
TanStack Query/i18next) actually belongs next to. Reads a little disjointed on a first pass.
Suggested improvement: move the blockquote up to directly follow the JS/TS & frameworks
subsection, ahead of "Accessibility."

## What's solid

- **All 7 promoted facts checked true against the live tree**, not just plausible: the TS2312
  discriminated-union constraint and the type-alias-with-intersection fix (10 instances, all
  converted, confirmed pattern-consistent); `resolveJsonModule: true` present in
  `salesperson/tsconfig.app.json:10`; the TanStack Query v5 mounted-observer requirement, matched
  against `salesperson/src/api/hooks.test.tsx`'s C4 network-effect `describe` blocks
  (`usePresenterResetAll`/`useResetMine`/`useAdvanceOrder`, each with a comment noting the
  observer is mounted so the invalidation is actually visible); the RTL heading/CTA-ambiguity fix,
  confirmed by `ResetControl.tsx`'s and `PresenterResetAllControl.tsx`'s now-distinct `t()` keys
  and `en.json` copy ("Session controls"/"Reset my session" vs. "Reset everyone"); the
  seed-placeholder test-scoping trap, matched word-for-word against
  `falkor-chat/docs/HISTORY.md`'s 2026-09-14 "fix cross-cutting test breakage" entry; the
  `flex-1`/`h-full` Chromium sizing quirk, matched against `Shell.tsx`'s still-present
  `<div className="flex-1"><Outlet/></div>`; and the i18next unbound-`t()` fact, matched
  byte-for-byte against `salesperson/node_modules/i18next/dist/cjs/i18next.js`'s actual `t(...args)
  { return this.translator?.translate(...args); }` body.
- **The discard is correctly verified, not just asserted.** `App.tsx`'s own header comment and
  `docs/plans/salesperson-ui.md` §4.11/§5.0 (v1.36) both independently confirm the provider-order
  fix landed as a genuine architectural change (a pathless `LayoutShell` layout route, not a
  provider reorder), and `layout/sessionBridge.ts`/`layout/injectBridge.tsx`/
  `layout/injectBridge.test.tsx` — the raw entry's own cited workaround — no longer exist in the
  tree. `falkor-chat/docs/reviews/salesperson-ui-s12b.md` confirms the Blocker/Major this closed.
- **The KB-vs-project-docs override is sound.** Both overridden entries (`resolveJsonModule`, the
  seed-placeholder trap) are technique facts that generalize beyond this one project's current
  state — a Vite/TS scaffold default and a test-scoping failure mode of a staged-handoff pattern —
  rather than facts about *this* project's present configuration, which is the right axis for
  routing to a knowledge base over project docs.
- **Cross-file consistency holds.** `claude/README.md`'s catalog row, `claude/AGENTS.md`'s roster
  parenthetical, and `frontend-engineer.md`'s own blockquote all describe the same file with the
  same topic list, and none contradicts another.
- **The README "Senior" → "Front-end" fix is a genuine stale leftover**, not a live word:
  `git log --all --oneline | grep -i senior` turns up only the retired `coding-senior` OpenCode
  agent (unrelated), no other current agent row in `claude/README.md` uses "Senior" framing
  (`coder`, `tdd-engineer` both read "Software engineer who implements…"), and the diff touches
  only that one word in the row — nothing else in the same line changed.
- **Graph state independently reconfirmed**: `MATCH (a:Agent {agentId:'frontend-engineer'})
  -[:PRODUCED]->(k:KaizenEntry) RETURN count(k)` → 0, and the legacy `author`-property read
  (`MATCH (k:KaizenEntry) WHERE k.author = 'frontend-engineer' RETURN count(k)`) → 0 as well.

## Open questions

None — this unit's scope is fully self-contained and everything needed to verify it was checkable
from the repo and the graph.
