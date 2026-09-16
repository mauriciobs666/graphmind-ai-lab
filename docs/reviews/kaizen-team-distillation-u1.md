# Kaizen-team distillation — U1 (`analyst`, 16-entry inbox)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (`docs/plans/kaizen-team-distillation-coordination.md`, unit U1)

## Scope & verdict

Reviewed the uncommitted working-tree diff produced by `cobb` distilling all 16 `analyst`-produced
raw `:KaizenEntry` nodes in the shared `kaizen_team` FalkorDB graph (per
`skills/agent-maintenance/SKILL.md` §5), against: (1) the actual repo state each promoted/discarded
claim cites, (2) the receiving files' existing content for placement/duplication, and (3) an
independent re-query of `kaizen_team`. Seven changed files: `claude/analyst/review-techniques.md`,
`claude/analyst/kaizen/history.md`, `claude/analyst/kaizen/plan.md`,
`claude/graph-dba/falkordb-quirks.md`, `claude/graph-dba/kaizen/history.md`, `skills/README.md`,
`skills/python-web-quirks/SKILL.md`. I did not evaluate `docs/plans/kaizen-team-distillation-coordination.md`
itself (only used it for scope confirmation, per the brief).

**Verdict: needs changes** — one Major (a mechanical duplicate-heading defect in a permanent
history record) blocks a clean commit; the technical content itself is unusually solid — every one
of the 10 promotions, both discards I could independently re-verify, and both routed-outward notes
checked out true against the cited source.

**CPG:** not applicable — this is a documentation/kaizen-distillation task with no code-level
component under review (the code the promoted entries *describe* was read directly, not analyzed
via CPG).

## Findings

### Major — `claude/analyst/kaizen/history.md` has the 2026-09-13 entry's heading duplicated back-to-back (lines 76-77)

The new 2026-09-16 entry was inserted above the existing 2026-09-13 entry, but the edit left the
2026-09-13 heading line duplicated immediately under itself:

```
76: ## 2026-09-13 — standing distillation pass: 12-entry `analyst` inbox, 9 promoted, 3 discarded
77: ## 2026-09-13 — standing distillation pass: 12-entry `analyst` inbox, 9 promoted, 3 discarded
78: (blank)
79: - **What:** `cobb` ran the standing kaizen-graph distillation over all 12 `analyst`-produced...
```

Confirmed against `git show HEAD:claude/analyst/kaizen/history.md`, where this heading appears
exactly once (at what is now line 5's position) — the duplicate is new in this diff, not
pre-existing. Verified no sibling file in this diff has the same defect (`awk` duplicate-heading
scan across all six changed `.md` files found only this one occurrence). This is a permanent,
lookup-only history record (per `AGENTS.md`'s doc convention it "may grow without bound" but is
never meant to read as broken) — fix by deleting the duplicate line before commit.

### Minor — `skills/README.md`'s "When to use" cell wasn't updated in step with `SKILL.md`'s own trigger list

`skills/python-web-quirks/SKILL.md`'s frontmatter added a new explicit trigger phrase, "an
SPA-fallback override on top of `StaticFiles(html=True)`", to its "Use for" list (alongside the new
description text). `skills/README.md`'s catalog row for the same skill got the parallel Description
cell update, but its "When to use" cell is byte-identical before/after
(confirmed: `git diff` shows only the Description and Owner/date cells changed, not the third
cell) — it still reads only "a method-matching or static-mount-shadowing question," without the new
explicit phrase. Not a wrong claim (the existing phrase loosely covers it), but it breaks the
pattern this same diff establishes (update both the skill's own trigger list and the catalog's
routing cell together) — add the same clause to `skills/README.md`'s "When to use" cell.

### Minor — `skills/python-web-quirks/SKILL.md` isn't in `cobb`'s own `Write`/`Edit` guard allowlist, despite being a now-repeated legitimate write target

`claude/cobb/hooks/guard-cobb-topic-writes.sh`'s allowed-path union lists `skills/agent-maintenance/*`
and `skills/agent-standards/*` (cobb's own skill packages) but not `skills/python-web-quirks/*`.
Yet `skills/README.md`'s own Owner cell (`cobb (analyst + coder inbox distillations, 2026-08-09 /
2026-08-11 / 2026-09-16)`) and `git log -- skills/python-web-quirks/SKILL.md` show at least four
prior distillation commits writing there. Every such write must be escalating to an `ask` prompt
today (the guard's `on_mismatch` default) rather than being explicitly allowed — not a correctness
problem (a human approves each one), but exactly the kind of "extend it, don't broaden the globs
above, when a new such doc surfaces" case the guard's own comment anticipates for
`cypher-mcp/README.md`. Worth adding `skills/python-web-quirks/*|*/skills/python-web-quirks/*` to
the allowlist given the established pattern — `cobb`'s call, outside this review's remit to fix.

### Nit — the `AGENTS.md` awk-verification note's exact line number doesn't reproduce today

`claude/analyst/kaizen/plan.md`'s parking-lot note claims re-running the documented `awk` command
"printed line 697 for a length-980 line actually at line 83 of `falkor-chat/AGENTS.md`." Re-running
it verbatim now: `awk 'length($0)>700{print FILENAME": "NR}' $(git ls-files '*AGENTS.md')` prints
`falkor-chat/AGENTS.md: 698`, not 697, and `falkor-chat/AGENTS.md`'s line 83 is indeed the
length-980 line (confirmed with `FNR`, which correctly gives 83). The underlying bug claim
(`NR` vs `FNR`) is entirely correct and reproduces; only the specific cited line number is off by
one, most likely because one of the upstream `*AGENTS.md` files gained/lost a line since the note
was written. Immaterial to the fix recommended, but worth a quick re-check before this note is
acted on.

## What's solid

- **Every promoted item checked out against live repo state**, not just against the entry's own
  wording: the starlette 1.3.1 `StaticFiles.get_response` claim matches the installed venv source
  line-for-line (`falkor-chat/server/.venv/.../staticfiles.py:147-152`); the RediSearch
  `_escape_fuzzy_token` regex matches `repository.py:161-180` exactly; the
  `resume_workflow_run`/`_drive_or_fault` asymmetry is real (`services.py:2554-2565` vs.
  `:2287`/`:2363`/`:2714`) and even independently documented in `executor.py`'s own `_drive`
  docstring; the `PresenterKeyScreen.tsx` "already fixed" disposition is correct (the gate is
  live at `PresenterKeyScreen.tsx:28`); the `test_storefront.py` combined-test citation
  (lines 861-916) is exact.
- **All 4 discards re-verified against their cited sources, not taken on the entry's word** — S4,
  S5, and S6-precheck review docs all state the identical findings at (approximately) the cited
  line ranges, and the S5/S6 fixes are confirmed shipped in `toolcalls.py:767-770` and commit
  `597dcf5` respectively.
- **Both routed-outward notes are genuinely true and genuinely out of remit** — the `NR`/`FNR` bug
  reproduces, and `cobb`'s own write guard (read directly, `guard-cobb-topic-writes.sh`) confirms
  neither root `AGENTS.md` nor `falkor-chat/docs/` is in its allowlist. Parking-lot placement
  (rather than silent drop, rather than a backlog item neither agent owns) is the right call.
- **Graph state independently confirmed**: both `MATCH (a:Agent {agentId:'analyst'})-[:PRODUCED]->(k) RETURN count(k)`
  and the legacy `author`-property read return `0` — all 16 entries cleared, none left behind.
- **No content duplication or placement collisions** found across the receiving files —
  `falkordb-quirks.md`'s new bullet sits cleanly beside the existing fulltext/fuzzy bullets with no
  overlap, and `review-techniques.md`'s 8 new sections don't restate any existing section (checked
  by topic-keyword grep across all existing `##` headings).
- This pass explicitly avoided the prior full-team sweep's M-2-class defect
  (`docs/reviews/kaizen-inbox-distillation2.md`): `skills/python-web-quirks/SKILL.md`'s own
  frontmatter description *and* trigger list were both updated together (the README gap above is a
  narrower miss of the same discipline, not a repeat of that full failure).

## Open questions

- None requiring stakeholder input — the Major finding is a one-line mechanical fix `cobb` can make
  directly before this unit is committed.
