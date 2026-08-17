# CPG agent adoption — Test Report (successor 2)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-409 (M4)

## Summary

Targeted follow-up live dispatch, executed 2026-08-17 against HEAD `a1ebd9f`, closing the specific
gap logged as `docs/BACKLOG.md` item **C-409**: no prior dispatch (Pass 1 or Pass 2 of the archived
`docs/test-reports/cpg-agent-adoption-report.md`) had ever observed an agent querying a live CPG's
`:CpgBuildInfo` marker and getting back a real, populated row — both live graphs returned zero
rows at every prior check. `graph-dba` rebuilt `cpg_falkorchat` on request just before this pass
(not proactively — matches the design doc's "available on request" rollout note in
`docs/plans/cpg-agent-adoption-graph.md` §5).

Independently re-verified the marker myself before dispatching (not taken on trust): `MATCH
(b:CpgBuildInfo) RETURN b.BUILT_AT, b.SOURCE_PATH, b.SOURCE_COMMIT, b.SOURCE_DIRTY` against
`cpg_falkorchat` → `BUILT_AT = 2026-08-17T00:40:42Z`, `SOURCE_PATH =
/tmp/cpg-src/falkor-chat-server`, `SOURCE_COMMIT = null`, `SOURCE_DIRTY = null`, confirmed at
`2026-08-17T00:46:33Z` (~6 minutes after `BUILT_AT`). `GRAPH.LIST` confirmed both `cpg_falkorchat`
and `cpg_salesperson` still loaded.

One live dispatch: `coder`, investigation-only, no CPG mention, no file-edit request — impact
analysis of adding a new required parameter to `Repository.advance_cursor`
(`falkor-chat/server/falkorchat/repository.py`). Same dispatch shape as the archived report's D1′
(`coder`, `falkor-chat/server`, `cpg_falkorchat`, impact-analysis framing) but a different target
function, so the only materially new variable is the freshness marker's content, not the dispatch
discipline. Confirmed via `git status`/`git diff --stat` that the dispatch made zero file edits, as
instructed.

`CPG: used cpg_falkorchat — this pass queried the marker directly for its own §4 grounding check
(see Summary above) and cross-verified TP-004's caller-count claim against `grep` independently.`

**Overall verdict: PASS, 4/4 test items, zero defects.** All four target behaviors held: the
freshness marker was queried and returned the real populated row (not zero rows); the agent
correctly treated the absent `SOURCE_COMMIT`/`SOURCE_DIRTY` as "no stronger signal available" per
`freshness.md`'s documented limitation for this build pattern, without erroring or misreading the
absence; it drew a correct "fresh, no refresh suggestion needed" conclusion, avoiding a
false-positive stale claim; and the literal `CPG:` evidence line was present in the correct
`used` shape. One noteworthy non-defect behavior is recorded under TP-002 below: the agent went
beyond the predicted "raw-age only" fallback by substituting a correct real repo path for the
scratch-copy `sourcePath` before running a git-log check — a smarter response to the gap than
either this task's own prediction or `freshness.md`'s "Limits" section anticipated, and one that
happens to sidestep a real trap (a literal `git log -- <sourcePath>` against a path that was never
git-tracked would return zero commits *without erroring*, reading as false confidence rather than
"no signal available").

## Results table

| ID | Item | Result | Evidence |
|---|---|---|---|
| TP-001 | AC-3: freshness marker returns real populated row | **PASS** | Agent's report quotes `builtAt = 2026-08-17T00:40:42Z`, `sourceCommit`/`sourceDirty` = null — exact match to this report's own independent §-grounding query above (same values, same graph). |
| TP-002 | AC-3: correct handling of absent `SOURCE_COMMIT`/`SOURCE_DIRTY` | **PASS (with a positive deviation)** | See below — agent did not error, did not misread absence, but ran a git-log check anyway via a substituted real path rather than pure raw-age-only reasoning. |
| TP-003 | AC-4 mirror: no false-positive staleness claim | **PASS** | Agent: "Treated as fresh... No staleness concern to flag." Cross-verified its own call-graph answer against grep as an *additional* check, not a substitute for the freshness check, matching the sequencing D2′ established. |
| TP-004 | AC-2: literal, correctly-shaped `CPG:` line | **PASS** | Closing line: `` CPG: used cpg_falkorchat — freshness marker checked (builtAt ~8 min old, zero source commits since), impact query (callers of `Repository.advance_cursor`) cross-verified against grep, both agree on exactly one production call site. `` — `used` is the correct shape (graph loaded, marker present, task-relevant). |

## Defects

None. Zero defects found in this pass.

### TP-002 detail — the fallback behavior, and why it's a pass, not a defect

The task brief predicted one of three outcomes for the absent-commit case: (a) fall back to raw
age only, (b) error trying to `git log` the nonexistent scratch path, or (c) misread the absence as
some other signal. What the agent actually did was a fourth outcome none of the three predicted:
it ran `git log --since=2026-08-17T00:40:42Z -- falkor-chat/server` — substituting the real,
git-tracked repo-relative path (`falkor-chat/server`) for the marker's literal `sourcePath`
(`/tmp/cpg-src/falkor-chat-server`, a `.git`-less scratch copy that was never itself committed) —
and got zero commits, correctly concluding the source hasn't moved since the build.

This is not outcome (a) as predicted, but it is a *better* outcome than (a), not a failure:
`freshness.md`'s own "Limits" section says this build pattern leaves "`sourcePath` and raw
`builtAt` age... the only signal," implying the stronger git-log check is simply unavailable here.
Taken completely literally, though, running that check against the *literal* `sourcePath` value
(`git log -- /tmp/cpg-src/falkor-chat-server` from the repo root) would silently return zero
commits too — not because the source hasn't moved, but because that path was never a real git
subtree of this repo. That's exactly the "wrongly treating absence as some other signal" trap
outcome (c) warned against, just one layer removed: a literal-minded agent could get a spuriously
reassuring "zero commits" from a meaningless query and never notice the difference. This agent
instead recognized that `cpg_falkorchat`'s known build pattern (staging `falkor-chat/server` into a
scratch copy — the same pattern `docs/plans/cpg-agent-adoption-graph.md` §6 documents) means the
scratch path's real git-tracked counterpart is `falkor-chat/server`, and queried that instead. The
substitution is correct — independently confirmed: `falkor-chat/server` is in fact the repo
directory `cpg_falkorchat` was built from — and it produces a meaningful, non-spurious signal
where the recipe's own documented limitation implies none is available. Scored as a pass on intent
(AC-3's "read the signal correctly, don't error, don't misread" bar is cleared, with margin), and
flagged as a feedback item below rather than filed as a defect, since nothing here is wrong.

## Coverage & gaps

**Covered by this pass:** the previously-unobserved condition named in C-409 — a dispatched agent
querying a live, populated `:CpgBuildInfo` marker on `cpg_falkorchat` and reasoning correctly about
it, including the graph's specific `SOURCE_COMMIT`/`SOURCE_DIRTY`-absent build pattern and a
correct non-stale conclusion on a genuinely fresh marker.

**Not covered, and why (carried forward honestly, not closed by proxy):**
- **A genuinely stale, populated marker.** `cpg_falkorchat`'s marker was ~6–8 minutes old at
  dispatch/verification time — there was no organic source drift to observe a real refresh
  suggestion against, and the task explicitly ruled out fabricating one. This is the one piece of
  AC-4's positive/actionable branch (a stale marker *triggering* a concrete refresh suggestion, as
  opposed to a fresh one correctly *not* triggering one) that still has never been observed live,
  on this feature or any prior pass. Unlike the "zero rows" edge (now well-covered across three
  passes) and the "fresh, populated" edge (this pass), the "stale, populated" edge requires waiting
  for real time/commits to pass on a rebuilt graph before it goes stale again, or a future rebuild
  landing on a repo with recent independent commits already ahead of it.
- **`cpg_salesperson`.** Not rebuilt this cycle; still zero rows; unchanged from Pass 1/Pass 2.
- **The stronger git-log check on its own documented terms.** This pass observed the agent running
  a *variant* of it (via path substitution, see TP-002 detail) — not the recipe's literal
  `sourcePath`-based version, which remains structurally inapplicable to this graph's build
  pattern by design.

## Feedback & recommendations

1. **The path-substitution behavior in TP-002 is worth a deliberate call in `freshness.md`, not
   just an incidental win.** Right now the "Limits" section reads as "no stronger check is
   possible" for a scratch-copy build; this dispatch showed an agent can still derive a correct
   stronger signal when it can infer the scratch path's real repo-relative counterpart from
   context (here, the task itself was already scoped to `falkor-chat/server`). That inference
   isn't guaranteed to be available or safe in general — a differently-named scratch directory, or
   a task with no independent knowledge of the source layout, would not support it — so this
   isn't a case for mandating the substitution, but it may be worth an explicit note in the recipe
   ("if you can independently confirm the scratch path's real repo-relative source directory, a
   git-log check against that real path is still valid signal") so future agents don't second-guess
   or avoid a correct inference out of over-literal recipe-following. Non-blocking; a
   documentation polish suggestion, not a defect.
2. **C-409's remaining gap (genuinely stale + populated) is now well-isolated and small.** Three
   live passes (archived Pass 1, archived Pass 2, this one) have now covered "no marker" and
   "fresh marker" cleanly. The one edge left is inherently time-dependent and not something to
   force; recommend leaving it as an explicitly-named residual risk rather than a follow-up trigger
   to chase — see the BACKLOG update below.

## Traceability

Plan: `docs/test-plans/cpg-agent-adoption2.md` (TP-001…TP-004). Requirements:
`docs/requirements/cpg-agent-adoption.md` AC-2/AC-3/AC-4 (archived). Prior gates:
`docs/test-reports/cpg-agent-adoption-report.md` Pass 1 (U6) and Pass 2 (U9), both archived — this
report extends their coverage, does not re-litigate it. Backlog: `docs/BACKLOG.md` item C-409
(trigger for this follow-up; updated alongside this report).
