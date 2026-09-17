# Agent knowledge-base strategy — Stage 0 implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

## Scope & verdict

Diff-scoped implementation review of `cobb`'s uncommitted K-030 Stage 0 delivery (working tree
against `HEAD`, no commits made): the four interim knowledge-base extractions
(`claude/teco/coordination-techniques.md`, `claude/architect/plan-authoring-techniques.md`,
`claude/data-scientist/statistical-method-techniques.md`,
`claude/tdd-engineer/test-design-techniques.md`) plus the accompanying edits to the four prompt
files, `claude/AGENTS.md`, `claude/README.md`, all four agents' `kaizen/history.md`, and K-030's
entry in `claude/cobb/kaizen/plan.md`. Verified by reading every removed prompt paragraph against
its landing spot in the new KB file (not by trusting section/word counts), independently
re-deriving all eight before/after word-count and longest-line figures, and diffing the roster/
catalog/history changes against the six existing agents' precedent. Out of scope: whether Stage 0
was the right call (already decided upstream), and the unrelated uncommitted changes to
`docs/plans/small-model-benchmarking-coordination.md` and the `model-bench/reports/`/
`skills/synced/` untracked files visible in `git status` — none of that is part of this unit.

**Verdict: approve with suggestions.** No content loss, no load-bearing guardrail misplaced, no
overstatement of K-030's status. Two small factual inaccuracies in `cobb`'s own reporting (a
section-count off-by-one, propagated into three documents) and a handful of FR-7 sections that
still bundle more than one independently-verifiable claim — worth a light follow-up pass, not a
re-do.

**CPG:** not applicable — this is a documentation/prompt-restructuring task with no code-level
component; `claude/` carries no CPG and none would apply here.

## Findings

### Minor — `architect/plan-authoring-techniques.md` has 13 sections, not the reported 12

`grep -c '^## ' claude/architect/plan-authoring-techniques.md` → **13**, listed in full at
`claude/architect/plan-authoring-techniques.md:12,19,26,36,42,48,62,70,77,83,91,98,108`. `cobb`'s
self-report, `claude/AGENTS.md`'s (implicit, no count) roster line, and — the one place this
matters — `claude/architect/kaizen/history.md`'s 2026-09-17 entry ("12 `##`-headed sections") and
`claude/cobb/kaizen/plan.md`'s K-030 table row ("new, 12") both state **12**. Word counts (1,208)
and every other figure check out; only the section tally is off by one. Low stakes on its own, but
it's now baked into two permanent history records rather than a self-report that gets discarded —
have whoever next touches either file correct "12" to "13" in place (a one-word edit, not worth a
dedicated unit).

### Minor — a few FR-7 sections still bundle more than one independently-verifiable claim

FR-7's bar is "one `##`-heading-delimited technique per section, each holding one claim where
possible" — the plan explicitly names `review-techniques.md`'s existing bundling as the defect not
to repeat. Three of the four new files hit that bar cleanly (`plan-authoring-techniques.md` and
`test-design-techniques.md` are exemplary — tighter than the six-agent precedent they're modeled
on; `statistical-method-techniques.md` is close). `coordination-techniques.md`, extracted from by
far the densest source prose, has a handful of sections that still fold in 2-3 separable claims:

- `claude/teco/coordination-techniques.md:73-100`, "Mutation-test the green-on-arrival tests" —
  bundles the implementer-facing backup/mutation discipline, a separate integration-time
  "mutate one argument yourself" instruction, and two distinct named fixture-defect shapes
  (invented-key, uniform-stub-value) under one heading.
- `claude/teco/coordination-techniques.md:117-132`, "A reviewer's suggested fix is a finding to
  judge, not an instruction to apply" — the core claim plus three genuinely distinct illustrative
  shapes (partial reproduction, method-vs-scope confusion, guard-folded-into-its-own-step) plus a
  separate corollary about a gate's fix-routing bias.
- `claude/data-scientist/statistical-method-techniques.md:72-85`, "A published statistic's
  arithmetic and provenance label are both part of the claim" — the heading's own "and" names two
  claims (exact-arithmetic computation; provenance-audit-before-transform) that don't share a
  mechanism, just a domain.

None of this loses content or misleads — each bundled section is still coherent prose on one
theme — but it's the same shape of imperfection the plan flagged, not an improvement on it, in
`coordination-techniques.md` specifically (the file drawn from the densest, most bullet-per-bullet
source). Worth a follow-up split pass whenever that file is next touched, not a blocker now: FR-7
itself only requires "where possible," and Stage 6's migration pass (§3 of the plan) will re-touch
every KB file regardless.

## What's solid

- **Zero content loss, verified by direct comparison, not by trusting stated counts.** Every
  removed paragraph in all four prompt diffs was read side-by-side against its landing spot in the
  new KB file; content survives close to verbatim (often literally verbatim), including the
  backup-verification `&&`/`test -s` shell idiom, every named origin/measurement, and every
  cross-reference. No paraphrase-thinning found anywhere.
- **Nothing load-bearing moved out of an always-loaded prompt.** Checked each file's Guardrails/
  Principles section post-edit: `teco.md` retains the commit grant, the "never mutate the working
  tree" rule, the hook-enforcement clause, and the ledger/pause-resume mechanics; `architect.md`
  retains the source/test/config prohibition, the commit grant, and "don't hand-wave";
  `tdd-engineer.md` retains the core red-green-refactor loop, altitude guidance, and the
  cross-referenced "rejected alternative is the mutant" rule; `data-scientist.md` retains its
  expertise-area routing structure. Everything moved is genuinely rare-path (recovery scenarios,
  specific review/design traps), matching the judgment test cobb was given.
- **All eight before/after word-count and longest-line figures re-derived independently
  (`wc -w`, `awk '{print length}' | sort -rn | head -1` against both the working tree and
  `git show HEAD:<path>`) and matched `cobb`'s self-report exactly** — `teco.md` 11,010→8,118
  (longest 3,105→1,874), `architect.md` 2,553→1,522 (1,617→718), `data-scientist.md`
  2,661→2,032 (1,247→826), `tdd-engineer.md` 2,646→2,138 (1,502→1,240). New-file word counts also
  matched exactly (4,758 / 1,208 / 992 / 758) — only the one section-tally figure above didn't.
- **Roster/catalog/pointer-line consistency is exact.** `claude/AGENTS.md` and `claude/README.md`
  name every new KB with phrasing and detail level matching the six existing agents' entries; the
  in-prompt pointer sentences ("live on demand in `<file>` — consult it when...") match the exact
  style already used by `analyst`, `devops`, `qa-engineer`, and `frontend-engineer`
  (cross-checked by grep against all four).
- **`kaizen/history.md` entries are accurate and appropriately dated 2026-09-17** in all four
  agent folders, each correctly labeled a "prompt restructure, not a kaizen distillation," each
  listing the same re-derived figures. **K-030's `cobb/kaizen/plan.md` entry does not overstate
  delivery** — status flipped 🔵→🟡 (not closed), explicitly states Stage 0 is interim relief, and
  correctly names both the still-open "knowledge base vs. restructure" policy question and the
  blocked Stages 1-7 graph-backed substrate as unresolved.

## Open questions

None requiring the caller's input — both findings above are self-contained follow-up items with an
obvious owner (whoever next touches the named file) and no design judgment attached.
