# Agent knowledge-base strategy — Stage 0 coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

Coordinates execution of **Stage 0 only** (`claude/docs/plans/agent-knowledge-base-strategy.md`
§3, "Ready to implement — Stage 0 only" section) — interim flat-file knowledge-base relief for
K-030's four agents (`teco`, `architect`, `data-scientist`, `tdd-engineer`). Stages 1–7 (the
graph-backed substrate) remain blocked pending `architect`'s plan revision to reflect the
2026-09-17 Option B / raw-capture-migration decisions recorded in
`claude/docs/requirements/agent-knowledge-base-strategy.md` — not part of this coordination.

Baseline word counts, measured directly (`wc -w`) at dispatch, 2026-09-17: `teco.md` 11,010,
`architect.md` 2,553, `data-scientist.md` 2,661, `tdd-engineer.md` 2,646.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `cobb` | `ae2167ef7e7153aa0` | accepted | commit `6d834f0` | `analyst` (`a622c2f6fe80f9e65`) → approve with suggestions | 349388 tok / 103 tools (+193362 tok / 33 tools review) |

## Notes

- All four agents' knowledge-base files, plus the shared `claude/AGENTS.md` roster and
  `claude/README.md` catalog, are touched by the same unit deliberately — parallelizing per-agent
  would collide on those two shared files (teco.md Guardrails, "two units touch the same file").
- `data-scientist` and `tdd-engineer` already carry partial knowledge bases
  (`lm-studio-model-notes.md`; `guard-testing-techniques.md` + `estimator-test-fixtures.md`) —
  Stage 0 for them means extracting their *remaining* on-demand-shaped content, not starting from
  zero.
- FR-7 authoring constraint (requirements doc, plan §3): new KB files follow the existing
  six-agent convention exactly — one `##`-heading-delimited technique per section, one claim per
  section where possible.

## Close-out (2026-09-17)

U1 accepted, committed `6d834f0`. `analyst`'s gate (`claude/docs/reviews/agent-knowledge-base-strategy-impl.md`)
found zero content loss and all figures independently re-derived and matching; two minor,
non-blocking findings: a section-count off-by-one (`architect/plan-authoring-techniques.md`,
12→13 — corrected directly in this commit, single-word fix in `architect/kaizen/history.md` and
two spots in `cobb/kaizen/plan.md`), and a handful of sections in
`teco/coordination-techniques.md` that still bundle 2–3 separable claims (FR-7 says "where
possible"; deferred to whenever that file is next touched or Stage 6's already-planned
migration pass, per the review's own recommendation — not worth a dedicated unit).

**Stage 0 is complete and closed.** Stages 1–7 (the graph-backed substrate) remain separately
blocked pending `architect`'s plan revision — not part of this coordination; see
`claude/docs/plans/agent-knowledge-base-strategy.md`'s own status header for that thread.
