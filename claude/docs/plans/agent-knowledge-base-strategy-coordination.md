# Agent knowledge-base strategy — Stage 0 coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

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
| U1 | `cobb` | `ae2167ef7e7153aa0` | gated | Stage 0 flat-file KBs for 4 agents (working tree, uncommitted) | `analyst` (`a622c2f6fe80f9e65`) → — | 349388 tok / 103 tools |

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
