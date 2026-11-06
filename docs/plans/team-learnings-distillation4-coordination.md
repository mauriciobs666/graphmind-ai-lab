# Team learnings distillation 4 (2026-11) — coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-063 (`claude/cobb/kaizen/plan.md`) · **Extends:** `docs/plans/team-learnings-distillation3-coordination.md`

Goal: `cobb` sweeps the next 40 pending raw entries of `ws:agent-team` (pending = `Fact:` documents not under any `_dispositioned_raw_*` key in `claude/cobb/scripts/kb-claim-manifest.json`; `skills/agent-maintenance/SKILL.md` §5 step 1); `analyst` gates it. Raw documents are not deleted; swept ids are recorded in a new `_dispositioned_raw_distillation4` manifest key.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `cobb` | `adee39ba6e6702843` | accepted (fix pass spot-checked by teco; no re-gate) | working-tree diff + `docs/reviews/team-learnings-distillation4-sweep.md` | `analyst` → — | — |
| U2 | `analyst` | `a88bc3b87f1c37714` | gated (approve with suggestions) | `docs/reviews/team-learnings-distillation4.md` | — → — | — |
