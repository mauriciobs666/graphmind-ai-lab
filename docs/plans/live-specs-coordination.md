# Live Specs — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** —

Build chain for `docs/requirements/live-specs.md` (Ready for design, `analyst` Pass 3 approve with
suggestions, `docs/reviews/live-specs.md`). Touches agent prompts/config → code-implementation
commit rule (commit per unit after its gate). Unit set is provisional until L1's plan is gated.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| L1 plan | `architect` | ac210462a800a6cea | accepted (v1.2, 3,078 w; teco line check 2026-10-31: §1 integrity-check clause, L3c curation carve-out, L5s catch-all) | `docs/plans/live-specs.md` | `analyst` afeaffe153a3133b1 → Pass 4 needs changes; Pass 5 approve with suggestions | 136k / 37; gate 125k / 27 |
| L2 root `AGENTS.md` conventions + three `REVIEW.md` | per L1 | — | queued | per L1 | `analyst` → — | — |
| L3 agent-prompt edits (AC-6, AC-9, teco feature-close step) | `cobb` | — | queued | `claude/` | `analyst` → — | — |
| L4 rebuild `cpg_falkorchat` (stale: 21 server commits since `5c82170f`) | `graph-dba` | a7c1dbec8f5c50443 | accepted (teco read `CpgBuildInfo`: built 2026-10-31T17:34:31Z from `c0eeb348`) | FalkorDB `cpg_falkorchat` | — → — | 91k / 22 |
| L5a–c falkor-chat catch-up (`DESIGN.md`, `SERVER.md`, `QUERIES.md`), one gate each | per L1 | — | queued | `falkor-chat/docs/` | `analyst` vs shipped code (AC-1) → — | — |
| L6 model-bench live spec | per L1 | — | queued (deferred: after typed-decision-models closes) | per L1 | `analyst` vs shipped code (AC-1) → — | — |

- 2026-10-31: Stakeholder: full `architect` plan + `analyst` gate; model-bench's spec (L6) is written
  only after the typed-decision-models chain (`docs/plans/typed-decision-models-coordination.md`)
  closes. L1 and L4 dispatched in parallel.
- 2026-10-31: Stakeholder Q1: model-bench's spec (L6) is written **inside typed-decision-models' close**, after its build units are gated and before its closing `HISTORY.md` entry. `analyst` Pass 4 independently found the same conflict and judged this reading valid.
- 2026-10-31: Stakeholder on P4-M2: no widening of `analyst`'s write guard; `analyst` drafts `REVIEW.md` content in its reviews, `teco`/`cobb` apply it; `security-expert` gates `analyst.md` edits. Architect resumed for v1.1 with Pass 4 + both answers.
- 2026-10-31: Pinned code sha for L5a–c checks: `c0eeb348` (= `cpg_falkorchat` source; no falkor-chat commits since). Open ledgers checked: none in flight on the falkor-chat spec set.
