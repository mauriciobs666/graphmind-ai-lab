# Work-item numbering and terminology — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** —

Requirements: `docs/requirements/work-item-numbering.md` (Ready for design; analyst review `docs/reviews/work-item-numbering.md`, 3 passes). Chain so far is docs-only (design → review); verified deliverables are held uncommitted and land in one commit at the chain's terminal state. Build units (allocator/lookup/check tooling, skill) are drawn after the plan gate.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 design plan | `architect` | `a7af14139c4b90236` | accepted (rev 3; 3615 words) | `docs/plans/work-item-numbering.md` | `analyst` → — | ~105k+136k tok (rev), 28 tool uses |
| G1 plan gate | `analyst` | `ad7deab2cf57f6c0f` | accepted | `docs/reviews/work-item-numbering.md` (new Pass) | Pass 4: needs changes (1 blocker, 5 major); Pass 5: needs changes (narrow, 3 major); Pass 6: approve with suggestions | — |

## Checkpoint 2026-11-09 (session paused at usage limit)

- State: U1 revision 3 in-flight (`architect` `a7af14139c4b90236`) against `docs/reviews/work-item-numbering.md` §Pass 5 (needs changes, narrow). Done-condition = Pass 5's stopping condition: fix N1, N2, N3 + the two growth cuts → approve with suggestions, no further pass; n1–n4 fold without re-review.
- On resume: `git status` / `wc -w docs/plans/work-item-numbering.md` (was 4549 before rev 3; expect it to fall). If the plan is unchanged since Pass 5, rev 3 did not land.
- Resume path: `SendMessage` `a7af14139c4b90236`; **cold-start fallback** (agentId may not survive a session reboot): fresh `architect` dispatch with a state-recovery brief — read plan + `docs/reviews/work-item-numbering.md` §Pass 5, apply N1–N3, growth cuts, n1–n4, in place.
- Then: verify N1–N3 by the commands in Pass 5 (re-run, don't trust), send a compact confirmation to `analyst` `ad7deab2cf57f6c0f` (cold fallback: fresh `analyst`, brief = Pass 5 stopping condition) → on approve-with-suggestions, mark G1 accepted, draw build units from the plan's 13-step table (steps 1–8 `tdd-engineer`, 9–10 `cobb`, 11 `teco`, 12 `coder`, 13 `teco` routing). Build chain is code → commit per unit from then on.
- Uncommitted docs-only set (commit together by explicit path at chain end or handoff to build): `docs/plans/work-item-numbering.md`, `docs/plans/work-item-numbering-coordination.md`, `docs/reviews/work-item-numbering.md`.

- Update: rev 3 landed (architect `a7af14139c4b90236`, plan 3615 words, was 4549). Next: send compact confirmation to analyst `ad7deab2cf57f6c0f` (Pass 5 stopping condition: N1–N3 + growth cuts); not yet re-verified by teco or analyst.

- Update: Pass 6 approve with suggestions; design chain (U1/G1) closed. Nits to fold in the build (no re-review): move 3.1's duplicate example into test 6; SKILL.md flush rule: `wi flush` runs only after a merge completes. Build units (plan steps 1–13) not yet drawn — awaiting stakeholder go-ahead to implement.
