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

## Build units (plan steps; code chain — commit per verified unit by explicit path; tree carries unrelated in-flight changes from other sessions, so commits are pathspec-only under `skills/work-items/`)

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| B1 (plan step 1) | `tdd-engineer` | `ac7238baf44ef7314` | accepted (committed; diff gate batched) | `skills/work-items/scripts/wilib/{rules,events}.py` + tests | `analyst` (diff gate, batched per ≤3 steps) → — | ~103k tok, 16 tool uses |
| B2 (step 2) | `tdd-engineer` | `a6cfead2df12dcbe8` | accepted (committed) | `skills/work-items/…` | — | ~95k tok, 11 tool uses |
| B3 (step 3) | `tdd-engineer` | `ac3a40cd64f27380a` | accepted (committed) |
| B4 (step 4 + CLI `flush`/`init` gap from step 3) | `tdd-engineer` | `a798c99c101250269` | accepted (committed) |
| B5 (step 5) | `tdd-engineer` | `adca059df71547b9f` | accepted (committed) | `skills/work-items/…` | — | ~91k tok, 12 tool uses |
| G2 diff gate steps 1–5 | `analyst` | `ad341f66ca9cc17cb` | delivered | `docs/reviews/work-item-numbering.md` Pass 7 | approve with suggestions (F1 resolve FR/AC parent defect, F2 silent-corruption test gap, F3 links shape decision) | — |
| B5b fixes F1/F2/minors | `tdd-engineer` | `a798c99c101250269` | accepted (committed 28ce49c5; ~132k tok) | `skills/work-items/…` | re-confirm by `analyst` after step 7 | — |
| U1b plan sync + F3 decision | `architect` | `a7af14139c4b90236` | delivered (F3: concerns functional; fixed-by/relates multi-valued via link op; step 6 split into 6 + 6b) | `docs/plans/work-item-numbering.md` | — | — |
| B6 (step 6 links/attrs) | `tdd-engineer` | `acbba0473b46da2aa` | in-flight | `skills/work-items/…` | — | — |
| B6b (step 6b headers/options) | `tdd-engineer` | — | queued | — | — | — |
| B7 (step 7 rebuild/check; sequential, same package) | `tdd-engineer` | — | queued | `skills/work-items/…` | — | — |
| B8 (step 8) | `tdd-engineer` | — | queued | keys | — | — |
| B9–B10 (steps 9–10) | `cobb` | — | queued — touches shared files (`claude/AGENTS.md`, agent prompts): do three-way diff check before commit | — | — |
| B11 (step 11) | `teco` | — | queued — root `AGENTS.md` shared file | — | — |
| B12–B13 (steps 12–13) | `coder` / `teco` | — | queued | — | — |
