# Anthropic API adoption — Coordination

> **Status:** active · **Owner:** `tico` · **Tracks:** —

Docs-only chain run by `tico` (requirements → plan → reviews → revision). Never touches source, tests or config: the instant a unit would need an implementer, the remaining chain goes to `teco`. Every verified deliverable stays uncommitted and commits together by explicit path at the chain's terminal state. Requirements: `docs/requirements/anthropic-api-adoption.md`.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| D1 plan | `architect` | a77d6bb990dbd17ab | delivered (uncommitted) | `docs/plans/anthropic-api-adoption.md` | D2 → needs changes | 202k tok / 30 tools |
| D2 plan review | `analyst` | a0366746239c92fb1 | delivered (uncommitted) | `docs/reviews/anthropic-api-adoption.md` `## Pass 3` | needs changes (1 B / 5 M / 5 m / 1 n) | — |
| D3 method note | `data-scientist` | aab6ada0c2a61ab88 | delivered (uncommitted) | `docs/plans/anthropic-api-adoption-ml.md` | — (its blocker B1 + M1–M5 against the plan go to D5) | — |
| D4 security review | `security-expert` | aa65152d5c7a64c71 | delivered (uncommitted) | `docs/reviews/anthropic-api-adoption-security.md` | approve with suggestions (0 B / 5 M / 8 m / 1 n) | — |
| D5 plan revision 1 | `architect` | a77d6bb990dbd17ab (resume) | delivered (uncommitted) | `docs/plans/anthropic-api-adoption.md` (in place; 3.4k → 4.6k words) | D6 → approve with suggestions (Pass 4: 0 B / 1 M / 2 m / 1 n) | 247k tok cumulative |
| D6 plan re-review | `analyst` | a0366746239c92fb1 (resume) | delivered (uncommitted) | `docs/reviews/anthropic-api-adoption.md` `## Pass 4` | approve with suggestions | 138k tok cumulative |
| D7 plan revision 2 | `architect` | a77d6bb990dbd17ab (resume) | delivered (uncommitted) | `docs/plans/anthropic-api-adoption.md` (in place): Pass 4 N1, m6, m7, n2, growth-stops items, stakeholder answers | D8 | — |
| D8 plan re-check | `analyst` | a0366746239c92fb1 (resume) | delivered (uncommitted) | `docs/reviews/anthropic-api-adoption.md` `## Pass 5` | approve with suggestions (0 B / 1 M / 1 m / 2 n) | 159k tok cumulative |
| D9 plan revision 3 | `architect` | a77d6bb990dbd17ab (resume) | accepted (tico line check 2026-11-19: `campaign.py` sub-item in §3.6.2 and U8 file list with mutants; 5,001 w) | `docs/plans/anthropic-api-adoption.md` (in place): Pass 5 N2 (`campaign.py` into U8), n3, n4 | `tico` line check against Pass 5 (no further analyst pass) | — |

## Notes

- 2026-11-19: stakeholder asked for the analyst review on plan landing, then the data-scientist and security reviews.
- 2026-11-19: Pass 3 open questions for the revision round: B1 (API path gets its own prompt and pins, or adapt the subagent definitions: architect's call); M1 (guard/responder calls count toward step tokens, or a stated floor plus a correct hosted flag: stakeholder question if the architect won't settle it).
- 2026-11-19: stakeholder answers folded into the requirements doc (floor for FR-10 tokens; FR-2/3/15 clarifications confirmed; key control = tripwire + human provisioning; Opus effort medium). Unconfirmed, still open: second Opus replicate, `claude-fable-5-2` generator, decision-models token bounds.
- 2026-11-19: stakeholder settled the three open spend questions (two Opus replicates; generator `claude-opus-5-5` with caveat, overriding the ML note's Fable; bounds 0.6M/1.0M). D10 `data-scientist` (aab6ada0c2a61ab88) revised the ML note: delivered. D11 `architect` (a77d6bb990dbd17ab) aligned plan §3.6.3/§3.5.6/U1b/U9/§6: accepted (tico line check 2026-11-19: no Fable text left in the plan, §6 lists risks only, 5,018 w).
- Open requirement of the family: model-bench whole-component live spec (`docs/plans/live-specs-coordination.md` L6) is owed before the feature closes.
- 2026-11-19: terminal state. Plan accepted (analyst Pass 5 approve with suggestions, N2 and the stakeholder decisions folded after it by line check); chain committed together. Implementation (U1a first, security gate before it) is `teco`'s; this ledger is its state of record. Outstanding: `qa-engineer` test plan (`docs/test-plans/anthropic-api-adoption.md`), model-bench live spec (L6).
