# Live Specs — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** —

Build chain for `docs/requirements/live-specs.md` (Ready for design, `analyst` Pass 3 approve with
suggestions, `docs/reviews/live-specs.md`). Touches agent prompts/config → code-implementation
commit rule (commit per unit after its gate). Unit set per `docs/plans/live-specs.md` §4.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| L1 plan | `architect` | ac210462a800a6cea | accepted (v1.2, 3,078 w; teco line check 2026-10-31: §1 integrity-check clause, L3c curation carve-out, L5s catch-all) | `docs/plans/live-specs.md` | `analyst` afeaffe153a3133b1 → Pass 4 needs changes; Pass 5 approve with suggestions | 136k / 37; gate 125k / 27 |
| L2a `REVIEW.md` seed drafts | `analyst` | aaf336d73b8a7e37c | accepted (Pass 6; `cobb` Pass 7 approve w/ suggestions) | `docs/reviews/live-specs.md` Pass 6 | — (drafted only) → — | — |
| L2b apply seeds to three `REVIEW.md` | `teco` | — | accepted (Pass 6 verbatim + Pass 7 replacements P7-2/3/4 and call 1 verbatim; 468/246/199 w) | `REVIEW.md`, `falkor-chat/REVIEW.md`, `model-bench/REVIEW.md` | `cobb` → — | — |
| L2c root/falkor-chat `AGENTS.md` + README conventions | `cobb` | — | queued | per plan §4 | `analyst` → — | — |
| L3a reviewer prompts (analyst, qa, security, data-scientist) | `cobb` | — | queued | `claude/` | `analyst`; `security-expert` for `analyst.md` → — | — |
| L3b prompts (graph-dba, cobb, tico) | `cobb` | — | queued | `claude/` | `analyst` → — | — |
| L3c teco/README/architect (switch-over commit) | `cobb` | — | queued | `claude/` | `analyst` + AC-9 sweep → — | — |
| L7 agent-team manual | `tico` | — | queued | `claude/docs/manuals/agent-team.md` | `teco` doc check → — | — |
| L4 rebuild `cpg_falkorchat` | `graph-dba` | a7c1dbec8f5c50443 | accepted (teco read `CpgBuildInfo`: built 2026-10-31T17:34:31Z from `c0eeb348`) | FalkorDB `cpg_falkorchat` | — → — | 91k / 22 |
| L5a `DESIGN.md` catch-up | `graph-dba` | a44b059b7a88b09ec | delivered (8,519→9,731 w; AC-3 grep 0 — teco re-ran; scoped check queued behind Pass 7) | `falkor-chat/docs/DESIGN.md` | `analyst` scoped → — | 236k / 83 |
| L5b `SERVER.md` catch-up | `coder` | a7ed26eacaac60cbf | in-flight | `falkor-chat/docs/SERVER.md` | `analyst` scoped → — | — |
| L5c `QUERIES.md` catch-up | `graph-dba` | a180eb2b7b147881a | in-flight | `falkor-chat/docs/QUERIES.md` | `analyst` scoped → — | — |
| L5s set-level AC-1 | `analyst` | — | queued | `docs/reviews/live-specs.md` | (is the gate) | — |
| L5d `HISTORY.md`/`BACKLOG.md` | `teco` | — | queued | `falkor-chat/docs/` | — (mechanical) | — |
| L6 model-bench live spec | `coder` | — | queued (inside typed-decision-models' close, before its closing `HISTORY.md` entry) | `model-bench/docs/DESIGN.md`, `FORMATS.md` | `analyst` vs shipped code (AC-1) → — | — |

- 2026-10-31: Stakeholder: full `architect` plan + `analyst` gate; model-bench's spec (L6) is written
  only after the typed-decision-models chain (`docs/plans/typed-decision-models-coordination.md`)
  closes. L1 and L4 dispatched in parallel.
- 2026-10-31: Stakeholder Q1: model-bench's spec (L6) is written **inside typed-decision-models' close**, after its build units are gated and before its closing `HISTORY.md` entry. `analyst` Pass 4 independently found the same conflict and judged this reading valid.
- 2026-10-31: Stakeholder on P4-M2: no widening of `analyst`'s write guard; `analyst` drafts `REVIEW.md` content in its reviews, `teco`/`cobb` apply it; `security-expert` gates `analyst.md` edits. Architect resumed for v1.1 with Pass 4 + both answers.
- 2026-10-31: Pinned code sha for L5a–c checks: `c0eeb348` (= `cpg_falkorchat` source; no falkor-chat commits since). Open ledgers checked: none in flight on the falkor-chat spec set.
- 2026-10-31: `docs/reviews/live-specs.md` is written serially: Pass 7 (`cobb`, L2b gate) in flight; L5a–c scoped `analyst` checks queue behind it.
- 2026-10-31: L5a findings for L5d / follow-ups (not this chain's code scope): (1) `Document.lshBand0..7` never written — teco verified (`background.py` computes bands, only the shortlist read uses them) → falkor-chat backlog defect; (2) `_drive_loop` lock value: L5a says `6cd35d650b5d` at `c0eeb348`, K-033 text quotes `71055f756280` — unverified by teco, for the L5a scoped check; (3) dormant `MENTIONS`/`REPLY_TO` writers; (4) R-1 `ctxVersion` CAS fix unbuilt, no item; (5) three §13 orphans (retention, cross-ws analytics, `GRAPH.CONFIG` review). Sibling gaps relayed at once to L5b (`ensure_row_identity`, `RAG_QUERY_TIMEOUT_MS`) and L5c (`find_update_shortlist` text + lshBand fact).
- 2026-10-31: Carried to briefs from Pass 7: P7-1 duplicate table replaces Pass 6's (L3a–L3c); P7-2 L2c leaves "Reviews run the growth-stops check (root `REVIEW.md`)." and repoints root `AGENTS.md`'s `Status:`-set citation, L3a–L3c repoint five prompt lines; P7-5 L2c's live-spec bullet gets a bold lead, growth-stops lead kept.
