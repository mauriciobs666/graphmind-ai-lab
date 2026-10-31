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
| L2c root/falkor-chat `AGENTS.md` + README conventions | `cobb` | afe49685688e25039 | accepted (Pass 11 M1/m1/n1/n2 replacement text applied verbatim by teco + `NR`→`FNR` bar fix (i1); root 3,198→3,345 w) | `AGENTS.md`, `falkor-chat/AGENTS.md`, `falkor-chat/README.md` | `analyst` a28b60a6703ee2caf → Pass 11 approve w/ suggestions | 94k / 37; gate 91k / 21 |
| L3a reviewer prompts (analyst, qa, security, data-scientist) | `cobb` | a55d7618cf6a6f45b | in-flight | `claude/` | `analyst`; `security-expert` for `analyst.md` → — | — |
| L3b prompts (graph-dba, cobb, tico) | `cobb` | a11b38d3ff293519a | in-flight | `claude/` | `analyst` → — | — |
| L3c teco/README/architect (switch-over commit) | `cobb` | — | queued | `claude/` | `analyst` + AC-9 sweep → — | — |
| L7 agent-team manual | `tico` | — | queued | `claude/docs/manuals/agent-team.md` | `teco` doc check → — | — |
| L4 rebuild `cpg_falkorchat` | `graph-dba` | a7c1dbec8f5c50443 | accepted (teco read `CpgBuildInfo`: built 2026-10-31T17:34:31Z from `c0eeb348`) | FalkorDB `cpg_falkorchat` | — → — | 91k / 22 |
| L5a `DESIGN.md` catch-up | `graph-dba` | a44b059b7a88b09ec | accepted (10,1xx w; P13-n1/n2 wording applied by teco; P8-m5 clause true once L5d files the `GRAPH.CONFIG` item) | `falkor-chat/docs/DESIGN.md` | `analyst` af2097f722346e76c → Pass 8 needs changes; fix `graph-dba` a02d841bd272c4466; Pass 13 approve w/ suggestions | 236k / 83; gate 180k+212k; fix 115k / 45 |
| L5b `SERVER.md` catch-up | `coder` | a7ed26eacaac60cbf | accepted (12,079 w; §1.9 → pointer to DESIGN §6.1 by teco after Pass 12) | `falkor-chat/docs/SERVER.md` | `analyst` a9d6cd3e571c65434 → Pass 9 needs changes; fix `coder` a29dc36e7fcd625f2; Pass 12 approve | 293k / 106; gate 193k+222k; fix 117k / 41 |
| L5c `QUERIES.md` catch-up | `graph-dba` | a180eb2b7b147881a | delivered (29,382→26,762 w; AC-3: 9 hits, all `currentVersion:`/`defVersion:` property names — teco re-ran; §14.7–9, §15.3 added; suite 460/461 on a throwaway container) | `falkor-chat/docs/QUERIES.md` | `analyst` ad85541cd9ba720fd → Pass 10 needs changes (B1; m1–m3, N1; claims 1–4 confirmed); fix `graph-dba` a584c6b72da5fc3f3 (fresh) in flight | 362k / 121 |
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
- 2026-10-31: L5b follow-ups for L5d: `FALKORCHAT_AGENT_NAME` dead config (teco verified: only `config.py` reads it); step-budget-exhaustion envelope on `/input` lacks `error` (unverified; Pass 9 checks); K-063 substance possibly resolved by §1.5. L5b gaps relayed to L5c (document-delete/SUPERSEDES queries; §14.4 → pointer). Review file: Passes 8/9 append atomically (concurrent gates).
- 2026-10-31: L5c follow-ups for L5d: stale `test_queries.sh` §14.8 `SUPERSEDES.matchId` assertion (expects Node, plan is Edge By Index Scan; Pass 10 checks); lshBand defect stated in QUERIES §14.9. teco verified `reference` intact on `falkordb-dev` (3 WorkflowDef, 15 Product, 10 Step) and no leftover throwaway container. L5c cross-refs relayed to the Pass 9 check.
- 2026-10-31: Pass 9 placement ruling for L5s: SERVER §1.9 → fold two facts into DESIGN §6.1 "Prompt" bullet, §1.9 becomes a pointer — held until Pass 8 lands (DESIGN.md single-writer). Pass 9 proposes root `REVIEW.md` line on facts stated twice in the spec set (apply at close per §3.4).
- 2026-10-31: L2c carry-forwards: (a) root Component-docs `model-bench/` row naming its live spec → add to L6's file list; (b) L3c's `teco.md` close step must carry delegation defaults (implementer type; `graph-dba` for canonical Cypher), the slug-less review-doc rule, and step 4's two-way integrity check + enforcer-exists check.
- 2026-10-31: Pass 8 confirmed `_drive_loop` lock `6cd35d650b5d` (stale `71055f756280` since `15a9749b`) → L5d rewrites K-033's quoted value. P8-m5: L5d must file the `GRAPH.CONFIG` review orphan (DESIGN §13 cites it as tracked). Pass 8 proposes root `REVIEW.md` line on carried-over contradictions (apply at close). Review file pass order is 7, 9, 8 (atomic appends).
- 2026-10-31: Pending trivial teco edit after the DESIGN fix lands: SERVER §1.9 → one-line pointer to DESIGN §6.1 "Prompt" bullet (number kept); L5s verifies.
- 2026-10-31: L3c carry-forward (b) also gets Pass 11 m2: "A narrowing change is a prune or a *Do not report* addition." L6 brief carries root Component-docs model-bench row and joins root `AGENTS.md` serialization (Pass 11 call 1). Pending REVIEW.md proposals for close: Pass 8 (carried-over contradictions), Pass 9 (fact stated twice).
- 2026-10-31: teco decision on Pass 10 m3: `SERVER.md` §1.4's query column is the single home of route→query; `QUERIES.md` §14.4 → pointer. L5d files Pass 10 M1 (`test_queries.sh` §14.8 assertion can never pass; suite red 460/461 at `c0eeb348`) and M2 (lshBand dead; K-064 text still calls it the primary net; 0/455 live Documents have bands).
