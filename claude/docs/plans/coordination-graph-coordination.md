# Coordination graph — docs-only coordination

> **Status:** active · **Owner:** `tico` · **Tracks:** — · **Item:** `GAL-F0002`

Docs-only chain: requirements → plan + graph note → independent review → revision. Hands the
whole remaining chain to `teco` at the first unit that needs an implementer (plan §4, S0 onward).

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 Requirements | `tico` | — | accepted | `claude/docs/requirements/coordination-graph.md` (committed 4f87c7ef) | stakeholder → Ready for design | — |
| U2 Implementation plan | `architect` | (prior session) | accepted | `claude/docs/plans/coordination-graph.md` | U4 | — |
| U3 Graph design note | `graph-dba` | (prior session) | accepted | `claude/docs/plans/coordination-graph-graph.md` (+ `claude/graph-dba/falkordb-quirks.md` crash quirk) | U4 | — |
| U4 Independent review of U2+U3 | `analyst` | a4a42445657f6f008 | gated | `claude/docs/reviews/coordination-graph.md` (Pass 1) | analyst → **needs-changes** (1 blocker, 8 major, 8 minor, 5 nit) | 160k tok |
| U5 Stakeholder answers to plan §6 Q1–Q6 | `tico` | — | accepted | requirements decision log 2026-10-21; to fold into U2 in U6 | — | — |
| U6a Plan revision (U4 findings + U5 answers) | `architect` | adb1bbc584d5ba9f6 | accepted | `claude/docs/plans/coordination-graph.md` Version 2 | U7 | — |
| U6b Graph note revision (M2, M3, M6 procedure, note minors) | `graph-dba` | a3e5c063d6117bc0b | accepted | `claude/docs/plans/coordination-graph-graph.md` | U7 | — |
| U7 Re-review of U6a+U6b | `analyst` | a4a42445657f6f008 | gated | `claude/docs/reviews/coordination-graph.md` Pass 2 | analyst → **needs-changes** (0 blocker, 4 major, 4 minor, 2 nit; all 22 Pass 1 findings fixed) | 248k tok |
| U8 Plan revision for Pass 2 (§3.2, §3.5, §3.7 + minors) | `architect` | adb1bbc584d5ba9f6 | accepted | `claude/docs/plans/coordination-graph.md` v2 in place | U9 | — |
| U8b Note: per-section KB lease + cancel ordering | `graph-dba` | a3e5c063d6117bc0b | accepted | `claude/docs/plans/coordination-graph-graph.md` §2.4/§6.4 | U9 | — |
| U9 Pass 3 (scoped to U8 sections) | `analyst` | a4a42445657f6f008 | gated | `claude/docs/reviews/coordination-graph.md` Pass 3 | analyst → **needs-changes** (0 blocker, 1 major P3-M1, 4 minor, 3 nit; all Pass 2 fixed) | — |
| U10a Plan: P3-M1 plan side + P3 minors/nits | `architect` | adb1bbc584d5ba9f6 | accepted | `claude/docs/plans/coordination-graph.md` | U11 | — |
| U10b Note: P3-M1 lease renew + post-POST cleanup | `graph-dba` | a3e5c063d6117bc0b | accepted | `claude/docs/plans/coordination-graph-graph.md` §6.4 | U11 | — |
| U11 Scoped confirmation of P3 fixes | `analyst` | a4a42445657f6f008 | accepted | `claude/docs/reviews/coordination-graph.md` Pass 4 | analyst → **approve-with-suggestions** (P4-m1 rides the S6 gate) | — |

## Notes

- 2026-10-21 — chain opened after the stakeholder asked for the analyst review; U5 runs in
  parallel with U4.
- 2026-10-21 — U5 answered: Q1 reformat once · Q1b parking-lot ideas ARE items (against the
  recommendation) · Q2 accept + flag · Q3 cap 6 · Q4 tico re-scopes later · Q5 stakeholder applies
  settings · Q6 keep RDB. U6 waits on U4's verdict.
- 2026-10-21 — U4 Pass 1: needs-changes. Stakeholder answered the review's two open questions:
  warn on **every** edit during a failure (stricter than proposed; still backed-off, non-blocking
  retries); approved milestones stay **fixed**. U6 split: U6a plan (architect), U6b graph note
  (graph-dba), in parallel on disjoint files; U7 = analyst Pass 2 over both.
- 2026-10-21 — U6b delivered (note Version 2): M2 §2.5 `statusToken`, M3 §2.2/§7.2(c)/§7.3, M6 §5.5 first-load, m1, m2, m8, n5; M4 graph side §6.4. Shared instance untouched (one PING). Section refs relayed to U6a.
- 2026-10-21 — U6a delivered (plan Version 2): all findings fixed, none declined (plan Appendix A). Plan Appendix B lists 5 follow-ups for the note → U6b resumed. New stakeholder Q7 (pull falkor-chat:K-070 in?) raised, non-blocking.
- 2026-10-21 — U6b follow-up delivered (plan Appendix B items 1–5 folded into note). Q7 answered: schedule falkor-chat:K-070 alongside S6, prerequisite of AC-8 acceptance, not folded in. U6a resumed to record Q7; then U7.
- 2026-10-21 — U6a closing edits done (Q7 resolved in plan §6, Appendix B resolved; no open questions). U7 dispatched to the same analyst (resumed).
- 2026-10-22 — U7 Pass 2: needs-changes (P2-M1..M4, local to plan §3.2/§3.5/§3.7). Stakeholder: urgent items may be inserted ahead of an approved milestone (flagged); the reformat converts open milestone-table rows into Milestone fields. U8 → same architect (resumed).
- 2026-10-22 — U8 delivered: all Pass 2 findings fixed, both 2026-10-22 decisions applied. Note needs the per-section lease (U8b, same graph-dba resumed) before U9.
- 2026-10-22 — U8b delivered: lease schema + acquire/release CAS (verified live on a throwaway container, 16-thread race ×5), cancel-path ordering, plus token-fenced mapping writes (not yet in the plan). U9 → same analyst (resumed).
- 2026-10-22 — U9 Pass 3: needs-changes, narrow (P3-M1 lease-loss after POST). Fixing P3-M1 and the minors/nits now (cheaper than deferring to implementation gates); U11 a short scoped confirmation.
- 2026-10-22 — U10b delivered: note §6.4 rules L1 renew-before-call, L2 clean-up-own-POST, L3 swap order (renew verified live on a throwaway container); new HTTP-timeout-under-TTL requirement relayed to U10a.
- 2026-10-22 — U10a delivered: all 8 Pass 3 findings fixed (L1–L3 cited, HTTP timeout < TTL, durable `started` marker, flock on failed.pending, hash-first first run, lint-checked milestone map). U11 → same analyst.
- 2026-10-22 — U11 Pass 4: **approve-with-suggestions — plan closed for implementation.** P4-m1
  (a client-timed-out KB POST that the server still committed: treat it as outcome-unknown, accept
  self-supersede on retry, test with a late-committing fake) carried to the S6 analyst gate.
  Chain terminal; every document committed together. Next unit (plan §4 S0/S1/S1b/S1c) needs
  implementers → whole remaining chain goes to `teco`, which resumes from this ledger.

## Addendum — work-item standard alignment (docs-only, 2026-11-20)

Opened by `tico` at the stakeholder's request; touches no code. The base chain above is closed.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| A1 Consistency audit vs work-item standard | `analyst` | af99b743e0528def6 | gated | `claude/docs/reviews/coordination-graph.md` Pass 5 | analyst → **needs-changes** (1 blocker, 5 major, 6 minor, 2 nit); D1–D4 answered 2026-11-20 | — |
| A2 Requirements edits (terminology, Item/Aliases FR/AC) | `tico` | — | delivered (uncommitted; FR-16/AC-15, FR-5/AC-4 feature need, terminology note) | both requirements documents | A5 | — |
| A3 Plan extension (header fields, item identity) | `architect` | ad678a043ab5b3b04 | delivered (uncommitted) | `claude/docs/plans/coordination-graph3.md` | A5 | — |
| A4 Graph note (identity key) | `graph-dba` | ad66c46cd139b3542 | delivered (uncommitted; note v3) | `claude/docs/plans/coordination-graph-graph.md` | A5 | — |
| A5 Re-review | `analyst` | a5ce726ff5b1723b7 | gated | `claude/docs/reviews/coordination-graph.md` Pass 6 | analyst → **needs-changes** (0 blocker, 2 major, 5 minor, 1 nit) | — |
| A6 Plan3 fixes (P6-M1, M2, m2, m3, retry) | `architect` | ad678a043ab5b3b04 | accepted | `claude/docs/plans/coordination-graph3.md` | A8 | — |
| A7 Note: drop stale-holder retry, unknown→unresolvable | `graph-dba` | ad66c46cd139b3542 | accepted | `claude/docs/plans/coordination-graph-graph.md` v3 | A8 | — |
| A8 Scoped confirmation | `analyst` | a5ce726ff5b1723b7 | accepted | `claude/docs/reviews/coordination-graph.md` Pass 7 | analyst → **approve-with-suggestions** (0 blocker, 0 major, 2 minor); both requirements stay Ready for design | — |

- 2026-11-10 — `coordination-graph2.md` (scheme-ID heading regex) was delivered without a ledger row; recorded here: extends the base plan, superseded in part by `coordination-graph3.md`.
- 2026-11-20 — A8 Pass 7: approve-with-suggestions; addendum closed. Next (implementation, plan3 S1d onward) goes to `teco`, which resumes from this ledger. Loose ends for `teco`: prune `.work-items/baseline.txt` lines for the two requirements files after a repo-wide `wi check`; flip `docs/plans/cross-initiative-coordination-ids.md` to `superseded` when `docs/plans/cross-initiative-coordination.md` (GAL-F0003) is written.

## Implementation phase (`teco`, opened 2026-11-20)

Resumed from the ledger above; plan 3 (`claude/docs/plans/coordination-graph3.md`) controls where it disagrees with the base plan and plan2. Base §4 step ids are reused; S1 is split (more than 5 files) into **S1a** (skeleton, config, keys, header parser, discover, ledger/review/requirements/kb parsers) and **S1e** (backlog parser, history, `lint`/`migrate`/`facts`/`history-coverage`). Carry-forward: plan3 §6 tests (12–13 included) ride S1d/S3/S7; P4-m1 (client-timed-out KB POST the server still committed = outcome-unknown) rides the S6 gate. Loose ends from the addendum: prune `.work-items/baseline.txt` lines for the two requirements files after a repo-wide `wi check`; flip `docs/plans/cross-initiative-coordination-ids.md` to `superseded` once `docs/plans/cross-initiative-coordination.md` is written. Docs in scope (scan): component `coordination-graph/{README,AGENTS,CLAUDE,docs/HISTORY}.md` (S11), `skills/coordination-graph/SKILL.md` + `skills/README.md` (S5), `teco.md`/`tico.md`/`audit-team.sh`/root+claude `AGENTS.md`/`agent-team.md` manual (S9, `tico` for the manual), `.work-items/baseline.txt`.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| S1a Skeleton + header + parsers | `coder` | `a7f89853b17710197` | gated (teco re-ran 221 passed; committed 739b6ec6) | `coordination-graph/` | `analyst` `acf7c62af8d2407a3` → **approve-with-suggestions** (`claude/docs/reviews/coordination-graph-impl.md` ## S1a: M1 unitKey cut, M2 review verdict, M3 `warnings`, M4 surviving mutants; m5 unknown tokens) | 260k tok / 77 tools |
| S1b Ephemeral fixtures | `devops` | `adb37bc40b660d102` | accepted (committed 6d8ff17c; smoke is delegate-observed) | `coordination-graph/tests/fixtures/live.py` | — (no gate per plan) | — |
| S1c cobb guard widening | `cobb` | `af73aec7d7092a340` | accepted (teco re-ran 3 hook probes; committed c85cd417) | `claude/cobb/hooks/guard-cobb-topic-writes.sh` (c85cd417) | `analyst` (light, `a8dbdb2623eb1817e`, `claude/docs/reviews/coordination-graph-guard.md`) → **approve-with-suggestions** (m1: `..` traversal passes `*` globs, pre-existing class in `guard-doc-writes.sh`; core fix left as follow-up for the stakeholder, not in S1c scope) | — |
| S1e Backlog/history/lint/migrate | `coder` | `abcd3b5ca1392d446` | gated (committed 8e151392) | `coordination-graph/coordgraph/*` | `analyst` `af8581f55d7eeffc7` (`## S1e` in the impl review) → **needs-changes** (1 blocker B1 migrate drops unreadable Priority, 3 major: M1 summary column dropped, M2 irrevocable mint with unfit titles/no pre-apply gate, M3 unpinned lint scope arg) | 283k tok / 48 tools |
| S1d `identity.py` | `tdd-engineer` | `abb320f06f2ecc5e2` | delivered (committed 7e78ee99; identity+lint seam tests 134 passed, teco re-ran; 15 mutants killed per delegate) | `coordination-graph/coordgraph/identity.py` | `analyst` → — | — |

## Addendum 2 — cross-initiative coordination plan (docs-only, 2026-11-20)

`GAL-F0003`; implementation sequencing against `teco`'s in-progress coordination-graph build is the plan's §5.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| B1 Plan | `architect` | a7a9c316260538807 | accepted | `docs/plans/cross-initiative-coordination.md` (+ ids note → superseded) | B2 | — |
| B2 Plan review | `analyst` | a90bd18ddde05a810 | gated | `docs/reviews/cross-initiative-coordination.md` Pass 3 | analyst → **approve-with-suggestions** (0 blocker, 4 major, 4 minor, 1 nit); stakeholder: dead ledgers summarised + teco closes them, `overlaps` kept | — |
| B3 Plan fixes (Pass 3) | `architect` | a7a9c316260538807 | accepted | `docs/plans/cross-initiative-coordination.md` (3 revision rounds: Pass 3, N1 + security, Pass 5 trims) | B4, B5 | — |
| B4 Security pass on FR-6 path scoping (§3.6, X3) | `security-expert` | a1c28d50d13490a2b | gated | `docs/reviews/cross-initiative-coordination-security.md` | security-expert → **approve-with-suggestions** (0 blocker, 3 major, 5 minor; all folded in) | — |
| B5 Scoped confirmation of Pass 3 fixes | `analyst` | a90bd18ddde05a810 | gated | `docs/reviews/cross-initiative-coordination.md` Pass 4–5 | analyst → **approve-with-suggestions** (Pass 5: 0 blocker, 0 major, 2 minor, 2 nit; trims taken) | — |
- 2026-11-20 — S1a delivered. Coder's calls needing owner confirmation (routed to the S1a gate): `slug_of` strips closed-set roles per kind (narrows note §2.2); status tally 72/986/`unknown` 104 vs note 74/911/41; `unitKey` leaves dashless unit cells whole; wrapper is `coordgraph.sh` (S5/S8/S11 invoke that, not `coordgraph`).
- 2026-11-20 — Addendum 2 closed: plan accepted for `teco`. Open risk for `devops`/stakeholder: who can write to the shared FalkorDB mirror key (a poisoned `:Source.path` is the main FR-6 attack route). `teco` applies the plan §6 dependency edits to the live ledger.
- 2026-11-20 — S1a gate rulings: `coordgraph.sh` and per-kind `slug_of` confirmed (spec edits owed: base §3.1, §3.4a(d), plan3 §3.4, note §2.2/§2.3); the tally gap is explained (not a defect; re-stamp note §2.5 from the frozen corpus file). M1 (add derived `unitLabel`) needs plan+note edits before S3 stores keys → U-spec1 (`architect`, `ab5b474f895ada38f`, plan + plan3) ∥ U-spec2 (`graph-dba`, `a177d3c5c4c51fac7`, note), both in flight; M2/M3/M4/m6/m7 are a coder fix round (S1a-fix) after S1e delivers (S1e is in flight on the same package).
- 2026-11-20 — U-spec1 (`architect`) delivered (uncommitted): base §3.1/§3.4a(d)/S1 row, plan3 §3.4 `Unit label` (rule re-derived over 1,296 live units, review's prior kept), §4 G3, §5. Awaiting U-spec2 (note) and S1e; then S1a-fix round and one analyst re-check of the spec edits against each other.
- 2026-11-20 — S1e delivered/committed. Implementer widened `migrate` beyond the one-line plan (parking-lot bullets, cpg bullets, status parentheticals, folded agent-plan tables, Milestone fields) — routed to the S1e gate as the question whether it exceeds plan3 §3.5 / the decision log. S1a-fix round dispatched fresh (`coder` `a3a83a34c78b65d5f`; the original coder carried 260k tokens): M1 `unitLabel`, M2, M3, M4, m6; m5 (status tokens) waits on U-spec2's ruling. Spec edits (U-spec1/U-spec2) still uncommitted.
- 2026-11-20 — U-spec2 (`graph-dba`) delivered (uncommitted): note v4 (slug rule, `unitLabel`, `warnings`, §2.5 tally re-stamped 986/104, PROFILE on throwaway container: no index needed). Rulings: `dispatched`/`executed` stay `unknown` — fix the cells, no aliases. One non-archived `unknown` remains: `H1` in `docs/plans/python-web-quirks-kb-coordination.md` (`executed 2026-11-10: …`) — another coordination's ledger, **left for S8a** (precondition: no `unknown` in a non-archived ledger). Exact-key precedence for unit lookup (40 labels in 10 ledgers collapse keys) put to `architect` (`ab5b474f895ada38f`, resumed) for plan3 §3.4; S1d's `resolve_need` must implement whatever it decides.
- 2026-11-20 — Architect adopted exact-key precedence (plan3 §3.4; 40 labels/10 ledgers: 25 exact-key, 15 label-only → still ambiguous); relayed to the S1a-fix coder. Dispatched in parallel: **S1d** `tdd-engineer` `a7710367497bc461a` (`identity.py`, only that file + its tests), **Pass 8** `analyst` `abb320f06f2ecc5e2` (spec round: plan/plan3/note v4 against each other and against code; appends `## Pass 8` to `claude/docs/reviews/coordination-graph.md`). Spec docs commit together after Pass 8.
- 2026-11-20 — S1a-fix (`coder` `a3a83a34c78b65d5f`, 427 passed, teco re-ran) committed; covers S1a gate M1 (code half), M2, M3, M4, m6; m7 left to S3 (write boundary); m5 left to S8a. The 19 changed review verdicts were hand-labelled by the implementer (3 refinements to the reviewer's rule); `small-model-catalog-sweep-impl.md` stays `needs-changes` by judgement. Re-check of the fixes is folded into the S1 closing gate (below).
- 2026-11-20 — S1e gate: needs-changes; parser/history/lint/wilib isolation hold, only `migrate` text handling + a lint test gap. Fix round S1e-fix → `coder` `a778ae294d93eb7d5` (fresh; owns migrate/history/test_lint). S1d (`a7710367497bc461a`) told the `Candidates` seam must carry reasons and that it owns that adapter in lint.py/shapes.py. **S4 must not run `migrate --apply` on real files until B1/M1/M2 are fixed, and S4's owner reviews the flagged titles before apply.** Coverage probe 15% is not a signal against the strict rule (268/444 IDs never appear in a history heading).
- 2026-11-20 — Ledger id correction: **S1d is `tdd-engineer` `abb320f06f2ecc5e2`; Pass 8 was `analyst` `a7710367497bc461a`** (my earlier note had them swapped). Pass 8: approve-with-suggestions (0 blocker/0 major, 5 minor, 2 nit; every stated figure reproduced). Fix round: plan-side → `architect` `ab5b474f895ada38f` (P8-m2..m6, n1, n2), note-side → `graph-dba` `a177d3c5c4c51fac7` (P8-m1, m3, m5, n1, n2, growth-stops), both resumed; S1d relayed the `U32~2` client rule, path-anchored local tokens and the shared pure precedence helper. Spec docs commit after these land (Pass 8's findings are minor; no re-gate beyond a scoped confirmation at the S1 close).
- 2026-11-20 — S1d delivered/committed (7e78ee99). Plan3 §6 tests 4, 5, 6, 9, 12 covered; 7/8 only at candidate level (delivery join is `history.py`'s); 13 is live (S3/S7). Open: plan3 §3.1(a)/test 5 say bare `K-065` is "ambiguous (exit 3)" but (a) qualifies with scope first, so exit 3 only when the referrer has no scope — architect wording fix owed at next plan3 revision. Full suite had 1 failing migrate test mid-flight (S1e-fix coder's file) — rerun when S1e-fix and the label fallback land. Next: S1 closing gate (`analyst`) over S1d + S1e-fix + label fallback.
- 2026-11-20 — Label fallback committed (d2a1899b; the 172+62-test modules in scope green, teco re-ran ledger/real_documents 62 passed). Dispatched in parallel: **S3a** `coder` `a8e05b46e959a4785` (store bootstrap/per-source replace/`rebuild --dry-run` + live tests on the ephemeral container only; must not touch `cli.py`, which the S1e-fix coder `a778ae294d93eb7d5` holds; S3b — sync/drift/rebuild parity/CLI wiring — follows) and **S5** `cobb` `a30db4979e8066381` (skill + `agent-maintenance` §1 template + catalog row; unbuilt behavior marked planned). S0 and S2 (cobb) wait: same agent, shared kaizen files. Gates owed: S3 → `graph-dba` then `analyst`; S5 → `analyst`; S1 close → `analyst`.
- 2026-11-20 — **PAUSED at the stakeholder's request (credit limit).** No new dispatches. In flight, allowed to finish: S1e-fix `coder` `a778ae294d93eb7d5`, S3a `coder` `a8e05b46e959a4785`, S5 `cobb` `a30db4979e8066381` — their results are unverified and uncommitted until integrated. Resume order: (1) integrate the three (re-run the full suite from `coordination-graph/`, check `git status --short .work-items` empty, commit each by path); (2) S1 closing `analyst` gate (S1d, S1e-fix, label fallback); (3) S3 gates (`graph-dba` then `analyst`), S5 gate (`analyst`); (4) S3b, S2, S0 (stakeholder applies the settings edit), S6, S7a/b, S4 (after migrate fixes + title review; stakeholder approves each `BACKLOG.md`), S8a, S8, S9, S10, S11. Cold-start fallback if agent ids no longer resolve: re-brief from this ledger and the gate reviews (`claude/docs/reviews/coordination-graph-impl.md`). Open follow-ups: guard `..` traversal (m1, `claude/docs/reviews/coordination-graph-guard.md`), `H1` unknown cell in `docs/plans/python-web-quirks-kb-coordination.md` (before S8a), plan3 §3.1(a) exit-3 wording, `.work-items/baseline.txt` prune, `cross-initiative-coordination-ids.md` supersession.
- 2026-11-20 — (paused) S5 `cobb` `a30db4979e8066381` delivered and committed ungated (description 762 chars < 1,024, teco-measured). Gate `analyst` owed. Plan↔CLI mismatches for the architect (CLI wins): `migrate --apply --titles-reviewed` not in the plan; `lint` reports `wilib-unavailable` as error where wilib is expected; `cli.py` never builds `Identity` so plan3 test 8's registry-backed join is unreachable from the CLI (S3b/S7 wiring); `history-coverage` missing from base §3.1 table; `query` has no owner step (S5 marked it S7a — confirm).
