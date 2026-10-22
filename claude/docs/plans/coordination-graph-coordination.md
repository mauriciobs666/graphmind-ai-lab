# Coordination graph — docs-only coordination

> **Status:** active · **Owner:** `tico` · **Tracks:** —

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
