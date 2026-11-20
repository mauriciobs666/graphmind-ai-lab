# Live-spec defects K-091, K-093..K-098 — coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-091, K-093, K-094, K-095, K-096, K-097, K-098

Goal: work the seven open items the live-specs review filed (`docs/BACKLOG.md`, each item's own
body is its brief: owner, done-condition, test strategy). K-092 is already closed
(`docs/HISTORY.md` 2026-11-09). Each item is one feature for the live-spec close (root
`AGENTS.md`): its spec sections (cited in the item) are updated at its close, not by its build unit,
then checked by `analyst` in `docs/reviews/live-spec-defects-impl.md` (one section per item).

Sequencing: implementer units run one at a time — they share `repository.py`/`services.py` and the
`ws:test` suite key. Advisory/design units run alongside them on scratch graph keys only.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 K-091 band write + backfill | `tdd-engineer` | `a06eeb5521c81ccf1` | in-flight (fix: surviving mutant) | code + tests | `analyst` → — | 196k / 66 |
| U2 K-097 CAS design + ABA check | `graph-dba` | `a8a5e4080fa48fd8c` | accepted | `docs/plans/workflow-input-cas-graph.md` | `analyst` `a0052887e4da8f4a4` → approve w/ suggestions (Pass 2) | 215k / 50 |
| U3 K-097 implementation | `tdd-engineer` | — | queued (after U2 gate, U1) | code + tests | `analyst` → — | — |
| U4 K-098 GRAPH.CONFIG sizing | `graph-dba` | `a02427cdbdf32b4c8` | in-flight | `docs/plans/graph-config-defaults-graph.md` | `analyst` → — | — |
| U5 K-098 set values | `devops` | — | queued (after U4 gate) | deployment config | `analyst` → — | — |
| U6 K-096 budget failure reporting | `tdd-engineer` | — | queued (after U1) | code + tests | `analyst` → — | — |
| U7 K-095 AGENT_NAME removal | `coder` | — | queued (after U6) | code | `analyst` → — | — |
| U8 K-093 MENTIONS | — | — | abandoned | stakeholder 2026-11-20: leave open, out of this batch | — → — | — |
| U9 K-094 quoting plan | `architect` | `ae4cd56c931f9746a` | accepted | `docs/plans/message-quoting.md` | `analyst` `a5276e9a50a55ecea` → approve (Pass 2) | 212k / 58 |
| U10 K-094 implementation | `tdd-engineer` | — | queued (after U9 gate) | code + tests | `analyst` → — | — |

Documentation impact (for each item's close, not its build unit): the DESIGN/SERVER/QUERIES
sections each item names; `docs/HISTORY.md` entry per item; `docs/BACKLOG.md` item removed at
milestone close (human applies); `server/.env.example` and start scripts for K-095 (build unit).

## Notes

- 2026-11-20 stakeholder decisions: K-096 report it; K-095 remove the dead copy; K-093 leave open (out of batch); K-094 add quoting.
- 2026-11-20 U2: ABA reproduced by `graph-dba` (worse than R-1; note §1). The note also added a dated entry to `claude/graph-dba/falkordb-quirks.md`. Gate evidence is preserved under `docs/reviews/workflow-input-cas-evidence/`.
- 2026-11-20 U9: the plan defers surfacing `quotedId` on `read_thread`/§9 reads and adding an MCP read-back. That becomes a follow-up backlog item at K-094's close.
- 2026-11-20 U2 gate: `docs/reviews/workflow-input-cas.md`. The ABA reproduced independently. `GET` exposes `ctxVersion` (an additive public field, to be named in HISTORY at close). Mixed deploy needs only an operator note. M1/m1–m5 were routed back to `graph-dba`. n5: U3 and U6 both edit `submit_workflow_input`, so they run serially.
- 2026-11-20 U2 Pass 2: approve with suggestions. The P2 findings came from the fixes themselves, not from the original note, so this is the stopping signal: P2-1..4 go back for one final edit, verified by `teco` with no Pass 3.
- 2026-11-20 U2 accepted. `teco` checked the P2 edit: the note is 2,520 words, the invariant is scoped to existing runs (§2 and the §4 hand-off), and the mutant table matches P2-4. P2-3's `parkedAt` claim was checked against `find_due_wait_candidates`/`_wait_due_at`. The quirks-file diff is graph-dba's one entry.
- 2026-11-20 U9 gate: `docs/reviews/message-quoting.md`. Six premises verified by execution. M1–M3/m1–m4 were routed back to `architect`. The review proposes two `falkor-chat/REVIEW.md` lines; they are applied verbatim at K-094's close (step 4).
- 2026-11-20 U1 delivered, with a solo re-run by `teco` (pytest 3672, queries 475/475). The delegate's two earlier failing query runs coincided with a concurrent `cpg/` pytest. `teco`'s mutant (ingest shingles the raw text, not the normalized text) survived 807 tests, so it went back to the delegate before the gate.
- 2026-11-20 U9 accepted (Pass 2 approve). N3 was fixed by `teco`, a one-sentence edit in the plan's §4. N4 (truncating the echoed `quoted_id` in the error detail) goes to U10's brief as optional.
