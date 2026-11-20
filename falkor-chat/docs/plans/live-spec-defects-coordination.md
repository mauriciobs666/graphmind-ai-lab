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
| U1 K-091 band write + backfill | `tdd-engineer` | `a06eeb5521c81ccf1`; F6 fix `a7afd84687b716cf5` | accepted | code + tests | `analyst` `a20ec9d1e5163523e` → approve w/ suggestions (Pass 2) | 250k / 88 + 76k / 13 |
| U2 K-097 CAS design + ABA check | `graph-dba` | `a8a5e4080fa48fd8c` | accepted | `docs/plans/workflow-input-cas-graph.md` | `analyst` `a0052887e4da8f4a4` → approve w/ suggestions (Pass 2) | 215k / 50 |
| U3 K-097 implementation | `tdd-engineer` | `a35b78e4b204f8e10` | in-flight | code + tests | `analyst` → — | — |
| U4 K-098 GRAPH.CONFIG sizing | `graph-dba` | `a02427cdbdf32b4c8` | in-flight (revision) | `docs/plans/graph-config-defaults-graph.md` | `analyst` `a34f76a361e21f807` → needs changes | 240k / 77 |
| U5 K-098 set values | `devops` | — | queued (after U4 gate) | deployment config | `analyst` → — | — |
| U6 K-096 budget failure reporting | `tdd-engineer` | `ade689ec2f7ffb5cc` | accepted | code + tests | `analyst` `a5cf8134b051914e0` → approve w/ suggestions | 143k / 39 |
| U7 K-095 AGENT_NAME removal | `coder` | — | queued (after U6) | code | `analyst` → — | — |
| U8 K-093 MENTIONS | — | — | abandoned | stakeholder 2026-11-20: leave open, out of this batch | — → — | — |
| U9 K-094 quoting plan | `architect` | `ae4cd56c931f9746a` | accepted | `docs/plans/message-quoting.md` | `analyst` `a5276e9a50a55ecea` → approve (Pass 2) | 212k / 58 |
| U10 K-094 implementation | `tdd-engineer` | — | queued (after U9 gate) | code + tests | `analyst` → — | — |
| C1 K-091 close: spec update | `tdd-engineer` | `a1093354e3d6b012f` | accepted | QUERIES §14.9, DESIGN §5.1/§7.1, BACKLOG K-064 | `analyst` `a20ec9d1e5163523e` (`docs/reviews/document-lsh-bands-impl.md` § Close check) → approve w/ suggestions | 136k / 31 |
| C2 K-096 close: SERVER.md §1.4/§1.7 | `tdd-engineer` | `ade689ec2f7ffb5cc` | accepted | `docs/SERVER.md` | `analyst` `a5cf8134b051914e0` (close check) → approve | — |

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
- 2026-11-20 U1 fix: one helper, `update_detection.document_lsh_bands`, now used by ingest, detection and backfill (`teco` grep: no other pipeline). The raw-text mutant is killed. Delegate's counts: pytest 3676, queries 475/475. Gate review: `docs/reviews/document-lsh-bands-impl.md`.
- 2026-11-20 U4 delivered. The live THREAD/OMP counts are 8, not 4 (auto-sized). AOF/maxmemory (§4.3) fall outside the key list; scope goes to the stakeholder after the gate, along with the shared `falkordb-dev` restart. Six quirk entries were held back because my brief's fence blocked them; to be revisited after the gate. Evidence is being copied to `docs/reviews/graph-config-defaults-evidence/`.
- 2026-11-20 U1 gate: approve with suggestions. Five argument- and fixture-level mutants survived (F1/F2); F3 adds a band-count guard. The live half of F4 (restarting the servers, `--apply` per workspace, the live near-duplicate probe) is held for the K-091 close as a stakeholder decision. Live today: 522/522 `ws:agent-team` and 29/29 `ws:acme` documents carry no bands. Two `falkor-chat/REVIEW.md` lines are proposed, to be applied at the close.
- 2026-11-20 U1 Pass 2: all five survivors killed. F6 (the copy guard reads by regex; shell mutants C1/C2 survive) came from the fix itself, so this is the stopping signal: a fresh `tdd-engineer` takes it, since the original sits at 250k context, and `teco` verifies C1/C2 itself with no Pass 3.
- 2026-11-20 U1 accepted. `teco` verified F6: C1 and a fresh duplicate `SET_DOC_BANDS=` mutant each turn `test_query_copies.py` red, and the real file passes 7. Delegate's suites: pytest 3694, queries 475/475. Still pending for K-091's close: live spec, rollout (server restarts, `--apply` per workspace, live probe; stakeholder decision), and two `falkor-chat/REVIEW.md` lines.
- 2026-11-20 K-091 close step 4: the review's two final lines were applied verbatim to `falkor-chat/REVIEW.md` `## Always check`. Integrity check: the diff is exactly those two additions; no prunes were proposed or made; neither line narrows a check, so no `cobb` gate. Step 5 (HISTORY) goes to the spec writer after C-1/C-2.
- 2026-11-20 K-091 closed: spec, `REVIEW.md` lines and HISTORY entry committed together. Rollout (restarts, `--apply`, live probe) awaits the stakeholder. The K-091 backlog item is removed at milestone close (human applies).
- 2026-11-20 U6 delivered. `teco` re-ran the suite (3699). `teco`'s mutant (hardcoding the error string instead of reading the ctx note) survived 208 tests; it went to the gate as a question, since it may be behaviour-equivalent. Gate: `docs/reviews/step-budget-reporting-impl.md`.
- 2026-11-20 U6 gate: approve with suggestions. F1: M6 is not equivalent under a concurrent writer. F2: pin the error's source (`teco`'s survivor). Both went back to the implementer. F3: thread-reply resumes (`trigger` → `resume_workflow_run`) still report no reason; at the K-096 close it becomes a follow-up backlog item (`teco` default, stakeholder told) and HISTORY says 'the @mention-start path'. One root `REVIEW.md` line is proposed, to be applied at the close.
- 2026-11-20 U6 accepted. `teco` verified it after the killed mutant battery: no leftover mutant (the sweep still calls `_drive_or_fault`, and the read-back is gated on `failed` + `error is None`). Suite 3701 passed. `teco`'s hardcoded-error mutant now fails one test. No Pass 2: both findings were test additions with reviewer-specified probes, and `teco` checked them.
- 2026-11-20 stakeholder decisions: K-098's AOF/`maxmemory` part is split out into its own backlog item (hand-off section in the note). Rollout: one runbook that `teco` prepares and the stakeholder runs (server restarts, backfill, verify, optional FalkorDB restart), once K-097 and K-098 are approved.
- 2026-11-20 U4 gate: needs changes. B1: the AOF safe order can still wipe the volume (now split out). M1: 64 MiB fails a 10-hub-seed `hybrid_search`. M2: the restart is under-sequenced. The revision went back to `graph-dba` along with permission to write its quirks entries. The reviewer reported one pytest failure (`test_sweep_buckets_a_release_that_exhausts_the_budget_as_resumed_not_faulted`) on its own containers and called it pre-existing. That is K-096's new test, which passed in `teco`'s run at `068bc951` (3701), so the reviewer most likely ran against U3's in-flight tree. Re-check at U3 integration.
- 2026-11-20 K-096 close step 4: CC-1 fixed by `teco` (one citation in the SERVER.md sentence). The review's final line was applied to root `REVIEW.md` `## Always check`. Integrity check: the first apply truncated the wrapped proposal to its first line; caught on the diff and replaced with the full text. The diff is now exactly that one addition and does not narrow any check.
- 2026-11-20 K-096 closed: SERVER.md, the root `REVIEW.md` line and the HISTORY entry are committed. F3 is filed as `GAL-I02-K0003` (a registry line flushed). CC-2 (a run deleted mid-drive during a storefront reset) is left unnamed in the spec, per the reviewer.
