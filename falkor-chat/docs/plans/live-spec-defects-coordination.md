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
| U3 K-097 implementation | `tdd-engineer` | `a35b78e4b204f8e10` | accepted | code + tests | `analyst` `a0052887e4da8f4a4` → approve w/ suggestions | 179k / 41 |
| U4 K-098 GRAPH.CONFIG sizing | `graph-dba` | `a02427cdbdf32b4c8`; Pass 2 fixes `a4c4436f264289eb8` | accepted (`dac41d6e`) | `docs/plans/graph-config-defaults-graph.md` | `analyst` `a34f76a361e21f807` → approve w/ suggestions (Pass 3) | 240k / 77; gate 294k / 107 |
| U5 K-098 set values | `devops` | `a827c5e8445b81892` | accepted (`8a49ebe9`) | deployment config | `analyst` `a41507a63d6beefa4` → approve (Pass 2, `docs/reviews/graph-config-defaults-impl.md`) | 169k / 38 |
| U6 K-096 budget failure reporting | `tdd-engineer` | `ade689ec2f7ffb5cc` | accepted | code + tests | `analyst` `a5cf8134b051914e0` → approve w/ suggestions | 143k / 39 |
| U7 K-095 AGENT_NAME removal | `coder` | `aac0dbeca19db649f` | accepted (`940dc1e2`); closed | code + `tests/test_agent_name_handoff.py` | `analyst` `a222dd293d994bcc9` → approve (Pass 2, `docs/reviews/agent-name-removal-impl.md`) | 91k / 17 |
| U8 K-093 MENTIONS | — | — | abandoned | stakeholder 2026-11-20: leave open, out of this batch | — → — | — |
| U9 K-094 quoting plan | `architect` | `ae4cd56c931f9746a` | accepted | `docs/plans/message-quoting.md` | `analyst` `a5276e9a50a55ecea` → approve (Pass 2) | 212k / 58 |
| U11 `test_queries.sh` assert-helper SIGPIPE fix (K-097 review F2) | `tdd-engineer` | `a66f81e5956af6364` | accepted (`8d629cad`) | `scripts/test_queries.sh` | `analyst` `a0052887e4da8f4a4` → approve (`docs/reviews/workflow-input-cas-impl.md` § U11 gate) | — |
| U12 `hybrid_search` bounded rewrite (K-098 note §3.1) | `tdd-engineer` | `a65b5e6619529c62f` | delivered (uncommitted; teco solo suite running, teco mutant `relatedContext` m.text→seed.text killed 2) | code (spec at close: QUERIES §6, DESIGN §8 + graph-dba `coverage.py` re-run) | `analyst` `a68eb72ff880d9934` → — (`docs/reviews/hybrid-search-bounded.md`) | 174k / 52 |
| U10 K-094 implementation | `tdd-engineer` | `ab2a3edf3705a0131` | accepted (`496c5d40`); closed `2cdff113` | code + tests | `analyst` `a5276e9a50a55ecea` → approve (`docs/reviews/message-quoting-impl.md`) | 236k / 64 |
| C1 K-091 close: spec update | `tdd-engineer` | `a1093354e3d6b012f` | accepted | QUERIES §14.9, DESIGN §5.1/§7.1, BACKLOG K-064 | `analyst` `a20ec9d1e5163523e` (`docs/reviews/document-lsh-bands-impl.md` § Close check) → approve w/ suggestions | 136k / 31 |
| C2 K-096 close: SERVER.md §1.4/§1.7 | `tdd-engineer` | `ade689ec2f7ffb5cc` | accepted | `docs/SERVER.md` | `analyst` `a5cf8134b051914e0` (close check) → approve | — |
| C3 K-097 close: DESIGN §6.2, QUERIES §12 | `tdd-engineer` | `a35b78e4b204f8e10` | accepted (`89306a90`; close `79aeadd8`) | `docs/DESIGN.md`, `docs/QUERIES.md` | `analyst` `a0052887e4da8f4a4` (close check) → approve w/ suggestions | — |
| C4 K-098 close: DESIGN §9/§10/§13, QUERIES §18.7, SERVER §1.5 | `graph-dba` | `a4c4436f264289eb8` | accepted (closed `e2f42594`) | live spec + backlog item | `analyst` `a41507a63d6beefa4` (close check) → Pass 1 needs changes, Pass 2 approve (`docs/reviews/graph-config-defaults-impl.md` `## Close check`) | check 218k / 41 |
| C4m K-098 manual `docs/manuals/operations.md` | `tico` | `a77e91699aa4e5616` | accepted | manual | skipped (trivial; `teco` ran `ps`/`logs` without `--env-file`: rc 1 each) | 58k / 5 |

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
- 2026-11-20 U3 delivered. `teco` re-ran: pytest 3709 (K-096's sweep test passes, so the K-098 reviewer's failure was a mid-edit tree); queries 486/486 ×2. The harness A2/A3 now give 409 (delegate's run). Diff gate: `docs/reviews/workflow-input-cas-impl.md`.
- 2026-11-20 U3 gate: approve with suggestions. F1: the sweep's `ctx_version` binding is unpinned. F2: `test_queries.sh`'s `echo | grep -q` under `pipefail` gives false results, which `teco` reproduced 20/20 (a 300 KB output with an early match is reported as a miss, so `assert_not_contains` falsely passes; 57 uses). Stakeholder (2026-11-20): fix F2 now, before further work. Order: U3 fixes (no `test_queries.sh` edits) → U3 commit → U11 helper fix and full re-run, which re-checks U3's negative cases → U10.
- 2026-11-20 U3 accepted: F1 (test 4b, red under the sweep re-read mutant), F3 and N1 closed. `teco` re-ran pytest: 3710. Its `test_queries.sh` negative cases are re-verified by U11.

## Pause checkpoint — 2026-11-20 (stakeholder: credit limit)

No new dispatches. The three in-flight units finish, and their results are recorded only, with no gate dispatched. Agent ids resolve only inside this session. Each line gives the cold-start fallback for a new session.
- **U11 helper fix** (`a66f81e5956af6364`). Cold start: re-brief a `tdd-engineer` from `docs/reviews/workflow-input-cas-impl.md` F2, using `git diff -- scripts/test_queries.sh` as the state. Next step: gate with `analyst`, then commit.
- **C3 K-097 spec** (`a35b78e4b204f8e10`). Cold start: `git diff -- docs/DESIGN.md docs/QUERIES.md` against plan §4 and the impl review's "spec close list". Next step: close check by `analyst`, append to `docs/reviews/workflow-input-cas-impl.md`. Then apply that review's proposed `falkor-chat/REVIEW.md` lines (one *Evidence bar* line, which U11 makes prunable) and write the HISTORY entry, which must name the public `ctxVersion` key on `GET /workflow-runs/{runId}`.
- **U4 K-098 revision** (`a02427cdbdf32b4c8`). Cold start: `docs/reviews/graph-config-defaults.md` against the current note. Next step: Pass 2 by the same reviewer. The persistence hand-off section becomes a new backlog item.
- **Not started:** U10 K-094 implementation (after U11), U7 K-095, U5 K-098 config, the closes for K-094/K-095/K-098, and the stakeholder-run rollout runbook (K-091 backfill + K-097 restart + optional K-098 restart).
- 2026-11-20 U11 delivered under the pause: both helpers plus 22 other early-exit pipe sites fixed (`<<<` here-strings, `grep -m1`, and `--` before needles). Probe: 12,012 wrong verdicts on the original helpers, 0 on the fixed ones. Suite 486/486 before and after, so no hidden failures; pytest 3710. Each K-097 §12.13 negative goes red when its condition is planted (P1/P2/P3). The probe evidence is committed at `falkor-chat/docs/reviews/test-queries-sigpipe-evidence/`. Once U11 is committed, the K-097 review's proposed *Evidence bar* line is prunable (don't apply it). Side note, out of scope: `test_queries.sh` line 114's `|| true` hides a failed `bootstrap_schema.sh`, which may deserve its own item.
- 2026-11-20 U4 revision delivered under the pause. The cap stays 64 MiB and `hybrid_search` is bounded in the query (new unit U12, a prerequisite of K-093's writer). `FALKORDB_THREAD_COUNT` lives in `config/falkordb.env`; `MAX_QUEUED_QUERIES` is 320 on dev and 128 on prod. A rehearsed restart procedure is in §5, the persistence hand-off in §8, and 7 quirks entries were added. Evidence: `docs/reviews/graph-config-defaults-evidence/graph-dba-r2/`. For Pass 2: the note grew from 4,191 to 4,926 words (run the growth-stops check), and its rollback fallback cites `git show HEAD:` (should be a pinned sha).

- 2026-11-20 resumed (stakeholder: full speed). Wave 1: `a0052887e4da8f4a4` U11 gate then C3 close check (sole `ws:test` user); `a34f76a361e21f807` U4 Pass 2 (scratch keys only); U7 `coder` (no FalkorDB suites). U10 waits on the U11 verdict and `ws:test`.
- 2026-11-20 U7 delivered: 128 FalkorDB-free tests, 5 mutants killed; `teco`'s default-value mutant (`:-Assistent` in `start_demo.sh`) killed. Full suite deferred to a solo run.
- 2026-11-20 U4 Pass 2 (`29c7582e`): needs changes, with the stopping signal. All four findings come from the revision: P2-M1 (`docker stop` on rollback overwrites the good `dump.rdb`, reproduced), m1 (U12→K-093 ordering untracked), m2 (`--env-file` on every compose command), m3 (§8 to BACKLOG). The Pass 3 stopping condition is the fixer's done-condition. A fresh `graph-dba` takes it, since the original is at 397k context. M1 was verified by execution: the bounded query uses 546 KiB in 16 ms and returns the same seeds, scores and related messages. Pass 2 replaces the proposed root `REVIEW.md` restart line; apply it at the K-098 close.
- 2026-11-20 U11 gate approve; K-097 close check approve w/ suggestions (`bb7ef112`, evidence `c7932fe0`). K-097 step 4: the two final lines (helper-probe, CAS-injection) applied verbatim to `falkor-chat/REVIEW.md` `## Always check`. Integrity: the diff is exactly those two additions; no prunes (the *Evidence bar* SIGPIPE line was never applied, and no enforcer exists for a prune); neither narrows, so no `cobb` gate. Step 5 HISTORY → `a35b78e4b204f8e10`. Filed for closeout: one low-priority shell-harness item (the line-114 `|| true`, plus `| grep -q` in `verify_catalog.sh`/`start_demo.sh`).
- 2026-11-20 U7 gate approve w/ suggestions. F1: the all-keys test counts prose (a dead `USER_ID` passes), so it goes back to the `coder` to count AST loads. SERVER.md §1.3: delete the row at the close. One proposed `REVIEW.md` line (name-read guards).
- 2026-11-20 K-097 closed (`79aeadd8`): HISTORY names the public `ctxVersion` key and the all-server restart; `REVIEW.md` lines committed with it. `teco` softened 'the only tests that kill' to 'the tests that kill' (the writer's evidence covered the targeted set only).
- 2026-11-20 U7 Pass 2: approve. The coder's departure (an alias counts only when loaded) was correct, and the reviewer's Pass 1 recipe was wrong; confirmed by execution. Committed `940dc1e2`; the FalkorDB-backed suite over it rides U10's full run, re-checked at U10 integration. The close writer is the same `coder`.
- 2026-11-20 K-095 closed: SERVER.md §1.3 row deleted, close check approve, one `REVIEW.md` *Always check* line applied verbatim (three-line diff, an addition, not narrowing), HISTORY entry.
- 2026-11-20 U4 Pass 2 fixes delivered (`91c53a2d`). The note is 4,666 words (`teco` `wc -w`), and the rollback is rehearsed on both branches (diff 0). `GAL-I02-K0004` is filed (registry flushed and committed). There are two new graph-dba quirks entries, but `claude/graph-dba/falkordb-quirks.md` also holds an uncommitted NUL-byte entry from another writer, so the file is held out until that owner is known. Pass 3 → `a34f76a361e21f807`.
- 2026-11-20 U4 Pass 3: approve w/ suggestions. The stopping condition was met and re-rehearsed independently on a named volume. P3-m1 (a `redis-cli` hang skips the kill fallback, so it needs `timeout 30`) and the abort-unpause nit were applied by `teco` as one-token edits (`dac41d6e`). For the K-098 close: DESIGN §10 (tuning, TIMEOUT, Observability bullets), §9 (Backfill/import row), §13 (open question → resolved), QUERIES §18.7; three root `REVIEW.md` lines and one `falkor-chat/REVIEW.md` line, verbatim in the review. The quirks entries `GAL-I02-K0004` cites must be committed with K-098.
- 2026-11-20 U5 delivered: 19 tests, 20/20 mutants; `teco`'s env-file `QUERY_MEM_CAPACITY` typo mutant killed (5 red). Runbook hazard to carry: under `start_falkordb.sh -d`, a bad value exits 0 and `--rm` erases the abort log; on no PONG, re-run the foreground dry run.
- 2026-11-20 U10 delivered. `teco` solo: pytest 3762 (3743 + U5's 19), queries 495/495. `teco`'s column-swap mutant (`authorFound`/`quotedFound` in the subsequent RETURN) killed, 7 red. Plan misses: the storefront partition guard and the method-level index pytest. N4 not taken. Spec close list in the implementer's report, to be checked by the gate.
- 2026-11-20 U5 gate: approve w/ suggestions. m1 (Dockerfile compose line), m2 (script and compose env parsers diverge), m4 and nits went back to `devops`. For the K-098 close: `docs/manuals/operations.md:58-59` → `tico`; SERVER §1.5 Layout gains `falkordb.env`; a backlog item for `devops` on start scripts not detecting a `-d` container exit; one root `REVIEW.md` line (two-parser probe). The note's own parse block is unsafe as written (planted `a[$(…)]` key executed), and devops's allowlist is what makes it safe.
- 2026-11-20 U10 gate: approve. Both departures were improvements, verified by execution, and all 16 suite copies match the sent text. Two `falkor-chat/REVIEW.md` lines are proposed (the copy-sweep line and a revised index-EXPLAIN line); apply them at the K-094 close. Committed `496c5d40` with evidence `docs/reviews/message-quoting-evidence/impl/`. U12 dispatched (owns `ws:test`); the K-094 close spec writer runs on docs alongside it.
- 2026-11-20 K-094 close check: approve w/ suggestions. C1 (a DESIGN §5.3 clause read a replay as a caller error) was fixed by `teco` as a one-clause pointer. C2 (`teco` had committed a mutated package copy plus two full suite copies as evidence) was removed in `9247276f`. Both `REVIEW.md` lines were applied verbatim (+2, additions). HISTORY → `ab2a3edf3705a0131`.
- 2026-11-20 K-094 closed (`2cdff113`). HISTORY's figures are the writer's own (3743, 495/495, 38 mutants: `teco` counted the report's table as 14+5+10+9).
- 2026-11-20 U5 Pass 2: approve. devops's stricter value class was confirmed by execution (the reviewer's class let edge spaces through, which compose trims). M11/M33 equivalent, M32 same outcome with a different message (nit). Final root `REVIEW.md` two-parser line (revised) is in the review. Committed `8a49ebe9`.
- 2026-11-20: C4 close check returned needs changes, narrowly (C1 DESIGN §10 rollback sentence is compose-false; C2 QUERIES §18.7 condition). It found the five K-098 REVIEW.md proposals non-narrowing, so no cobb gate is needed. Fixes sent to `a4c4436f264289eb8`.
- 2026-11-20: U12 delivered (3797 passed, 495/495, as reported). Its gate moved from graph-dba to analyst because U12 disputes graph-dba's own §3.1 (the scoped plan does not use the Thread index). ws:test passes from U12 to teco's solo run.

## Rollout runbook (stakeholder-run)

Run this once every unit above is committed. The code is not live until you do: the only
running server (`:8200`, started 2026-11-19 01:42) predates K-091 (`4d35b84e`), K-096
(`068bc951`), K-097 (`7cff155b`), K-095 (`940dc1e2`), K-094 (`496c5d40`) and U12, and
`falkordb-dev` still runs the image's own module args.

**A. Optional: put K-098's FalkorDB config live.** Follow `docs/plans/graph-config-defaults-graph.md`
§5 Step 4, steps 1–9, exactly as written there (dry run, quiesce, inventory, SAVE, stop with log
and backup copy, start, inventory diff, rollback, resume). Step 9's restart of the servers on the
current code is part B, so after step 8 go straight to B.
- Hazard: `start_falkordb.sh -d` cannot report a bad module arg (`GAL-I02-K0006`). If step 6's
  `redis-cli PING` never answers `PONG`, the container has already exited and been removed.
  Re-run step 1's foreground dry run to see the error, and use step 8 to roll back.
- If you skip A, nothing below depends on it. K-098's config takes effect at your next restart
  of `falkordb-dev` through `start_falkordb.sh`.

**B. Restart every falkor-chat server on the current code.** K-097's CAS needs every process
serving a workspace on the new code at once, never a mix of old and new.
1. Stop each running server, `:8200` included. Check with `ss -ltnp | grep -E ':(8000|8100|8200) '`
   that none is left.
2. Start each one again the way it was started (`./scripts/start_agent_team.sh` for `:8200`;
   `start_server.sh` / `start_demo.sh` for any others).
3. Run `./scripts/verify_workflows.sh` and `./scripts/verify_catalog.sh`. Both must exit 0.
4. The `falkor-chat-agent-team` MCP connection drops on restart. Reconnect it with `/mcp`.

**C. K-091 LSH-band backfill.** Do this only after B, per the script's own header.
1. Dry run: `./scripts/backfill_document_lsh_bands.sh acme agent-team nlq-eval pwq-test`.
   Pass the bare workspace id, not `ws:acme`; the prefixed form fails with
   `ws:ws:…: Invalid graph operation on empty key`. On 2026-11-20 teco's dry run listed 29 /
   535 / 12 / 518 documents without bands. `demo`, `eval` and `k069probe` listed 0.
2. Apply: the same command with `--apply`.
3. Verify: re-run the dry run. Every workspace must list 0. The script is idempotent, so a
   non-zero count (a document an old server ingested in between) is fixed by re-running
   `--apply`.
