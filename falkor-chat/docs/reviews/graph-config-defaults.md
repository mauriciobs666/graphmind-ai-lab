# Review: GRAPH.CONFIG defaults for production (K-098 graph-dba note)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-098

## Scope and verdict

**Reviewed:** `falkor-chat/docs/plans/graph-config-defaults-graph.md` (first version, so the
growth-stops check does not apply yet), against `falkor-chat/docs/BACKLOG.md` `### K-098`,
`falkor-chat/docs/DESIGN.md` §9–§11 and §13, `falkor-chat/docs/QUERIES.md`,
`falkor-chat/docs/SERVER.md`, `falkor-chat/docs/test-reports/capacity-report.md`,
`falkor-chat/scripts/start_falkordb.sh`, `falkor-chat/compose.yaml`,
`claude/graph-dba/falkordb-quirks.md`, the image entrypoint (`image_run_sh.txt`) and the query texts
in `falkor-chat/server/falkorchat/repository.py` as of `4d35b84e`. Root `REVIEW.md` and
`falkor-chat/REVIEW.md` were applied.

**Evidence:** the graph-dba run's files were copied unchanged to
`falkor-chat/docs/reviews/graph-config-defaults-evidence/`. My own probes are in its `analyst/`
subdirectory. They ran in four throwaway containers I started myself (`analyst-k098-a…d`, same image
`falkordb/falkordb:v4.18.11`, ports 16591–16594). All four have been removed. Against `falkordb-dev`
I ran only read-only commands: `docker inspect` and `INFO persistence`/`INFO memory`/`CONFIG GET`.
I did not touch `ws:test` or `reference-test` on the shared instance.

**Verdict: needs changes.**
- **B1:** the AOF "safe order" in §4.3 / §5 Step 3 can still wipe the volume.
- **M1:** 64 MiB fails a legitimate sub-second `hybrid_search` once entity extraction is filling
  `MENTIONS` edges.
- **M2:** the restart in §5 Step 4 has no save check, no dry run, no rollback, and a verify step
  that cannot detect data loss.

The `FALKORDB_ARGS` mechanics are right and verified. So are the `ASYNC_DELETE` trap, the
`OMP_THREAD_COUNT 1` direction, the queue arithmetic, and suite compatibility. Once M1 and M2 are
fixed, Steps 1, 2, 4 and 5 can ship. §4.3 should be split out (see the answer to question 3 in
*Open questions*).

CPG: considered, not relevant — the questions are engine configuration and query memory, which only a live engine can measure; no `cpg_falkorchat` call-graph query answers them.

## Findings

### B1 (blocker) — the AOF wait condition can read "done" before any rewrite has run, and the trap becomes permanent at the next clean stop

Note §4.3 says to wait for `aof_rewrite_in_progress:0` and `aof_last_bgrewrite_status:ok`, then add
`--appendonly yes` and restart. **Verified:** I ran `BGSAVE` on a 935 MB dataset, then
`CONFIG SET appendonly yes`. Redis only *schedules* the rewrite (`aof_rewrite_scheduled:1`,
`aof_rewrites:0`, `aof_last_rewrite_time_sec:-1`). Both of the note's fields already read "done"
at that point (`analyst/aof_scheduled_rewrite_probe.txt`). `falkordb-dev` runs RDB saves often
(`save … 60 10000`, `rdb_saves:51` this uptime). So a faithful operator can hit exactly this window,
restart with the flag set, and get the empty start the note warns about.

That start is worse than the note says. **Verified:** after the empty start, one `docker stop`
writes the empty dataset over `dump.rdb`. A restart *without* the flag then also comes up empty
(`DBSIZE 0`, `analyst/aof_trap_probe.txt`). Nothing can be recovered, and the procedure has no
backup step.

**Fix (in the split-out item, see question 3):**
1. Copy the volume first, e.g. `docker run --rm -v falkordb-data:/d -v <dir>:/b alpine cp -a /d /b`.
2. Gate the restart on all of these: `aof_rewrite_scheduled:0`, `aof_rewrite_in_progress:0`,
   `aof_rewrites` ≥ 1 (or `aof_last_rewrite_time_sec` ≠ -1), `aof_last_bgrewrite_status:ok`, and an
   `appendonly.aof.manifest` present in `appendonlydir/`.
3. The gate's positive control is the BGSAVE-in-progress state above. Run that state, and the gate
   must stay red in it.

### M1 (major) — `QUERY_MEM_CAPACITY` 64 MiB is below the worst case of the read it was sized for

§3.1 calls 18.4 MiB "the extreme case" and claims 3.5× margin. But in `ws:deep` only one of the k=10
ANN seeds is the deep, hub-mentioning message. `build_deep.py` embeds 20 tail messages beside 2,000
random noise vectors, so the other nine seeds are noise. The `collect(DISTINCT related)` expansion
aggregates **per seed**, so its memory scales with **k × fan-in**. **Verified** with the verbatim
`repository.hybrid_search` text and k=10 (`analyst/multi_seed.jsonl`):

| Shape (10 near-duplicate seeds) | Min. cap | Uncapped | At 64 MiB |
|---|---|---|---|
| each seed mentions an entity with 100k fan-in, short threads | **68.7 MiB** | 367 ms | **fails** |
| all seeds at depth ~100k, 50k-hub (`ws:deep` shape) | 49.1 MiB | 12.4 s | passes (1.3×) |
| expansion alone, 10 seeds × 50k hub | 34.5 MiB | 179 ms | passes |

The walk does **not** add up across seeds: 14.9 MiB for 10 deep seeds, the same as for one. The
expansion does. A 100k-message workspace is the capacity report's own sizing unit, and an entity
that most of its messages mention is ordinary once extraction lands. In that workspace, a read
finishing in under 0.4 s fails on the cap. The failure reaches the responder's RAG path
(`responder.py:104`).

**Fix:**
- Re-size against `k × max fan-in × ~70 B` plus the walk, or bound the expansion inside the query.
  The query route is a design decision for graph-dba; I have not tested any rewrite. If the value is
  raised (128 MiB, say), restate the `THREAD_COUNT ×` product (8 × 128 MiB = 1 GiB) against shard
  RAM.
- Make the next measurement a **coverage probe over the axes the read varies on**, not a longer list
  of shapes: k (1–10), seeds per thread (spread vs clustered), seed depth, entity fan-in,
  `MENTIONS` per seed (the synthetic graphs use 2), and scoped vs unscoped. Run it as a round before
  the value is chosen, not inside the step that sets it.

### M2 (major) — the §5 Step 4 restart is not sequenced safely enough for the shared instance

**Verified** with `docker inspect falkordb-dev` and read-only `INFO`:
- **No dry run.** The container runs with `--rm`, so its logs vanish on stop. The edited
  `start_falkordb.sh` always mounts `falkordb-data` (`start_falkordb.sh:51,59`), so it cannot be
  dry-run on a spare port. That would open the live volume twice, which `compose.yaml:6-8` warns
  corrupts it.
- **No save check.** `docker stop` allows 10 s before SIGKILL. The note relies on the RDB written
  at shutdown but never checks it. Right now `used_memory` is 483 MB against 54 MB RSS, so most of
  the instance is swapped out and the save can be slower than the last `bgsave` (2 s).
- **No rollback.** If the new string aborts the start (cf. `ASYNC_DELETE 1`), the instance is down
  and the logs are gone.
- **Step 5 cannot fail on data loss.** `verify_workflows.sh <ws>` checks two workflow defs in one
  workspace. A lost `cpg_falkorchat`, a lost `ws:agent-team` or lost `ws:demo` messages would still
  pass.

**Fix:**
1. Dry-run the exact edited `docker run` line against an anonymous volume on a spare port, or add a
   `FALKORDB_VOLUME` override.
2. Before stopping: run `redis-cli SAVE`, check that `LASTSAVE` advanced, and record `GRAPH.LIST` plus
   a per-graph `MATCH (n) RETURN count(n)` inventory.
3. Stop with `docker stop -t 120`.
4. State the rollback:
   `FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000" ./scripts/start_falkordb.sh -d`.
5. Step 5 compares the inventory before and after. Its positive control is a throwaway graph that
   exists only in the "before" snapshot.

### m1 (minor) — `THREAD_COUNT 8` hard-coded in a shared default (question 4)

Pinning is right. **Verified:** auto-sizing ignores both container CPU limits. With `--cpus 2`,
`THREAD_COUNT` is 8. With `--cpuset-cpus 0-1`, `THREAD_COUNT` is still 8 while `nproc` and the
`OMP_THREAD_COUNT` default are both 2 (`analyst/container_b_probes.txt`).

But `FALKORDB_ARGS` is all-or-nothing. Changing the thread count on another box means retyping all
nine keys, `ASYNC_DELETE yes` included. And the string is pasted as two separate literals (script and
compose), which contradicts the note's "cannot drift".

**Fix:** add a separate `FALKORDB_THREAD_COUNT` that the default string interpolates. Put that
default in one file both entry points read, e.g. an env file used via `--env-file` and
`env_file:`. Production sets the value to the container's CPU allotment. The `MAX_QUEUED_QUERIES`
rule and the memory-cap product both derive from it.

### m2 (minor) — heaviest-write grounding, and writes the note did not measure

The "heaviest write" row measured `Repository.create_document` (`repository.py:1060`). Live ingest
calls `create_document_with_auto_supersede` (`services.py:1320`).

**Verified:** I re-ran the live method at 500k characters (618 chunks), including the supersede of
an identical document, under 64 MiB. It passes (`analyst/writes_probe.txt`). So the conclusion
holds, but the row should cite the live method.

**Also verified:** `_RESET_ALL_PARTICIPANTS_CYPHER` deletes every participant in one write. It
passes under 64 MiB at 1,000 participants (10k messages) and at 5,000 (50k messages). So the cap is
not the binding limit there; time is (see question 2).

The cap is instance-wide on `falkordb-dev`, which also hosts `cpg_*`, `ws:eval` and ad-hoc
`cypher-mcp`/`ccg` reads. None of those were measured. §5 Step 4's "go live early" option should
say so.

### m3 (minor) — clustered seeds make the walk a timeout problem well below 100k depth

The 1.3 s figure in §3.1 and follow-up 1 is for one deep seed. With all 10 seeds in one 100k thread,
the verbatim read takes **12.4 s** (verified). That is past `RAG_QUERY_TIMEOUT_MS` (5 s) at a depth
of roughly 40k, so `hybrid_search` fails on time long before memory. This raises the priority of
follow-up 1 (`Thread {threadId: seed.threadId}`). State it with the clustered figure.

### m4 (minor) — §4.3's `maxmemory` effect is understated

**Verified:** above `maxmemory` with `noeviction`, **every `GRAPH.QUERY` is refused, reads
included** (`OOM command not allowed`). Only `GRAPH.RO_QUERY` and `GRAPH.DELETE` still run
(`analyst/maxmemory_denyoom_probe.txt`).

"70% of the container's memory limit" is also undefined here. `falkordb-dev` has no limit
(`HostConfig.Memory=0`), and it shares the 5.8 GB VM with two `ccg-falkordb` containers. Fold both
points into the split-out item.

### m5 (minor) — `MAX_QUEUED_QUERIES 128` versus the note's own rule for this box

The arithmetic checks out. 8 + 25 = 33 admitted matches the c=32 rows. The 100-connection pool cap
shows in `queue_probe.txt` ("Too many connections" ×28 at 128 clients). But §3.5 says a shared dev
instance with three server processes "may want 256", and §5 applies 128 to exactly that instance.
Either pick 256 for `falkordb-dev`, or say why 128 is enough there. The
`cypher-mcp`/`ccg` clients also count toward Σ pools.

### m6 (minor) — §6 spec list is incomplete

I grepped DESIGN, SERVER and QUERIES for every key name, `maxmemory`, AOF and `GRAPH.CONFIG`,
unfiltered. The hits are DESIGN `:928`, `:930`, `:936-937`, `:942-948` and `:1000-1002`. The note
covers all of them.

Three related sentences become incomplete and are missing from the list:
- **QUERIES.md §18.7** (`:3538-3544`) says the reset's real bound is `FALKORDB_SOCKET_TIMEOUT`. A
  reset can now also fail server-side on the cap, atomically, with a `ResponseError` rather than a
  `TimeoutError`.
- **DESIGN §9**, write-paths table (`:882`): "bound transaction memory" is now enforced.
- **DESIGN §10 Observability:** the `mem consumption exceeded capacity` log signal.

`falkor-chat/README.md:96` and the `start_falkordb.sh` row in `falkor-chat/AGENTS.md` list env
overrides. They are not spec, but Step 1 should name them.

### n1 (nit) — evidence strength and two small misstatements

- The OMP table is one 15 s sample per point. The c=8 and c=16 gaps (~3%) are within unmeasured
  noise. The c=1 result (+21%) and the repeated deep-walk timings (1.23 vs 1.29–1.32 s) carry the
  conclusion, and it holds.
- "`OMP_THREAD_COUNT` default = `THREAD_COUNT`" is wrong: with a cpuset they diverge (m1).
- The capacity report's "1.3 headroom" multiplies per-workspace RAM *inside* `maxmemory`. It is
  not a separate 30% reserve.
- §7 still cites the scratch path. Cite `falkor-chat/docs/reviews/graph-config-defaults-evidence/`.

## What's solid

Each item below was re-checked by me, not inherited from the note.

- **`FALKORDB_ARGS` replaces the image's string; it does not add to it.** My container started
  with only the validated string, and `MODULE LIST` showed exactly that string. The image's
  `MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000` was gone.
- **The validated string keeps every non-default the image sets.** The image sets exactly those
  three keys (`docker inspect` env, `image_run_sh.txt`). `TIMEOUT` and `RESULTSET_SIZE` are
  repeated, and `MAX_QUEUED_QUERIES` is deliberately changed. The image sets no `REDIS_ARGS`, so
  adding them later replaces nothing.
- **`ASYNC_DELETE 1` aborts startup; `yes`/`no` work.** Reproduced: exit 1 with
  `Module … initialization failed … server aborting`. `ASYNC_DELETE yes` reads back as 1,
  `ASYNC_DELETE no` as 0.
- **Suites are compatible with the new string.** Container with the full string:
  `./scripts/test_queries.sh` gave **475/475**. The server pytest suite gave **3696 passed, 1
  failed**. Container with the image's own args: the identical result. The one failure is the same
  test in both, so it predates this change
  (`test_workflow_timers.py::test_sweep_buckets_a_release_that_exhausts_the_budget_as_resumed_not_faulted`,
  on the current dirty tree). Both suites ran against my own containers, so `reference`, `ws:test`
  and `reference-test` on `falkordb-dev` were untouched. The `wf_repo` tests and the
  `reference-test` wipe ran only there.
- **Measurement method.** Bisection at 1 KiB resolution reproduced exactly. My
  single-deep-seed walk (14.88 MiB) matches the note's 14.9 MiB.
- **Write behaviour under the cap.** Breaches are atomic: nothing is committed. Plan-cache literal
  folding is a real effect, and `THREAD_COUNT` must be set at load time.
- **Admission arithmetic** (`THREAD_COUNT` + `MAX_QUEUED_QUERIES`) and the 100-connection client
  pool both match the evidence.
- **Not changing `CACHE_SIZE` or `TIMEOUT_*`** is argued well.

## Open questions

1. **Question 3, AOF and `maxmemory`: split them out.** The trap itself is stated correctly: a first
   `--appendonly yes` start over an RDB-only directory comes up empty (reproduced). The order of the
   steps is also right: enable at runtime, wait for the rewrite, then set the flag and restart. But
   the wait condition is unsafe (B1), the trap becomes permanent after one clean stop (B1), and there
   is no backup step. `maxmemory` needs sizing for a VM that several unlimited containers share, and
   once it is hit, `GRAPH.QUERY` reads are refused too (m4). None of this is in K-098's key list.
   It changes the persistence contract (DESIGN §10) and can lose data permanently. So it should be
   its own backlog item with its own runbook and gate, and **not ship in the same devops step**. §6's
   persistence-bullet edit then moves to that item too.
2. **For the backlog, outside K-098** (found while probing; each is verified):
   - `_RESET_ALL_PARTICIPANTS_CYPHER` grows super-linearly with participants: 5.4 s at 1,000, 136 s
     at 5,000, still running past 190 s at 10,000. QUERIES §18.7's "40× headroom" against the 10 s
     socket timeout runs out at roughly 1,400 participants. Past that, the client raises while the
     server keeps deleting.
   - `scripts/backfill_thread_ids.sh:60` uses a whole-graph form the planner runs as
     label-scan-then-walk-back. That is O(n²): more than 326 s on one 100k-message thread without
     finishing. It is a write, so `TIMEOUT` cannot stop it, and its walk memory never trips the cap.
     Its own header points to the per-thread form; the default should be that form.
   - Whether to raise these items is for the backlog owner to decide.
3. **Who owns the M1 decision?** Either the value goes up, or the expansion is bounded in the query.
   The second is a query change, and K-098's scope says "no change to queries". If graph-dba
   prefers it, the value can ship as an interim (e.g. 128 MiB) with the query fix as its own item.
   The user or the backlog owner should choose.

## Proposed REVIEW.md lines

Sources: B1, M1 and M2 of this review. None of these is enforced by an existing test, CI job or
hook. Each changes what a reviewer does.

### Proposed REVIEW.md lines

Target **root `REVIEW.md`**, `## Always check` (additions):
- A procedure that waits on a status flag before an irreversible step names the state in which the
  flag already reads "done" before the work has run (scheduled, queued, never-run defaults). It
  requires a counter or timestamp that proves the work ran. *(B1)*
- A plan step that restarts a shared stateful service has three parts. It verifies before the stop
  that the state was saved. It records an inventory of every graph or key before the stop and
  compares it after the restart. It names its rollback command. A verify step that checks one
  workspace is a bound, not proof. *(M2)*

Target **`falkor-chat/REVIEW.md`**, `## Always check` (addition):
- A per-query memory or size bound sized from measured reads is checked at the read's full fan-out.
  Per-seed aggregation scales with k (the ANN fan-out) and with each seed's edges, so a measurement
  with one heavy seed among k is a bound, not the worst case. *(M1)*

No prunes proposed.

## Appendix

### A1 — probe inventory (`falkor-chat/docs/reviews/graph-config-defaults-evidence/analyst/`)

| File | What it shows |
|---|---|
| `build.py`, `multi_seed_bisect.py` → `multi_seed.jsonl` | `ws:r1`: 100k-deep thread, last 10 messages near-duplicate embeddings, 50k-hub. `ws:r2`: 10 near-duplicate seeds in short threads, each mentioning a 100k-fan-in hub. Bisection of the verbatim `hybrid_search` and its parts (M1, m3). |
| `container_b_probes.txt` | `ASYNC_DELETE 1` abort; `yes`/`no` load. `THREAD_COUNT` under `--cpus 2` and `--cpuset-cpus 0-1` (m1). |
| `maxmemory_denyoom_probe.txt` | Above `maxmemory`: `GRAPH.QUERY` read and write refused; `RO_QUERY` and `GRAPH.DELETE` allowed (m4). |
| `aof_trap_probe.txt` | Naive `--appendonly yes`, then empty; after one stop, a no-AOF restart is also empty (B1). |
| `aof_scheduled_rewrite_probe.txt` | Rewrite scheduled behind a BGSAVE: the note's two fields already read "done" (B1). |
| `writes2.py`, `writes3.py` → `writes_probe.txt` | Live `create_document_with_auto_supersede` at 500k characters under 64 MiB. `reset_all` scaling. Backfill plan and runtime (m2, question 2). |
| `pytest_candidate.txt`, `pytest_baseline.txt`, `test_queries_candidate.txt` | Suite runs on own containers (*What's solid*). |

`writes.py` is the first attempt at the write probes. It was stopped once its backfill query ran
past 326 s, and that runtime and the plan are what `writes_probe.txt` records. The traceback at
the top of `writes_probe.txt` is my builder's Cypher error, which I fixed before the `reset_all`
runs, not an engine finding.
