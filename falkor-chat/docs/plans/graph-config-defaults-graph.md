# GRAPH.CONFIG defaults for production

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** K-098

## 1. Scope and how it was measured

This note gives each `GRAPH.CONFIG` key that K-098 names (`falkor-chat/docs/BACKLOG.md` `### K-098`)
a value for the first production deployment: the reason, whether it can change at runtime, and the
exact place and syntax for `devops` to set it. A few other keys that are worth deciding are covered
too (§4). `TIMEOUT` is already decided (`falkor-chat/docs/DESIGN.md` §10) and is only carried along
here, because the module-arg string that sets everything else has to repeat it.

This note changes no query or schema. §3.1 specifies one query change as a separate unit. Nothing
here was changed on the shared `falkordb-dev` instance:
- live values were read with `GRAPH.CONFIG GET *`, `MODULE LIST`, `INFO` and `CONFIG GET`;
- live graph shapes were read with `GRAPH.RO_QUERY`;
- every `GRAPH.CONFIG SET`, every runtime-vs-load-time check and every memory measurement ran in a
  throwaway container that this run started and then removed (§3).

Engine: `falkordb/falkordb:v4.18.11` (graph module `41811`, Redis 8.6.3). Host: 8 logical CPUs
(`nproc`), 5.79 GB RAM (`INFO memory` `total_system_memory_human`), WSL2.

## 2. Live values today (`falkordb-dev`, read 2026-11-20)

| Key | Live value | Who set it |
|---|---|---|
| `THREAD_COUNT` | **8** | engine default (= host CPUs; ignores `--cpus` and `--cpuset-cpus`, §3.2) |
| `OMP_THREAD_COUNT` | **8** | engine default (= CPUs the process may run on; follows a cpuset) |
| `CACHE_SIZE` | 25 | engine default |
| `MAX_QUEUED_QUERIES` | 25 | image `FALKORDB_ARGS` |
| `QUERY_MEM_CAPACITY` | 0 (unlimited) | engine default |
| `ASYNC_DELETE` | 1 | engine default |
| `TIMEOUT` / `TIMEOUT_DEFAULT` / `TIMEOUT_MAX` | 1000 / 0 / 0 | image `FALKORDB_ARGS` (`TIMEOUT`); defaults |
| `RESULTSET_SIZE` | 10000 | image `FALKORDB_ARGS` |
| `VKEY_MAX_ENTITY_COUNT` | 100000 | engine default |
| `DELTA_MAX_PENDING_CHANGES` | 10000 | engine default |
| `NODE_CREATION_BUFFER` | 16384 | engine default |
| `EFFECTS_THRESHOLD` | 300 | engine default |
| `CMD_INFO` / `MAX_INFO_QUERIES` | 1 / 1000 | engine default |
| `BOLT_PORT` | 65535 (disabled) | engine default |
| `DELAY_INDEXING` | 0 | engine default |
| `JS_HEAP_SIZE` / `JS_STACK_SIZE` | 268435456 / 1048576 | engine default |
| Redis `maxmemory` / `maxmemory-policy` | 0 / `noeviction` | Redis default |
| Redis `appendonly` / `save` | `no` / `3600 1 300 100 60 10000` | Redis default |

**The backlog's snapshot is stale for the two thread keys.** K-098 and DESIGN §13 record
`THREAD_COUNT 4` / `OMP_THREAD_COUNT 4`; today both read **8**. Neither the scripts nor compose set
them, so the engine sizes them to the CPUs it sees, and the WSL2 VM now exposes 8 instead of 4. The
value follows whatever box the container lands on. That is the argument for pinning them (§3.2).

**Where config is set today.** Nowhere in this repo. `falkor-chat/scripts/start_falkordb.sh` and
`falkor-chat/compose.yaml` pass no module args. The three non-default values come from the image's
own `FALKORDB_ARGS` env var (`MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000`), which the
image's entrypoint (`/var/lib/falkordb/bin/run.sh`) appends to
`redis-server … --loadmodule falkordb.so ${FALKORDB_ARGS}`. Redis server flags go through
`REDIS_ARGS` the same way. `falkor-chat/scripts/start_server.sh` only calls `start_falkordb.sh`.

**Consequence for devops.** Setting `FALKORDB_ARGS` replaces the image's string; it does not add
to it. The new string must repeat `TIMEOUT 1000` and `RESULTSET_SIZE 10000`, or those keys fall
back to the engine defaults.

## 3. Per-key decisions

Runtime settability below was tested with `GRAPH.CONFIG SET` in the throwaway container
(`runtime_settability.txt`). Booleans take `yes`/`no` at runtime. `0`/`1`/`false` are refused
with `Failed to set config value`.

### 3.1 `QUERY_MEM_CAPACITY` — **67108864 (64 MiB)**, was 0 (unlimited)

**Which read is the heaviest.** The GraphRAG reads are `Services.hybrid_search` (two `RO_QUERY`s: the
Message half, `repository.hybrid_search`, at k=10, and the Chunk half, `repository.search_chunks`,
at k=30) and `Services.search_documents` (`search_chunks` with k up to 600: `limit` ≤ 200 ×
`SEARCH_DOCUMENTS_OVERFETCH` 3). The heaviest by far is **`hybrid_search`'s Message half**. It has
two unbounded terms:
- `MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed)` walks from each seed back to its thread's HEAD, so it
  grows with the seed's **depth**.
- `OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related) … collect(DISTINCT
  related)[..5]` collects **every** co-mentioning message per seed before keeping 5. It grows with
  **k × the seed's total entity fan-in**. No `Message-[:MENTIONS]->Entity` edge exists in any live
  graph today, because the edge has a reader and no writer (K-093), so this term is zero on live
  data now.

**Live data cannot exercise this.** The largest live workspace, `ws:demo`, has 12,106 messages, none
embedded, and threads of 15 or fewer (`live_graph_shapes.txt`, `live_vector_shapes.txt`). The
measurement therefore ran on synthetic graphs in throwaway containers.

**Method.** Query texts were copied verbatim from `repository.py`. For each read, a bisection found
the smallest `QUERY_MEM_CAPACITY` at which it still succeeds. The tracked amount is deterministic,
with three repeats identical to the KiB (`measure_parts.jsonl`).

The coverage graph `ws:cov` (`graph-dba-r2/build_cov.py`) varies every axis the read depends on:
- k ∈ {1, 5, 10};
- entity fan-in of 100 / 1k / 10k / 100k;
- 1 mention per seed vs 10 mentions × 10k;
- depth ~20 / 5k / 10k / 100k;
- seeds clustered in one thread vs spread over 10;
- the scoped vs unscoped variant.

Each scenario's 10 seeds are near-duplicate embeddings, so the ANN returns all of them. Results are
in `graph-dba-r2/coverage.jsonl`. The table shows k=10 unscoped; every scenario scales linearly
from k=1 to k=10 in the expansion term, and the scoped variant gives the same numbers.

| Scenario (10 seeds) | Current query: min. cap / latency | Bounded rewrite (below): min. cap / latency |
|---|---|---|
| spread, short threads, fan-in ~10 | 576 KiB / 2 ms | 576 KiB / 2 ms |
| fan-in 10k | 8.1 MiB / 37 ms | 576 KiB / 7 ms |
| fan-in 100k (k=1 / 5 / 10) | 7.4 / 34.6 / **68.7 MiB** / 424 ms | 576 KiB / 17 ms |
| 10 mentions × 10k each | 68.7 MiB / 383 ms | 576 KiB / 24 ms |
| depth 10k, clustered | 2.2 MiB / 174 ms | 576 KiB / 2 ms |
| depth 5k, spread over 10 threads | 1.4 MiB / 42 ms | 576 KiB / 3 ms |
| depth 100k, clustered (k=1 / 5 / 10) | 15.1 MiB / 1.3 / 7.0 / **12.3 s** | 576 KiB / 3 ms |
| depth 100k clustered **and** fan-in 100k | **83.2 MiB / 13.4 s** | 576 KiB / 18 ms (scoped: 24 ms) |

So the current query needs about 370 KiB plus **k × Σ fan-in × ~70 B** plus about 150 B per hop of
the deepest walk; the walk does not add up across seeds. Its latency reaches the 5 s
`RAG_QUERY_TIMEOUT_MS` with clustered seeds between 10k depth (0.17 s) and 100k (12.3 s). The other
reads are small:

| Read | Min. cap | Latency |
|---|---|---|
| `search_chunks` k=30 / k=600 (`search_documents` max) | 418 / 450 KiB | 2 / 7 ms |
| `read_thread`, 10k messages (heaviest non-RAG read; result rows are not counted) | 469 KiB | 267 ms |
| `create_document` at `MAX_DOCUMENT_CHARS` (500k chars, 618 chunks) | 4.5 MiB | 10 ms |

The live ingest path, `create_document_with_auto_supersede`, passes under 64 MiB at the same size,
including the supersede (`analyst/writes_probe.txt`). So does `reset_all` at 5,000 participants:
its limit is time, not memory.

**What the cap does.** It is enforced per query, on reads **and writes**, and the error is
`Query's mem consumption exceeded capacity`. A write that hits it **commits nothing**. That is the
only server-side bound a write has on this build, where `TIMEOUT` cannot stop one (DESIGN §10). A
runaway `collect()` over 30M values was killed in 0.35 s under 64 MiB; uncapped, it took +1.66 GB
(`runaway_check.txt`). The check runs after an allocation, so one big allocation can overshoot it:
that query's `range()` list still raised peak memory by 512 MB. Streaming shapes (a cartesian
product feeding `count(*)`) allocate nothing and are left to `TIMEOUT`.

**Decision: keep 64 MiB, and bound the expansion in the query (separate unit, below).** Re-sizing
the cap was rejected. The current query's memory is k × fan-in with no ceiling: a cap sized for
today's worst case (83 MiB) is overtaken by the next workspace that is larger or more entity-dense.
Its latency also fails the 5 s RAG timeout long before memory matters. The bounded rewrite needs
576 KiB in every scenario above. 64 MiB is then over 14× the heaviest legitimate query measured
(`create_document`, 4.5 MiB) and still stops runaways early.

Worst-case query memory in flight is `THREAD_COUNT` × 64 MiB, which is 512 MiB at 8 threads. It
must fit inside `maxmemory` together with the data, so it is sized with the persistence item
(GAL-I02-K0004).
The capacity report's 1.3 headroom factor multiplies per-workspace RAM inside `maxmemory`; it is not
a separate reserve for this.

**Until the unit lands, 64 MiB is safe on today's data.** The expansion matches nothing until
K-093 builds the writer, and live threads are 15 messages or fewer. Two ways the interim can break:
- **K-093's writer ships first.** Then 64 MiB fails the current query at k × fan-in ≈ 950k: 10 seeds
  on a 95k-fan-in entity. So **the query unit is a prerequisite of K-093's writer.**
- **A thread grows long.** The walk alone hits the cap at a depth of about 430k, but at 10–100k it
  already fails the 5 s timeout. That case needs the unit regardless of K-093.

**The query-change unit** (`tdd-engineer`, reviewed by `graph-dba`; it changes the QUERIES.md §6
canonical query, `repository.hybrid_search` and DESIGN §8 where it describes the read). The new
text, both variants:
```
CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score
MATCH (t:Thread {threadId: seed.threadId})
[MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t)]      -- scoped variant only
CALL { WITH seed
       OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message)
       WITH DISTINCT related LIMIT 5
       RETURN collect(related) AS expanded }
RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role,
       score, [m IN expanded | m.text] AS relatedContext
ORDER BY score ASC, msgId ASC
LIMIT $limit
```
- **Why each part.** The `threadId` anchor plans as `Node By Index Scan | (t:Thread)` in place of
  the walk. The subquery's `DISTINCT … LIMIT 5` stops the traversal early: `GRAPH.PROFILE` shows 50
  records for 10 seeds on a 100k-fan-in entity (`graph-dba-r2/profile_v2_redis.txt`). `msgId ASC`
  makes ties deterministic; the order of tied scores is unspecified today.
- **Equivalence, verified per scenario** (`graph-dba-r2/equivalence.jsonl`). The rewrite returns the
  same seed set, the same scores, and the same `related` list per seed, in the same order. The only
  difference is the order among exactly tied scores. The current query is deterministic against
  itself; it is the plan change that reorders the ties.
- **Precondition.** The rewrite finds a seed's thread by `threadId`, not by reachability. A message
  without `threadId`, or one whose `threadId` disagrees with its chain, would drop out. Live: 0
  missing and 0 mismatched (`graph-dba-r2/live_threadid_check.txt`). `backfill_thread_ids.sh`
  covers older data.
- **Unchanged on purpose.** As today, the seed itself counts among its own `related` messages,
  since it mentions the shared entity (`graph-dba-r2/callsubquery_check.txt`). Whether to exclude it
  belongs to K-093's contract decision.
- **Done-condition.** `test_queries.sh` and the server suite stay green. A fixture test pins
  equivalence on a hub-mentioning thread. `graph-dba` re-runs `graph-dba-r2/coverage.py` against the
  shipped text and confirms the bound.

### 3.2 `THREAD_COUNT` — **`FALKORDB_THREAD_COUNT`, set per deployment** (8 on this box), was 8 by auto-detection

**Measurement.** The host has 8 logical CPUs. Each config was restarted from one RDB snapshot of
`ws:synth` and driven for 15 s per point. The read mix was `hybrid_search`'s Message half,
`search_chunks` k=30, a 50-message `read_thread` and a full-text query, all with
`MAX_QUEUED_QUERIES 25` (`threads_tc*_omp*.txt`):

| `THREAD_COUNT` / `OMP` | c=1 ops/s (hybrid p50) | c=8 | c=16 (hybrid p99) | c=32 |
|---|---|---|---|---|
| 4 / 4 | 135 (23.6 ms) | 554 | 572 (106 ms) | **6,208 rejected** |
| 4 / 1 | 136 (20.6 ms) | 534 | 606 (91 ms) | **7,785 rejected** |
| 8 / 8 (today) | 125 (26.6 ms) | 634 | 661 (114 ms) | 640, 0 errors |
| **8 / 1** | **151 (19.6 ms)** | **655** | **681 (102 ms)** | 650, 0 errors |
| 16 / 1 | 155 (19.2 ms) | 614 | 683 (126 ms) | 686, 0 errors |

Throughput levels off at 8 threads, and 16 gain nothing. The bigger cost of too few threads is
admission: at 4 threads, 32 clients overflow 4 running + 25 queued (§3.5). The Python load generator
shares the CPUs, so the absolute plateau partly reflects the client; the comparison between configs
holds. Each point is one 15 s sample, so the ~3% gaps at c=8 and c=16 are within noise.

**Recommendation: `THREAD_COUNT` = the CPUs the FalkorDB container is allotted, set explicitly per
deployment as `FALKORDB_THREAD_COUNT`** (8 on this box). Auto-detection cannot be trusted for it.
It moved from 4 to 8 between the K-098 snapshot and today without anyone choosing it. It also
ignores container CPU limits: under `--cpus 2` and under `--cpuset-cpus 0-1` it still reads 8. Under
the cpuset, `nproc` and the `OMP_THREAD_COUNT` default both read 2 (`graph-dba-r2/cpu_limit_probe.txt`).
The `MAX_QUEUED_QUERIES` rule (§3.5) and the in-flight memory product (§3.1) both derive from it, so
it is its own variable in the per-deployment file (§5). **Load-time only.**

### 3.3 `OMP_THREAD_COUNT` — **1**, was 8

OpenMP threads parallelize GraphBLAS work inside one query. On this workload of many small
concurrent queries that costs more than it returns. At c=1, OMP 1 is +21% throughput, and hybrid p50
drops from 26.6 to 19.6 ms. The 100k-deep walk runs 1.23 s against 1.29–1.32 s. 8 × 8 also permits
64 compute threads on 8 CPUs. Revisit only if whole-graph `algo.*` analytics arrive. **Load-time
only.**

### 3.4 `CACHE_SIZE` — **25, unchanged (written out explicitly)**

This is the number of cached execution plans per graph. The server issues about 130 distinct query
texts, so a busy workspace's hot set exceeds 25. That costs little (`cache_probe.txt`):
- A miss adds about 0.06 ms (`get_message` 1.11 vs 1.05 ms, `read_thread` 2.21 vs 2.14 ms).
- A parameterized entry is below the memory noise floor.
- **A literal-heavy entry can be huge.** Constant folding keeps a literal's value in the cached
  plan: each distinct text of `UNWIND range(1,5000000)` held +80 MB after finishing, and the
  parameterized form held nothing (`plan_cache_folding.txt`). The app always parameterizes, so the
  exposure is ad-hoc Cypher (`redis-cli`, `cypher-mcp`, ccg `raw_query`), and a larger cache only
  multiplies it.

**Load-time only.**

### 3.5 `MAX_QUEUED_QUERIES` — **Σ client pools − `THREAD_COUNT`; 320 on `falkordb-dev`**, was 25

Admission is `THREAD_COUNT` running plus `MAX_QUEUED_QUERIES` waiting: 8 + 25 = **33 today**.
Anything past that is refused at once with `Max pending queries exceeded`, which falkor-chat
surfaces as a 500. At c=32 with 4 threads, half the queries were refused (§3.2).

One server process can put **up to 100** queries in flight: `FALKORCHAT_THREAD_LIMIT` = 100, and
the falkordb-py pool is redis-py's default of 100 connections (`max_connections or 100`; `db.py`
passes none). With 128 clients on one pool, 28 got `Too many connections`, client-side
(`queue_probe.txt`).

The queue does not bound wait time. Under a 64-client burst of 1.25 s reads with `timeout 5000`, 59
**completed** at wall times up to 28 s and 5 timed out (`queue_probe.txt`). The bound is
falkor-chat's 10 s `FALKORDB_SOCKET_TIMEOUT`, which abandons the call; it does not cancel the query.

**Rule: `MAX_QUEUED_QUERIES` ≥ Σ(client pool sizes on the instance) − `THREAD_COUNT`.** A legitimate
burst is then queued, and it drains in well under a second for typical reads at ~650 ops/s.
- **`falkordb-dev`: 320.** Up to three app processes share it, each with a 100-connection pool:
  `:8000` (chat or the storefront, same port), `:8100` (`load_test.sh`) and `:8200` (agent-team).
  The `cypher-mcp` stdio clients add a few sequential calls: ~303 − 8, rounded up.
- **A production instance with one app process: 128.**

Alert on `GRAPH.INFO`'s `# Waiting queries` depth. **Runtime-settable.**

### 3.6 `ASYNC_DELETE` — **yes, unchanged (written out explicitly)**

On, `GRAPH.DELETE`, and a graph overwritten by `RENAME`, are freed on a background thread. Off,
Redis's main thread frees a ~1.25 GB workspace (DESIGN §11) inline and stalls every tenant for
seconds. There is no reason to turn it off, and no timing was taken.

**The value form is a trap.** As a module arg, `ASYNC_DELETE yes` loads, but **`ASYNC_DELETE 1`
aborts the server start** with a log line that does not name the key. `CMD_INFO 1` fails the same
way (`loadtime_bool_check.txt`). `GRAPH.CONFIG GET` *prints* booleans as `1`/`0`, so a `GET` pasted
into the arg string bricks the start. Write `yes`/`no`. Runtime-settable, with `yes`/`no` only.

## 4. Other keys

### 4.1 Repeated in the arg string, values unchanged

The env-file string replaces the image's own `FALKORDB_ARGS` (§2), so these must be repeated.
- **`TIMEOUT` 1000.** Decided in DESIGN §10; GraphRAG reads pass `RAG_QUERY_TIMEOUT_MS` (5000).
  Runtime-settable.
- **`RESULTSET_SIZE` 10000.** It is the bound for ad-hoc tools; every app read except `read_thread`
  has `LIMIT` ≤ 200. **Known gap:** a thread past 10,000 messages is silently truncated by
  `read_thread` (§6). Runtime-settable.
- **`TIMEOUT_DEFAULT` / `TIMEOUT_MAX` stay 0 / 0, not written.** DESIGN §10 keeps the legacy single
  knob. A runaway write now has a memory bound (§3.1), but still no time bound.

### 4.2 Engine defaults kept, not written

- **`VKEY_MAX_ENTITY_COUNT` 100000** (RDB/replication chunking) and **`EFFECTS_THRESHOLD` 300 µs**
  (write replication by effects): revisit with the first replica. Both runtime-settable.
- **`DELTA_MAX_PENDING_CHANGES` 10000**: ingest batches are already bounded. Runtime-settable.
- **`NODE_CREATION_BUFFER` 16384**: no workload reason to change it. Load-time only.
- **`CMD_INFO` / `MAX_INFO_QUERIES` yes / 1000**: these feed `GRAPH.INFO`, which is §3.5's alert
  signal.
- **`BOLT_PORT`** stays off (RESP only, DESIGN §13) until K-018 decides otherwise.
- **`DELAY_INDEXING`, `JS_*`, folder keys**: unused by this workload.

### 4.3 Redis `maxmemory` and AOF

Out of scope: filed as GAL-I02-K0004 (§8).

## 5. The change: config edits for `devops`, restart procedure for the runbook

The patched script, compose file and env file below were prototyped as scratch copies and
exercised; the repo's own files are untouched. Diffs and checks are in `graph-dba-r2/proto/` and
`graph-dba-r2/proto_arg_checks.txt`.

**Step 1 — new file `falkor-chat/config/falkordb.env`** (per deployment; this content is
`falkordb-dev`'s):
```
# FalkorDB module args for THIS deployment, read by scripts/start_falkordb.sh and (via
# `docker compose --env-file config/falkordb.env`) by compose.yaml. KEY=value, no quotes.
# Booleans are yes/no, never 1/0: a 1 aborts the server start. Sizing: docs/plans/graph-config-defaults-graph.md.
FALKORDB_THREAD_COUNT=8
FALKORDB_MODULE_ARGS=OMP_THREAD_COUNT 1 CACHE_SIZE 25 MAX_QUEUED_QUERIES 320 QUERY_MEM_CAPACITY 67108864 ASYNC_DELETE yes TIMEOUT 1000 RESULTSET_SIZE 10000
```

**Step 2 — `falkor-chat/scripts/start_falkordb.sh`.** After `FALKORDB_CONTAINER_NAME=…`, add:
```bash
FALKORDB_VOLUME="${FALKORDB_VOLUME:-falkordb-data}"
FALKORDB_ENV_FILE="${FALKORDB_ENV_FILE:-$(cd "$(dirname "$0")/.." && pwd)/config/falkordb.env}"

# Module args: a full FALKORDB_ARGS in the environment wins as-is (the rollback path). Otherwise
# compose it from the env file, where an already-set FALKORDB_THREAD_COUNT/FALKORDB_MODULE_ARGS wins.
# The file is parsed, not sourced: its values contain spaces.
if [ -z "${FALKORDB_ARGS:-}" ]; then
  while IFS='=' read -r key value; do
    case "$key" in ''|'#'*) continue ;; esac
    if [ -z "${!key+x}" ]; then printf -v "$key" '%s' "$value"; fi
  done < "$FALKORDB_ENV_FILE"
  FALKORDB_ARGS="THREAD_COUNT ${FALKORDB_THREAD_COUNT:?set in $FALKORDB_ENV_FILE} ${FALKORDB_MODULE_ARGS:?set in $FALKORDB_ENV_FILE}"
fi
```
In **both** `docker run` branches, replace `-v falkordb-data:/var/lib/falkordb/data` with
`-e FALKORDB_ARGS="${FALKORDB_ARGS}" -v "${FALKORDB_VOLUME}":/var/lib/falkordb/data`, and echo the
module args in the banner. Add `FALKORDB_ENV_FILE`, `FALKORDB_THREAD_COUNT`, `FALKORDB_ARGS` and
`FALKORDB_VOLUME` to `usage()` and the header comment, and to the two places that list the
script's overrides: `falkor-chat/README.md` ("Override default ports") and the `start_falkordb.sh`
row in `falkor-chat/AGENTS.md`. `FALKORDB_VOLUME` exists so the edited script can be run against a
scratch directory, never the live volume (Step 4, item 1).

Verified on the prototype: the default composes the full string; `FALKORDB_THREAD_COUNT=4` changes
only that token; a full `FALKORDB_ARGS` is used as-is even with no env file; and a missing env file
with no override fails before `docker run`.

**Step 3 — `falkor-chat/compose.yaml`**, `services.falkordb`:
```yaml
    environment:
      # Per-deployment values live in config/falkordb.env: run `docker compose --env-file config/falkordb.env up`.
      FALKORDB_ARGS: "THREAD_COUNT ${FALKORDB_THREAD_COUNT:?run docker compose --env-file config/falkordb.env} ${FALKORDB_MODULE_ARGS:?run docker compose --env-file config/falkordb.env}"
```
`docker compose --env-file config/falkordb.env config` renders the same string as the script.
Without `--env-file`, every compose command refuses with the message above, `ps` and `down`
included, so compose can never start on the image's defaults by accident. Part of this unit's
done-condition: the `up` and `down` lines in `compose.yaml`'s header (`:3-4`) and in
`falkor-chat/README.md`'s compose block (`:103-104`) carry `--env-file config/falkordb.env`.

**Step 4 — restart procedure** (for the stakeholder-run runbook to quote; the load-time keys need
one restart of `falkordb-dev`). Rehearsed end to end on a throwaway, both rollback branches with a
planted wrong-data start (`graph-dba-r3/rollback_rehearsal.txt`).
1. **Dry run, never on the live volume.** In a terminal (foreground, so the start log is on screen
   even though the container is `--rm`):
   `D=$(mktemp -d) && chmod a+rwx "$D" && FALKORDB_VOLUME="$D" FALKORDB_PORT=16399 FALKORDB_WEB_PORT=13099 FALKORDB_CONTAINER_NAME=falkordb-dryrun ./scripts/start_falkordb.sh`.
   In a second terminal, check that `redis-cli -p 16399 MODULE LIST` shows the args string and
   `redis-cli -p 16399 GRAPH.CONFIG GET '*'` shows each value. Then press Ctrl+C.
2. **Quiesce every client**, or a stray write shows up as a false "differs" in step 7. Stop the
   falkor-chat servers (`:8000`, `:8100`, `:8200`), any test suite or `load_test.sh` run, other
   agents' sessions, and the `cypher-mcp` containers
   (`docker ps --format '{{.Names}} {{.Image}}' | awk '$2 ~ /^cypher-mcp:/ {print $1}' | xargs -r docker stop`).
   `redis-cli CLIENT LIST` must then print one line, its own. Then freeze writes until the stop:
   `redis-cli CLIENT PAUSE 1800000 WRITE`. It holds every `GRAPH.QUERY`, reads included, while the
   `GRAPH.RO_QUERY` inventory, `SAVE` and the shutdown save still run (`graph-dba-r3/pause_probe.txt`).
3. **Inventory before** (read-only):
   ```bash
   inv() { redis-cli GRAPH.LIST | sort | while read -r g; do
     n=$(redis-cli GRAPH.RO_QUERY "$g" "MATCH (n) RETURN count(n)" timeout 120000 | sed -n 2p)
     e=$(redis-cli GRAPH.RO_QUERY "$g" "MATCH ()-[r]->() RETURN count(r)" timeout 120000 | sed -n 2p)
     i=$(redis-cli GRAPH.RO_QUERY "$g" "CALL db.indexes() YIELD label RETURN count(label)" timeout 120000 | sed -n 2p)
     echo "$g nodes=$n edges=$e indexes=$i"; done; }
   inv > inventory-before.txt
   ```
4. **Save, and time it.** `redis-cli LASTSAVE`, then `time redis-cli SAVE` (expect `OK`), then
   `redis-cli LASTSAVE` again; it must have advanced. `SAVE` blocks the instance while it runs, which
   is why step 2 comes first.
5. **Stop, keeping the shutdown log.** `docker logs -f falkordb-dev > stop.log 2>&1 &`, then
   `docker stop -t 300 falkordb-dev`. The grace is generous because `falkordb-dev` is mostly swapped
   out (483 MB used against 54 MB RSS) and Redis writes its RDB again on SIGTERM. A grace that
   expires sends SIGKILL mid-save. Use at least 10× the `SAVE` time from step 4. `stop.log` must end
   with `DB saved on disk` and `Redis is now ready to exit`. If it does not, do not start the new
   container. Then copy the last good state aside, with the pinned image so nothing is pulled:
   `B=$(mktemp -d ~/k098-backup.XXXX) && docker run --rm -v falkordb-data:/d -v "$B":/b --entrypoint cp falkordb/falkordb:v4.18.11 -a /d/dump.rdb /b/`.
   Keep `$B` until step 7 passes.
6. **Start:** `./scripts/start_falkordb.sh -d`, then wait for `redis-cli PING` to answer `PONG`.
7. **Inventory after:** wait until every graph answers 0 to `CALL db.indexes() YIELD status WHERE
   status <> 'OPERATIONAL' RETURN count(status)`, then run `inv >
   inventory-after.txt` and `diff inventory-before.txt inventory-after.txt`, which must print
   nothing and exit 0. Positive control: the same `diff` against `inventory-before.txt` plus one
   appended fake line (`k098_control nodes=1 edges=0 indexes=0`) must exit 1. Then check
   `GRAPH.CONFIG GET '*'` against Step 1's values.
8. **Rollback**, if step 6 fails or step 7 differs. Never `docker stop` here: a SIGTERM writes
   whatever the new instance holds over `dump.rdb`, and an auto-save may already have done so
   (`analyst/pass2/rollback_overwrite_probe.txt`). So exit without saving, then restore the copy:
   ```bash
   redis-cli SHUTDOWN NOSAVE || docker kill falkordb-dev    # SIGKILL if it does not answer
   docker run --rm -v falkordb-data:/d -v "$B":/b --entrypoint cp falkordb/falkordb:v4.18.11 -a /b/dump.rdb /d/dump.rdb
   FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000" ./scripts/start_falkordb.sh -d
   ```
   These are the image's own args. If the edited script itself is at fault, start with the pre-edit
   script, pinned at `2769a99c`, the last commit before the edit (before step 1, check that
   `git log --format=%h -- falkor-chat/scripts/start_falkordb.sh` lists it right after the edit's
   commit; `HEAD` is the edited script once that is committed):
   `git show 2769a99c:falkor-chat/scripts/start_falkordb.sh > /tmp/start_falkordb.prev.sh && bash /tmp/start_falkordb.prev.sh -d`.
   Then re-run step 7's inventory and `diff`, which must exit 0.
9. **Resume.** Restart what step 2 stopped, the way it was started, then run
   `./scripts/verify_workflows.sh` and `./scripts/verify_catalog.sh`. The step-2 pause ends with
   the old process.

**Going live early with the two runtime keys** (`GRAPH.CONFIG SET QUERY_MEM_CAPACITY 67108864`,
`MAX_QUEUED_QUERIES 320`) needs no restart. The cap would then also cover the `cpg_*`, `ws:eval`
and ad-hoc `cypher-mcp`/`ccg` reads on `falkordb-dev`, which none of this note's measurements
covered. The query error names the cause, and the cap can be raised live.

## 6. At close

**Live-spec sentences that change** (a delegated writer updates them at K-098's close, root
`AGENTS.md`):
- **DESIGN §10, tuning bullet.** Today it reads "`GRAPH.CONFIG` — `THREAD_COUNT` (size to cores),
  `QUERY_MEM_CAPACITY`, `MAX_QUEUED_QUERIES`, `TIMEOUT_DEFAULT`/`TIMEOUT_MAX`, `CACHE_SIZE`." It
  becomes the decided values with one-clause reasons:
  - `THREAD_COUNT` = the container's CPU allotment, set per deployment, because auto-sizing ignores
    container limits;
  - `OMP_THREAD_COUNT 1`;
  - `CACHE_SIZE 25`;
  - `MAX_QUEUED_QUERIES` ≥ Σ client pools − threads;
  - `QUERY_MEM_CAPACITY` 64 MiB, which bounds writes too, commits nothing on breach, and is
    runtime-settable;
  - `ASYNC_DELETE yes`.

  It adds that the values live in `config/falkordb.env`, read by `start_falkordb.sh` and by
  `compose.yaml` through `--env-file`, that booleans are `yes`/`no`, and it cites this note for
  the measurements.
- **DESIGN §10, TIMEOUT posture bullet.** "bounded batches … and the existing API input caps are the
  only write-path protection" gains: plus the `QUERY_MEM_CAPACITY` memory bound, which still has no
  time bound.
- **DESIGN §10, Observability bullet.** Add the `mem consumption exceeded capacity` error and
  `GRAPH.INFO`'s waiting-queries depth as signals.
- **DESIGN §9, write-paths table, Backfill/import row.** "bound transaction memory" is now also
  enforced by the engine (`QUERY_MEM_CAPACITY`), on top of batch sizing.
- **QUERIES §18.7.** "The module's `TIMEOUT` argument applies to reads only … the real bound is
  `FALKORDB_SOCKET_TIMEOUT`" gains a server-side failure mode. A reset can also fail atomically on
  the memory cap, raising `ResponseError` rather than `TimeoutError`, and leaving nothing deleted.
- **DESIGN §13, the open-question sentence.** "…and a pre-production review of the live
  `GRAPH.CONFIG` defaults (`THREAD_COUNT 4`, …, `ASYNC_DELETE 1`)" moves to the resolved list as one
  line ("live `GRAPH.CONFIG` values decided, §10"). The stale 4/4 snapshot goes. DESIGN §10's
  persistence bullet moves with GAL-I02-K0004.

**Follow-ups outside K-098** (candidates for the backlog owner):
1. The `hybrid_search` query unit (§3.1). K-093's body records that it blocks K-093's writer; it is
   urgent on its own once any thread passes about 10k messages.
2. `repository.read_thread` has no `LIMIT`, so `RESULTSET_SIZE` silently truncates threads longer
   than 10,000 messages.
3. Under a burst, a queued read can run long after its client's 10 s socket timeout has abandoned
   it (§3.5). Re-measure queue depth on real traffic.
4. From the review's probes: `reset_all` grows super-linearly with participants (5.4 s at 1,000,
   136 s at 5,000). `backfill_thread_ids.sh`'s whole-graph default is O(n²). See
   `falkor-chat/docs/reviews/graph-config-defaults.md`, open question 2.

## 7. Evidence

All evidence is under `falkor-chat/docs/reviews/graph-config-defaults-evidence/`. File names in this
note are relative to it.
- **Round 1, in the top level.** The live read-only probes, which include
  `shared_untouched_check.txt`; the runtime and load-time checks; the `ws:synth`/`ws:deep`
  measurements; the thread-count, plan-cache and queue runs; the arg-string validation; and the
  AOF probes.
- **`graph-dba-r2/`** (this revision):
  - the `ws:cov` coverage probe: `build_cov.py`, `coverage.py` → `coverage.jsonl`;
  - equivalence and plans: `equivalence.jsonl`, `profile_v2_redis.txt`, `scoped_v2_check.txt`,
    `callsubquery_check.txt`;
  - the re-verified CPU-limit and AOF-rewrite facts: `cpu_limit_probe.txt`,
    `aof_scheduled_rewrite_reverify.txt`;
  - the live `threadId` precondition: `live_threadid_check.txt`;
  - the prototype edits and their checks: `proto/`, `proto_arg_checks.txt`;
  - the restart rehearsal: `rehearse_restart.sh` → `restart_rehearsal.txt`, `rh_stop.log`,
    `rh_inventory-*.txt`;
  - `shared_untouched_check_r2.txt`.
- **`graph-dba-r3/`**: the rollback rehearsal (`rehearse_rollback.sh` → `rollback_rehearsal.txt`,
  `r3_inventory-*.txt`), the write-pause and `maxmemory` probes, and `shared_untouched_check_r3.txt`.
- **`analyst/`**: the reviewer's own probes, cited where this note relies on them.

Every container this work started (`k098-gdba-*`) has been removed. Against `falkordb-dev` it ran
only reads.

## 8. Hand-off: persistence item

Filed as `falkor-chat/docs/BACKLOG.md` `### GAL-I02-K0004`.
