# Connected code graph — P1 `raw_query` memory controls

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** C-1001 (P1-38)

Read-only look for the P1-38 exit report (`docs/plans/connected-code-graph-p1-gates.md`, "`raw_query` memory limit"). Configured state is read from `cpg/infra/ccg-falkordb.sh`, `cpg/ccg/store/acl_users.txt` and `cpg/clients/common.py`. Behavior was observed on my own throwaway of `falkordb/falkordb:v4.18.11` (Redis 8.6.3, graph module 41811, port 16410, unauthenticated, 768 MiB container cap, removed afterwards). The running `ccg-falkordb` was not contacted, so its live values are what `ccg-falkordb.sh up` sets, not what I read from it (`status` verifies them).

## Controls that exist and are on (per the script)

| Control | Value | What it bounds |
|---|---|---|
| Redis `maxmemory` / policy | 1536mb / `noeviction` | Refuses **write** commands (`denyoom`) when `used_memory` is above it. Does not stop a read-only query (below). |
| Container cap | `--memory 2g --memory-swap 2g` | Kernel OOM-kills the container, not the VM. Because nothing persists, the kill loses the snapshot. |
| `TIMEOUT_MAX` | 10000 ms (image default is 0, unlimited) | Wall time per query. The client sends its limit on every call; the server refuses a larger one. |
| Client socket timeout | limit + `SOCKET_MARGIN_S` | Client side only. |
| `RESULTSET_SIZE` | 10000 (image default) | Rows returned; a reply of exactly that size is a typed `ResultTooLarge`. It bounds the reply, not the working set. |
| Raw role ACL | `+graph.ro_query +graph.explain +info`, keys `~ccg_<s>` `~ccg_<s>__ov_*` | No writes, no `GRAPH.QUERY`/`PROFILE`, no `GRAPH.CONFIG`, no other keys. |

## Controls that exist but are off

- **`QUERY_MEM_CAPACITY`** (per-query memory cap, bytes) is **on since CO-8** (`ccg-falkordb.sh up` sets 268435456; `status` flags any other value): see "Blast radius" below. Image default 0 (off). Module-wide, not per user; not settable by the raw role (no `graph.config`). Observed on the first note's throwaway: with 20 MiB set, a `collect` over 3,000,000 strings returned `Query's mem consumption exceeded capacity`, process peak 51 MiB; with 0 the same query succeeded at a peak of 293 MiB.
- **`MAX_QUEUED_QUERIES`** 25 (image default): bounds the queue, not memory.

## Controls that do not exist

- **`MEM_LIMIT`**: `GRAPH.CONFIG SET MEM_LIMIT` answered `Unknown configuration field`, and `GRAPH.CONFIG GET *` lists no such field in this build.
- **A per-user or per-role memory limit**: none (ACL has no memory dimension; `maxmemory-clients` bounds client buffers only, and is 0).
- **A `maxmemory` guard on `GRAPH.RO_QUERY`**: `COMMAND INFO` shows `graph.RO_QUERY` flags `readonly`, no `denyoom`, whereas `graph.QUERY` has `denyoom`. With `maxmemory` below `used_memory`, `RO_QUERY` still ran (returned 1) and `GRAPH.QUERY` was refused with OOM. The read-only `raw_query` path is therefore outside the `maxmemory` bound entirely.

## Blast radius of `QUERY_MEM_CAPACITY` 256 MiB (CO-8)

Measured on my own throwaways (`falkordb/falkordb:v4.18.11`, ports 16480 and 16481, the script's `up`, 2g container cap, removed afterwards) over the proving system built from committed HEAD `73613471` (Symbol 18,395, CallSite 99,551, ConfigEntry 347,733; `used_memory` 393.56 MiB). `ccg-falkordb`, `ccg-falkordb-tw-16390` and `falkordb-dev` were not contacted.

**Does not break the engine or the build.** The full build (extract, assemble, load, swap) completed under the 256 MiB setting twice (2014 s and 1908 s). 180 engine asks (impact and depends-on over 25 highest-in-degree symbols, 40 random symbols and 25 random contract points) all succeeded at 256 MiB; at 16 MiB one failed (`name_out`) and at 4 MiB 24 failed (`name_out` 22, `name_in` 2). Heaviest engine read, measured by raising the cap through powers of two until the query ran: `name_in` over a 1,000-id chunk (`config.CHUNK`) of the highest-degree symbols succeeds at 128 MiB and fails at 64 MiB (10,000 rows, 1.2 s); every other read succeeds at 32 MiB or less (most at 1 to 4 MiB). Heaviest loader statement, on 10,000 rows (the most a read returned; the loader's batch is up to 20,000, not run at that size): node batches 32 MiB (`Symbol`, `ConfigEntry`), edge batches 16 MiB. The headroom is a factor of 2 on `name_in` at this store's size; its memory grows with the number of call sites that resolve by name to the chunk's symbols, so a larger system needs the figure re-measured. A `name_in` over the cap fails as a `StoreError` (the reader halves its chunk only when the reply fills `RESULTSET_SIZE`).

**Bounds.** A single query that allocates through the accounted operators is refused with `Query's mem consumption exceeded capacity`, and the instance stays up: `collect(toString(x))` over 3,000,000 and 20,000,000 rows, a sorted `UNWIND range(1,20000000)`, and (light 8,000-node and 100,000-node synthetic stores) a collected cross product, each refused in 0.2 to 8.6 s with the container not OOM-killed. Eight consecutive refused queries on one instance left the container between 189 and 308 MiB (start 58 MiB): a refusal still costs roughly the cap plus overhead in process memory while it runs.

**Does not bound.**
- **A cross product whose rows are aggregated, over nodes with realistic property weight.** On a 100,000-node store of 8-property nodes, `MATCH (a:C),(b:C) WITH a,b LIMIT 30000000 RETURN size(collect([a.callSiteId,b.callSiteId]))` was **not refused and OOM-killed the container** in 8.4 s (`OOMKilled=true`) with the cap at 256 MiB, 64 MiB and 16 MiB alike; the same shape on the real store (the CallSite label, via `Client.raw_query`) did the same (process at 1.82 GiB when sampled), twice, so the snapshot (graphs, ACL users, runtime config) is lost and a rebuild restores it. `MATCH (a:C),(b:C) RETURN size(collect(a.callSiteId))` was not refused either: stopped by the 10 s time limit with the container at 540 MiB (cap 256 MiB), 524 MiB (16 MiB) and 470 MiB (64 MiB). So the cap is not a hard limit on memory: some operator results are not counted against it, and no value tried closed that. The 10 s time limit and the 2g container cap remain the only bounds for such a shape.
- Several concurrent queries: each is bounded on its own, if at all. Three concurrent cross-product-aggregate queries with a 3 s time limit each OOM-killed a throwaway (reproduced on port 16530, `OOMKilled=true`), so a shorter time limit does not remove the exposure.
- Everything outside query execution (`maxmemory` and the container cap still govern the rest); a write path (`GRAPH.QUERY`) is the same setting but unreachable by the raw role.

## What the P1 report may say

A `raw_query` is a read-only query with a 10 s time limit, a 10,000-row reply cap and, from CO-8, a 256 MiB per-query memory cap on the operators that count against it (an instance started before CO-8, or restarted behind `up`'s back, has no cap: `status` reports it). The cap refuses the large-`collect`/sort shapes measured above. It does **not** bound a cross product feeding an aggregate: that shape was measured taking the instance to its 2 GiB container cap and being OOM-killed with the cap set, and a kill drops the whole no-persistence snapshot (graphs, ACL users, runtime config), which a rebuild plus `create_users` restores. Per-user limits are P4 (FR-24); a real bound on that shape needs a different control (not designed here).
