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

- **`QUERY_MEM_CAPACITY`** (per-query memory cap, bytes): image default 0 (off); `ccg-falkordb.sh` never sets it (no match under `cpg/infra`, `cpg/ccg`, `cpg/clients`). Observed enforced: with 20 MiB set, a `collect` over 3,000,000 strings returned `Query's mem consumption exceeded capacity`, process peak 51 MiB. With 0, the same query succeeded and the peak was 293 MiB. It is a module-wide cap, not per user. Not settable by the raw role (no `graph.config`).
- **`MAX_QUEUED_QUERIES`** 25 (image default): bounds the queue, not memory.

## Controls that do not exist

- **`MEM_LIMIT`**: `GRAPH.CONFIG SET MEM_LIMIT` answered `Unknown configuration field`, and `GRAPH.CONFIG GET *` lists no such field in this build.
- **A per-user or per-role memory limit**: none (ACL has no memory dimension; `maxmemory-clients` bounds client buffers only, and is 0).
- **A `maxmemory` guard on `GRAPH.RO_QUERY`**: `COMMAND INFO` shows `graph.RO_QUERY` flags `readonly`, no `denyoom`, whereas `graph.QUERY` has `denyoom`. With `maxmemory` below `used_memory`, `RO_QUERY` still ran (returned 1) and `GRAPH.QUERY` was refused with OOM. The read-only `raw_query` path is therefore outside the `maxmemory` bound entirely.

## What the P1 report may say

A `raw_query` is a read-only query with a 10 s time limit and a 10,000-row reply cap, but no memory cap. A single query that builds a large intermediate (`collect`, a cross product, a wide variable-length expansion) can allocate until the container's 2 GiB cap is hit, which OOM-kills the instance and drops the whole no-persistence snapshot (graphs, ACL users, runtime config); `maxmemory` does not prevent this because read-only graph queries are not gated by it. A rebuild restores it. Per-user limits are P4 (FR-24). I did not test an unbounded query and do not claim a specific allocation rate; the limit is the time limit times whatever the query allocates per second.

## Recommendation (not applied)

Add `QUERY_MEM_CAPACITY` to the P1 module config in `ccg-falkordb.sh up` (and to `status`' drift check), for example `GRAPH.CONFIG SET QUERY_MEM_CAPACITY 268435456` (256 MiB). It is module-wide, so first confirm that the engine's own heaviest queries and the build's writes stay under it. Peak query memory and process peak differ (20 MiB cap, 51 MiB process peak), so 256 MiB leaves headroom under `maxmemory`. If adopted, the report's limit statement narrows from "can exhaust instance memory" to "one query is capped at 256 MiB, several concurrent ones are not".
