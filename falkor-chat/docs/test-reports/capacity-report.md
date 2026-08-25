# Capacity — measured RAM, throughput & hot-read plans

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — · **Version:** 1.0

> **Scope.** Every capacity measurement taken against this deployment: the per-message RAM line at
> 1024 dims, the M1 append-path load test, the hot-read `GRAPH.PROFILE` closeout, and the
> per-workspace RAM budget + shard-packing table. `DESIGN.md` §11 carries only the three numbers
> that shape the design and links here for the rest.
>
> **Moved out of `DESIGN.md` §11/§11.1/§11.2 on 2026-08-24** — measurements are test-report
> content, not design. Citations to `DESIGN.md` §11 still land on the summary that remains there.

> **Re-measure trigger:** engine upgrades (hot-read plans can degrade to a label scan) and the
> first pilot workspace running the **real** embedding model at production volume. Harness:
> `scripts/load_test.sh` → `scripts/load_append.py`.

---

## 1. Per-message RAM line at 1024 dims

Measured live (`falkordb/falkordb:edge`, 4096 realistic messages bulk-loaded into a
1024-dim-indexed scratch workspace: `msgId/text/role/createdAt/threadId` + inline `vecf32`
embedding + `POSTED_BY` + full `NEXT` chain + HEAD/TAIL; `INFO memory` delta):

| Component | Bytes/message |
|---|---|
| raw `vecf32` embedding (1024 × 4 B) | 4,096 |
| node + attrs (text ~50 chars, ids, role, `createdAt`, `threadId`) + edges (`NEXT`, `POSTED_BY`) | ~1,900 |
| HNSW vector index + range-index entries + allocator overhead | ~6,400 |
| **Total observed** | **12,387 ≈ 12.4 KB** |

- Rule of thumb: **~12.5 KB/message at 1024 dims ≈ 1.25 GB per 100k-message workspace**
  (vs ~17–18 KB extrapolated at 1536 — the dim cut saves roughly a third). The bootstrap
  default stays 1536 (chosen per workspace); set `EMBEDDING_DIM=1024` for the decided model
  (`DESIGN.md` §1.3) **before** workspace creation — vector index dimension is fixed at creation.
- `threadId` cost: one short string, ~50–60 B/message, no index — noise (<0.5%) against the
  12.4 KB line. `ReadCursor.lastReadMsgId`: one string per (member, thread) cursor — negligible.
- Ingestion datapoint: ~1,178 msg/s with 256-row `UNWIND` batches incl. embeddings, single
  client — bulk batches of 100–500 rows sit comfortably inside the write-path safety envelope
  (writes are unkillable by TIMEOUT, `DESIGN.md` §10 — keep batches bounded).
- **Measurement caveat (upstream filing recommended):** `GRAPH.MEMORY USAGE` reported
  `indices_sz_mb: 0` while the HNSW index demonstrably held 4096 vectors — on this edge build
  it **under-reports vector-index memory**. Size workspaces from **`INFO memory` deltas**, not
  `GRAPH.MEMORY USAGE`, until fixed upstream.

> **Still open:** re-measure on a pilot workspace with the **real** embedding model before scaling
> out. The chat-core floor and the budget/packing table below (§2–§3) are measured; the embedded
> row is still extrapolated from a synthetic vector load.

## 2. M1 append-path load test + hot-read PROFILE closeout

Measured live on `falkordb/falkordb:edge` through the **M1 REST service path** — 16 concurrent
posters, 3,000 messages, one channel / 16 threads, each `POST /threads/{id}/messages` a full
`services.post_message` round trip (actor + mention validation, role derivation, `QUERIES.md` §4 v2
guarded write). This is the **live request path**, not the bulk-`UNWIND` ingestion datapoint
(§1, ~1,178 msg/s single-client batched). Harness: `scripts/load_test.sh` →
`scripts/load_append.py`.

| Metric | Value |
|---|---|
| Sustained append throughput | **~614 msg/s** (16 clients, single graph) |
| Append latency p50 / p90 / p99 | **24.4 / 30.6 / 40.7 ms** (max 146 ms) |
| Errors | 0 / 3,000 |

Throughput is **graph-write-bound** (FalkorDB serialises writes per graph key), so the
per-thread fan-out only removes first-post/TAIL race dispatch from the latency sample — a single
busy channel lands the same ceiling. Each post is ~4 round trips (`thread_exists` +
`resolve_member_kinds` + `thread_has_head` + the write), so this is a conservative service-layer
figure, not raw Cypher throughput.

**Hot-read plans — all four hit an index-backed anchor; none degraded to a `NodeByLabelScan`**
(`GRAPH.PROFILE` on the loaded `ws:load` graph, raw plans archived by the harness). Re-profile on
engine upgrades:

| Hot read | Anchor op | Verdict |
|---|---|---|
| `QUERIES.md` §4 thread read | `Node By Index Scan \| (t:Thread)` → HEAD/NEXT walk | index ✓ |
| `QUERIES.md` §9.1 since-read (thread) | `Node By Index Scan \| (t:Thread)`; keyset predicate folds into a `Filter` on the walk | index ✓ |
| `QUERIES.md` §9.2 since-read (ws-wide) | `Node By Index Scan \| (m:Message)` on `createdAt`; composite `OR` folds into the scan, **no residual Filter** | index ✓ |
| `QUERIES.md` §5 full-text search | `ProcedureCall` (`db.idx.fulltext.queryNodes`, RediSearch full-text index) | index ✓ |

Confirms the `falkor-chat/AGENTS.md` standing note (Formulation-A composite keyset still plans as a bare
`Node By Index Scan` with no residual Filter on this build) and the `DESIGN.md` §9 plan claim.

## 3. Per-workspace RAM budget & shard packing (`INFO memory` deltas)

**Chat-core floor (M1, no embeddings) — measured `INFO memory` `used_memory` delta:** 3,000
messages added **3,173,056 B → ~1.06 KB/message** (node + `text`/ids/`role`/`createdAt`/`threadId`
attrs + `NEXT`/`POSTED_BY` edges + `createdAt` range index + `msgId` constraint index + full-text
index entry) → **~101 MB per 100k-message workspace**. That sits *below* the ~1.9 KB
node-line estimate (§1), confirming that at 1024 dims the embedding (4 KB) + HNSW/range overhead
(~6.4 KB) dominate — **~85% of the 12.4 KB/message total is vector, not chat.**

**Per-workspace RAM budget line (per 100k messages):**

| Profile | Per message | Per 100k-msg workspace |
|---|---|---|
| M1 chat-core (no embeddings) — *measured* | ~1.06 KB | **~101 MB** |
| M2 with 1024-dim embeddings (§1) | ~12.4 KB | **~1.25 GB** |
| M2 with 1536-dim embeddings (§1) | ~17–18 KB | **~1.7 GB** |

**Shard:workspace packing ratio** = (shard `maxmemory`) ÷ (per-workspace RAM × 1.3 headroom for
writes / RDB fork / index build; no eviction of graph keys, `DESIGN.md` §10). Worked example on a 32 GB shard
with `maxmemory` ≈ 22 GB:

| Workspace profile (100k msgs) | Fits per 22 GB shard |
|---|---|
| chat-core only (~101 MB) | **~170 workspaces** |
| 1024-dim embedded (~1.25 GB) | **~13 workspaces** |
| 1536-dim embedded (~1.7 GB) | **~10 workspaces** |

Size real deployments from the **embedded** row (M2 is the target); the chat-core floor is the
M1 reality and the lower bound. `GRAPH.MEMORY USAGE` still reported all-zero
`indices_sz_mb`/`total_graph_sz_mb` for the loaded `ws:load` graph (the §1 caveat holds even
with **no** vectors present) — budget from `INFO memory` deltas, never `GRAPH.MEMORY USAGE`.

---

