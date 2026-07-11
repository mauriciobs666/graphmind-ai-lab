# Plan — Documentation-architecture consolidation: one authoritative decision register

> Status: 🔵 proposed · Author: architect · Date: 2026-07-05
> Scope: **doc-only**. No code, no schema, no query behavior. This plan relocates *design
> decisions* into a single authoritative home; a separate `coder` pass executes it.
> Companion item **K-019** owns stale test counts (110 / 126), the embedding-model/dim §13
> "still open" wording, and §12/§14.1 scope wording — **out of scope here**; one-line overlaps
> are flagged where they touch.
> Findings re-verified against the live docs on 2026-07-05.

---

## 1. Goal & the target layering model

**Goal.** Design decisions have accreted *outside* the canonical design doc. `DESIGN.md` §1
"Decisions locked in" holds only **3** rows (the top-level axes); `AGENTS.md`'s "Decisions locked
in" table holds **~18** detailed decisions *with rationale*; `docs/BACKLOG.md` holds a 6-row
"Locked M2 stack decisions" block. So `AGENTS.md` has quietly become the de-facto authoritative
home for ~15 design decisions the DESIGN register never absorbed. This violates the project's own
stated discipline — already applied to queries ("the exact verified Cypher lives in ONE place —
QUERIES §4; do not copy — link, the duplication is what lets copies drift") but never to
decisions. This plan applies that single-authoritative-home rule to decisions, consistently.

**The target layering model (user-approved).**

| Doc | Role | Contains decisions? |
|---|---|---|
| `docs/DESIGN.md` | **Current state** — the blueprint | **Yes — §1 is the single authoritative decision register.** It absorbs the detailed decisions from AGENTS.md + plan.md, each with its rationale/consequence. Where the body already explains a decision (thread model §5, write paths §5.3/§9, indexes §7, engine facts §2, workflow §6), §1 carries the *statement* and points to that section — it does **not** re-explain. |
| `docs/HISTORY.md` | **The diff** — dated changelog | No. Provenance only (when/why a thing was decided). Stays a log. Every durable decision it records must be reflected in DESIGN; history entries stay intact as dated records. |
| `docs/BACKLOG.md` | **The future** — forward-looking backlog | No decision *store*. User-approved locked decisions **graduate** into DESIGN; plan.md keeps a pointer. |
| `AGENTS.md` | **Pointer map** for agents | A terse **do-not-reopen index** — one line per locked decision (name + link to its DESIGN home), **rationale removed** (rationale lives once, in DESIGN). Not deleted — agents get the quick locked-list without reading 667 lines of DESIGN. |

**Mental model to preserve:** **DESIGN = state · history = the diff · plan = the future ·
AGENTS = pointer map.** A decision (state) lives once in DESIGN; a decision *event* stays in
history; neither AGENTS.md nor plan.md is a parallel decision store.

**The single-authoritative-home principle.** Each design decision has exactly ONE home (DESIGN §1
register, cross-linked to at most one body section that details it). Every other mention is a
one-line pointer. This plan must not *create* duplication while removing it — see the §1.2 rule
in Risks (§7): a §1 register row is a *statement + link*, never a second full copy of the body
prose.

---

## 2. Decision-register migration table (core deliverable)

Every AGENTS.md "Decisions locked in" row + every plan.md "Locked M2 stack" row, mapped to its
target DESIGN home and the pointer the source keeps. "DESIGN body home" = a body section that
already explains it (register row = statement + link); "NEW" = no body home, the §1 register row
carries the decision itself (or points to QUERIES where the mechanics already live canonically).

### 2.1 From `AGENTS.md` "Decisions locked in" table (18 rows)

| # | Decision | Currently authoritative in | Target DESIGN home | AGENTS.md keeps (pointer) |
|---|---|---|---|---|
| D1 | **Single store** — FalkorDB for all domain data, no secondary store | AGENTS row + DESIGN Philosophy blockquote | Philosophy header + **§1.2 row** → §2 | `Single store (FalkorDB, no 2nd store) → DESIGN §1.2` |
| D2 | **One graph per workspace** (`ws:{id}`) | **Already DESIGN §1** (Tenancy axis) + §3 | §1 (existing axis) — no new row needed | `Per-workspace graph → DESIGN §1/§3` |
| D3 | **Thread-scoped `NEXT`** linked list | AGENTS row; body §5.2 | §5.2 (body) + **§1.2 row** | `Thread-scoped NEXT → DESIGN §1.2 (§5.2)` |
| D4 | **No DayBucket** (rejected alternative) | AGENTS row; DESIGN status-line only ("DayBucket removed") | **§1.2 row — NEW** (no body prose exists) | `No DayBucket → DESIGN §1.2` |
| D5 | **`Thread` owns `HEAD`+`TAIL`** pointers | AGENTS row; body §5.2 | §5.2 (body) + **§1.2 row** | `Thread owns HEAD/TAIL → DESIGN §1.2 (§5.2)` |
| D6 | **`Message.role` inline property** | AGENTS row; body §5.1/§5.2 | §5.1/§5.2 (body) + **§1.2 row** (merge with D16) | `Message.role inline+derived → DESIGN §1.2 (§5.1)` |
| D7 | **`coalesce(userId, agentId)` member identity** | AGENTS row; QUERIES §2 (no DESIGN body) | **§1.2 row → QUERIES §2** (no DESIGN body added) | `coalesce member identity → DESIGN §1.2 (QUERIES §2)` |
| D8 | **Vector indexes via DDL**, not a procedure | AGENTS row; body §2/§7.1 | §2/§7.1 (body) + **§1.2 row** | `Vector index via DDL → DESIGN §1.2 (§7.1)` |
| D9 | **Index before constraint, always** | AGENTS row; body §2/§7.1 | §2/§7.1 (body) + **§1.2 row** | `Index-before-constraint → DESIGN §1.2 (§7.1)` |
| D10 | **`Message.embedding` inline `vecf32`** | AGENTS row; body §5.1/§5.2/§7.3 | §5/§7.3 (body) + **§1.2 row** | `embedding inline vecf32 → DESIGN §1.2 (§5.2)` |
| D11 | **Vector score = cosine distance** (ASC) | AGENTS row; body §7.1/§8 | §7.1/§8 (body) + **§1.2 row** | `cosine-distance ASC → DESIGN §1.2 (§8)` |
| D12 | **`status` as property, not label** | AGENTS row; body §6.2 | §6.2 (body) + **§1.2 row** | `status as property → DESIGN §1.2 (§6.2)` |
| D13 | **`ctx`/`input`/`output` flat/serialised** | AGENTS row + rule 8 (**no DESIGN body**) | **§1.2 row — NEW** + one **NEW body line in §6.2** | `flat ctx/input/output → DESIGN §1.2 (§6.2)` |
| D14 | **`Message.threadId` denorm, unindexed** | AGENTS row; body §5.1 (K-007 bullet) | §5.1 (body) + **§1.2 row** | `threadId denorm/unindexed → DESIGN §1.2 (§5.1)` |
| D15 | **Guarded-CREATE write paths + status row** | AGENTS row; body §5.3/§9 | §5.3/§9 (body) + **§1.2 row** | `guarded-CREATE writes + status row → DESIGN §1.2 (§5.3/§9)` |
| D16 | **`Message.role` values `user`/`assistant`, derived** | AGENTS row; body §5.1 | §5.1 (body) — **merged into D6's §1.2 row** | (folded into D6 pointer) |
| D17 | **Composite `(createdAt, msgId)` keyset cursor** | AGENTS row; QUERIES §9.1/§9.3 (DESIGN §12 mention only) | **§1.2 row → QUERIES §9** (no dedicated DESIGN body; §12 keeps its mention) | `composite keyset cursor → DESIGN §1.2 (QUERIES §9)` |
| D18 | **Member ids namespace-unique across `User`/`Agent`** | AGENTS row; QUERIES §2/§7 | **§1.2 row → QUERIES §2/§7** (+ optional 1-line §5.1 invariant, see D2-of-old-draft) | `member-id namespace-unique → DESIGN §1.2 (QUERIES §2/§7)` |

**Tally:** 18 AGENTS rows → **1 already in DESIGN §1** (D2), **16 become §1.2 register rows**
(D6+D16 merge to one), of which **13 already have a DESIGN body home** (statement + link only) and
**5 are net-new content** — D4 (DayBucket, a rejected alternative with no prose), D13 (flat
`ctx`, needs one new §6.2 body line), and D7/D17/D18 (mechanics already canonical in QUERIES, so
the §1.2 row points there rather than adding DESIGN body).

### 2.2 From `docs/BACKLOG.md` "Locked M2 stack decisions" (6 rows, lines 56-69)

All six are **user-approved (2026-07-04)** and durable → graduate into a new DESIGN **§1.3 "M2
stack — decided, pending implementation"** block, cross-linked to the RAM/roadmap sections that
already carry the numbers. plan.md's block is then replaced by a one-line pointer to §1.3.

| Decision | Currently in | Target DESIGN home | plan.md keeps |
|---|---|---|---|
| Embedding model **Qwen3-Embedding-0.6B** (GGUF Q8_0) | plan.md §"Locked M2 stack" | **§1.3 row** → §11 (RAM), §12 (M2) · *K-019 overlap: it removes the §13 "open" line* | pointer → DESIGN §1.3 |
| **`EMBEDDING_DIM=1024`** (MRL 512/256 later) | plan.md | **§1.3 row** → §11 (12.5 KB/msg line) · *K-019 overlap: §13 wording* | pointer → DESIGN §1.3 |
| Agent LLM **Qwen3-4B-Instruct-2507** Q4_K_M | plan.md | **§1.3 row** → §12 (M2 roadmap) | pointer → DESIGN §1.3 |
| Runtime **LM Studio** (Windows host, WSL2 mirrored net) | plan.md | **§1.3 row** → §10/§12 | pointer → DESIGN §1.3 |
| **VRAM budget 6 GB** (RTX 4050, co-resident) | plan.md | **§1.3 row** → §11 | pointer → DESIGN §1.3 |
| Upgrade path **`qwen3-embedding:4b`** (same 1024-dim MRL, re-embed only) | plan.md | **§1.3 row** → §12 | pointer → DESIGN §1.3 |

> **K-019 overlap (one line, as required):** K-019 owns rewriting DESIGN §13's "Embedding model &
> dimension — open" question and the §11 "embedding model still open" phrase. This plan owns the
> *positive relocation* (the locked block → §1.3). To avoid two hands on §13: **either sequence
> this plan's §1.3 add before K-019's §13 rewrite, or fold both into one K-019+consolidation pass**
> (plan.md already suggests K-019 folds into the K-008 gate). Coordinate; do not double-edit §13.

---

## 3. DESIGN.md §1 additions (verbatim-ready for the coder)

Restructure §1 into three subsections. **§1.1 is the existing 3-row axes table, unchanged.** Add
**§1.2** (detailed register) and **§1.3** (M2 stack). Register rows are *statement + rationale +
where-detailed*, never a re-copy of body prose.

### §1.2 — Locked design decisions (detailed register)

> Each row is the authoritative statement of the decision; the "Detailed in" column links to the
> body section that explains the mechanics (or to QUERIES.md where the mechanics are canonical).
> Do not re-explain the mechanics here.

| Decision | Rationale / consequence | Detailed in |
|---|---|---|
| **Single store** — FalkorDB holds all domain data; no secondary store | Project philosophy: one engine, one query language, one ops model | Philosophy header, §2 |
| **Thread-scoped `NEXT` linked list** | Users read threads, not channel feeds; O(1) append; Thread stays sparse | §5.2 |
| **No DayBucket** *(rejected alternative)* | Designed for channel-wide ordering; dropped when the thread-scoped model was chosen | §5.2 |
| **`Thread` owns `HEAD` + `TAIL` pointers** | Thread stays sparse — exactly 2 edges regardless of message count | §5.2 |
| **`Message.role` inline property; values `user`/`assistant` derived server-side** from the author label (`User→user`, `Agent→assistant`), never trusted from the caller | Filter by role without traversing `POSTED_BY`; agents author first-class (K-007) | §5.1, §5.2 |
| **`coalesce(u.userId, a.agentId)` for member identity** (two indexed `OPTIONAL MATCH` + `coalesce`) | `User` has `userId`, `Agent` has `agentId` — both are members; anchored lookup avoids the `OR`-scan | QUERIES §2 |
| **Vector indexes via DDL**, not a procedure | `db.idx.vector.createNodeIndex` is not registered on this build | §2, §7.1 |
| **Index before constraint, always** | `GRAPH.CONSTRAINT CREATE` requires a pre-existing range index | §2, §7.1 |
| **`Message.embedding` inline as `vecf32`** | Single-query vector + traversal hybrid retrieval | §5.2, §7.3 |
| **Vector score is cosine *distance*** (0 = identical) → `ORDER BY score ASC` | Most-similar-first ranking | §7.1, §8 |
| **`status` as a property, not a label** | Avoids re-labeling churn on state changes; index it for "all running" reads | §6.2 |
| **`ctx` / `input` / `output` are flat/serialised strings** | FalkorDB stores scalars + scalar lists only — no nested maps; never query inside them | §6.2 |
| **`Message.threadId` denormalized inline, unindexed** | Nav metadata for §9.2/§5 rows; HEAD/NEXT walk stays canonical; unindexed saves RAM/write cost (K-007) | §5.1 |
| **Guarded-CREATE write paths** (`FOREACH`+`CASE` per path) with an always-returned status row; **no MERGE on `Message`** | Retry replay is a no-op (`dupMsg`); first-post race refused (`hadHead`); uniqueness constraint is the backstop (K-007) | §5.3, §9 |
| **Composite `(createdAt, msgId)` keyset cursor** (`ReadCursor.lastReadAt`/`lastReadMsgId`) | Timestamp alone is not a total order — same-ms ties skipped rows; cursor reads are lossless (K-007) | QUERIES §9.1/§9.3 |
| **Member ids are namespace-unique across `User`/`Agent`**; `ensure_user`/`ensure_agent` are v2 guarded-CREATE queries returning `(created, existed, collided)`; cross-label collision refuses (`MemberIdCollisionError`) | A shadow node with the other label's id eclipses it in every `coalesce` lookup (K-010) | QUERIES §2/§7 |

**One new body line (D13)** — add to DESIGN §6.2, right after the `StepRun` node listing:

> `ctx` (on `WorkflowRun`) and `input`/`output` (on `StepRun`) are **flat, serialised strings**,
> not nested maps — FalkorDB stores only scalars and scalar lists. Queries never filter *inside*
> them (see §1.2).

### §1.3 — M2 stack (decided 2026-07-04, pending implementation)

> User-approved M2 stack. Locked here; implemented in K-008/K-013 (see docs/BACKLOG.md). Numbers
> detailed in §11 (RAM) and §12 (M2 roadmap).

| Component | Decision | Rationale | Detailed in |
|---|---|---|---|
| Embedding model | **Qwen3-Embedding-0.6B** (GGUF, Q8_0) | Best small-model MTEB quality; 100+ languages (PT-BR + EN); ~0.6 GB resident | §11, §12 |
| Vector dimension | **`EMBEDDING_DIM=1024`** (MRL 512/256 later) | Native dim; ~12.5 KB/message with HNSW — the §11 RAM line | §11 |
| Agent LLM | **Qwen3-4B-Instruct-2507** Q4_K_M (non-thinking) | RAG answering, not CoT; low latency; `-Thinking-2507` a drop-in for M3 | §12 |
| Runtime | **LM Studio** on the Windows host (OpenAI-compatible), reached from WSL2 (mirrored networking → localhost) | Reuses the severino path; zero new moving parts; Ollama fallback | §10, §12 |
| VRAM budget | **6 GB dedicated** (RTX 4050) — embedder + 4B LLM co-resident | Do not plan around shared-RAM spill | §11 |
| Upgrade path | **`qwen3-embedding:4b`** — same family, same 1024-dim MRL | Re-embed only; no schema change | §12 |

---

## 4. AGENTS.md rewrite spec — table becomes a terse pointer index

The `## Decisions locked in — do not reopen without strong cause` section stays (user decision:
keep the quick locked-list), but its **rationale column is removed** and each row becomes a
one-line pointer to the DESIGN home. Replace the two-column *Decision | Rationale* table with a
two-column *Decision | Home* index. Intended new format (3 example rows):

```markdown
## Decisions locked in — do not reopen without strong cause

> Rationale lives once, in `docs/DESIGN.md` §1 (the authoritative register). This is the quick
> do-not-reopen index — follow the link for the *why*.

| Decision | Home |
|---|---|
| FalkorDB is the single store (no secondary store) | DESIGN §1.2 → §2 |
| One graph per workspace (`ws:{id}`) | DESIGN §1.1 (Tenancy) / §3 |
| Thread-scoped `NEXT` linked list | DESIGN §1.2 → §5.2 |
| `Thread` owns `HEAD`+`TAIL` pointers | DESIGN §1.2 → §5.2 |
| … (one line per decision, in the §2.1 mapping's "AGENTS.md keeps" column) … | … |
| Member ids namespace-unique across `User`/`Agent` | DESIGN §1.2 → QUERIES §2/§7 |
```

Rules for the rewrite:
- One row per decision from the §2.1 table's "AGENTS.md keeps" column (D6 and D16 collapse to one
  row; D2 points at the existing §1.1 axis).
- **No rationale text, no query bodies, no schema DDL** in AGENTS.md — pointers only.
- Leave the separate "## Live-verified FalkorDB facts" section **as-is** — it is already a lean
  pointer layer (general quirks live in `claude/graph-dba/falkordb-quirks.md`; the rest mirror
  QUERIES §4/§9 notes). Not part of this decision-register move.
- Leave the "## Message write paths (two variants)" invariant list as-is — it is operational
  working-context (write-path contract), and it already links to QUERIES §4 rather than copying
  bodies. (Optional tidy: it may point its role/guard/cursor bullets at DESIGN §1.2; not required.)

---

## 5. plan.md + history.md handling

### 5.1 `docs/BACKLOG.md` — graduate the locked stack, keep a pointer
- Replace the "## Locked M2 stack decisions (2026-07-04, user-approved)" table (lines 56-69) with
  a one-line pointer:
  > **M2 stack (embedding model/dim, agent LLM, runtime, VRAM, upgrade path) is locked in
  > `docs/DESIGN.md` §1.3** (decided 2026-07-04). Implemented in K-008/K-013.
- Keep the `bootstrap_schema.sh` / `EMBEDDING_DIM=1024` operational note that follows (lines
  67-69) — it is a *procedure reminder for K-008*, not a decision store; or move its essence into
  the §1.3 "Detailed in" pointer to §11. Coder's discretion; low stakes.
- Everything else in plan.md (K-011…K-019 items, sequencing, parking lot) is forward-looking work
  → **stays**.

### 5.2 `docs/HISTORY.md` — history-only decision scan
Scanned all entries (2026-06-11 → 2026-07-05) for durable decisions that exist **only** in history
and never reached DESIGN. **Result: none found — the discipline held.** Spot checks:
- REST-over-gRPC transport decision (2026-06-11) → already DESIGN §14.1 (with rejected-gRPC
  rationale). ✅
- `create_channel`/`create_thread` plain `CREATE`, non-idempotent (K-007) → DESIGN §9 row. ✅
- Cursor-vs-limit chronological fix + reader-`isMention` flag (K-004) → QUERIES §9 + DESIGN
  §15.2. ✅
- Flat `GET /messages/{mid}` route (K-005 Fork 4) → DESIGN §14.4. ✅
- `list_channels` creation-time-ordering trade-off (K-001) → QUERIES §3 inline. ✅
- K-005 Fork 3(a) "dead `isMention` highlight removed from JS" → web-implementation decision, not
  §1-register material (correctly log-only).

**Action for history.md: none.** It stays a dated log, untouched. (Belt-and-braces: the durable
decisions it *records* are exactly the K-007/K-010 ones now landing in §1.2 — history keeps the
*event*, DESIGN §1.2 gets the *state*.)

---

## 6. The two folded-in moves (kept in this one plan)

### A1 — De-duplicate the GraphRAG query: DESIGN §8 → pointer to QUERIES §6  *(flagship, do first)*
DESIGN §5.3 already models the discipline ("Canonical Cypher: QUERIES §4 … this section
describes their shape only, so the two never drift"). **§8 breaks that same rule** and has
**already drifted**:
- `DESIGN.md:343` returns `seed.text AS hit, score, [m IN expanded | m.text] AS context` —
  **no `LIMIT`**, no `seed.msgId`/`seed.role`.
- `QUERIES.md:401-404` returns `seed.msgId, seed.text, seed.role, score, [...] AS relatedContext
  … ORDER BY score ASC LIMIT $limit`.

Same query, two RETURN shapes, one missing `LIMIT` — the exact drift the philosophy exists to
prevent. QUERIES §6 also carries the `$channelId` workspace-wide note and the embedding-set
companion query that §8 lacks — QUERIES is unambiguously the fuller, canonical copy.

**Move:** delete the embedded ```` ```cypher ```` block at `DESIGN.md:332-345`. **Keep §8's
prose** (vector+traversal rationale, cosine-distance-ASC note, `GRAPH.RO_QUERY`/replica-lag note,
lines 347-353) — it is design-level. Add the pointer in §5.3's exact form:
> **Canonical Cypher: `docs/QUERIES.md` §6.** This section describes the read path's *shape*
> only, so the two never drift.

**Done looks like:** §8 has no Cypher body; a reader is routed to QUERIES §6; the drift is gone
because there is one body. `grep -c '```cypher' docs/DESIGN.md` drops by one; no fence inside §8.

### A2 — Promote the M3-coordination decision: m1-chat-mcp.md Appendix B → DESIGN §6
`m1-chat-mcp.md` Appendix B (lines 467-504) is an **Accepted** ADR (2026-06-21): coordination
(task lifecycle / "room state") lands later as an **M3 `WorkflowDef` of `kind:'process'`**,
**not** a flat `Task` node; agent *presence* is out of scope. Durable, governs M3 — but exists
**only** inside a completed M1 plan's appendix, absent from DESIGN §6 (which models the workflow
engine) and §13.

**Move (promote-then-link, not a verbatim copy):** add a short DESIGN §6 note, e.g. **"§6.3
Coordination is workflow, not a separate primitive"**, in 2-4 lines:
> Agent/team coordination (task lifecycle, "room state") is modelled as an M3 `WorkflowDef` of
> `kind:'process'` over `Step` + `TRANSITION` + `StepRun` — **not** a flat `Task` node or a
> presence field. This avoids a parallel model that would later need migrating into the engine
> (single-store philosophy). Full rationale/ADR: `docs/archive/plans/m1-chat-mcp.md` Appendix B.

Optionally add a one-line §13 note that coordination is *deferred-to-M3*, not open. Appendix B
**stays** as the ADR of record (full comparison table). **No source deletion.** This is a
decision-promotion (2-4 lines), not a wholesale ADR move.

**Done looks like:** a DESIGN §6 reader learns coordination is deferred-to-and-modelled-by the
workflow engine without opening a completed plan; the plan keeps the reasoning.

---

## 7. Risks & invariants to preserve

1. **Do not violate "QUERIES.md is the single source for query bodies."** A1 *enforces* it (removes
   a copy); it must not spawn a new copy. **No move in this plan adds a query body to DESIGN or
   AGENTS** — decisions D7/D17/D18 point *to* QUERIES; they never copy Cypher.
2. **Do not create DESIGN-internal duplication.** The whole point is one home per decision. A §1.2
   register row is a **statement + rationale + link** to the body — it must NOT re-copy the body's
   mechanics. §5.2 explains the thread model *once*; §1.2 states "thread-scoped `NEXT`" and links.
   The one net-new body line (D13, §6.2) is genuinely new content, not a duplicate.
3. **Do not strip the kaizen historical record.** history.md stays untouched (§5.2); plan.md loses
   only the graduated locked-stack table (replaced by a pointer), never its work items or parking
   lot.
4. **Doc-only ⇒ both suites untouched.** No `repository.py`, `QUERIES.md` body, `test_queries.sh`,
   or schema is touched — pytest **110** and query suite **126/126** hold by construction. If
   QUERIES §6 is opened for A1 it is **read-only** (the canonical body does not change).
5. **A1 must preserve §8's design-level prose** — delete only the Cypher body, keep the rationale
   and replica-lag notes. Do not over-delete.
6. **A2 is decision-promotion, not an ADR move** — 2-4 lines into DESIGN §6, link the rest; keep
   Appendix B's comparison table in the plan.
7. **Respect the K-019 boundary.** Do not fix test counts, the §13 embedding "open" wording, or
   §12/§14.1 scope here. The §1.3 M2-stack graduation *touches the same area* as K-019's §13
   rewrite — **coordinate sequencing** (§2.2 note): add §1.3 first or fold into K-019; never
   double-edit §13.
8. **AGENTS.md must stay lean** — pointer index only, no rationale, no bodies. The "Live-verified
   facts" and "write paths" sections are out of this move (already pointer-shaped).

---

## 8. Execution sequencing (for the coder)

Ordered, each step self-contained; the tree/docs stay coherent after each.

1. **A1 — DESIGN §8 dedup** (first; highest value, lowest risk). Delete the ```` ```cypher ````
   block at `DESIGN.md:332-345`; keep the prose; add the "Canonical Cypher: QUERIES §6" pointer in
   the §5.3 style. Verify QUERIES §6 untouched. *Check:* `grep -c '```cypher' docs/DESIGN.md`
   dropped by one; no fence in §8.
2. **DESIGN §1 restructure** — split §1 into §1.1 (existing axes, unchanged), **§1.2** (detailed
   register, the 16-row table from §3), **§1.3** (M2 stack, the 6-row table from §3). Add the one
   new §6.2 body line for D13 (flat `ctx`). *Check:* every AGENTS decision + every plan.md
   locked-stack row has a §1.2/§1.3 home.
3. **AGENTS.md rewrite** — replace the "Decisions locked in" rationale table with the terse
   pointer index (§4 format); one row per decision, linking to §1.2/§1.3 (or §1.1 for D2). Remove
   all rationale text. *Check:* no rationale, no Cypher, no DDL in the section; row count matches.
4. **plan.md graduation** — replace the "Locked M2 stack decisions" table (lines 56-69) with the
   one-line pointer to DESIGN §1.3 (§5.1). Keep the `EMBEDDING_DIM=1024` procedure note (or fold
   its essence into §1.3's §11 pointer). *Check:* no decision table remains in plan.md; work items
   intact.
5. **A2 — promote coordination decision** — add DESIGN §6.3 note (2-4 lines) + back-link to
   `m1-chat-mcp.md` Appendix B; optional §13 one-liner. Appendix B unchanged. *Check:* DESIGN §6
   states the decision; plan retains the ADR.
6. **history.md** — no edit (verified no history-only durable decision). Optionally add one dated
   log entry summarising this sweep (e.g. `K-020: doc-architecture consolidation — §1 register`)
   if the user wants the event recorded.
7. **K-019 coordination** — if K-019 has not yet rewritten §13, ensure the §1.3 add (step 2) and
   K-019's §13 "open"→"decided" rewrite land in one coherent pass (do not double-edit §13).

**No suite re-run required** (doc-only); the cheap final check is `grep -c '```cypher'
docs/DESIGN.md` (§8 body gone) and a grep confirming AGENTS.md's decisions section carries no
rationale prose.

---

## Ready to implement

- **16 AGENTS.md decisions migrate into DESIGN §1.2** (D6+D16 merge; D2 already lives in the §1.1
  axes table). Of the 16: **13 already have a DESIGN body home** (§1.2 row = statement + link),
  and **5 are net-new** — D4 (No DayBucket, rejected-alt), D13 (flat `ctx`/`input`/`output`, needs
  one new §6.2 line), and D7/D17/D18 (mechanics canonical in QUERIES §2/§7/§9 — §1.2 row points
  there, no DESIGN body copy).
- **6 plan.md locked-stack decisions graduate into a new DESIGN §1.3** ("M2 stack — decided,
  pending implementation"); plan.md keeps a one-line pointer. Two of the six (embedding model +
  dim) overlap K-019's §13 rewrite — sequence/fold to avoid double-editing §13.
- **history-only decisions found: none.** The K-007/K-010/K-002 fold-ins already pushed every
  durable decision into DESIGN/QUERIES; history keeps the dated *events* and stays untouched.
- **AGENTS.md "Decisions locked in"** becomes a terse pointer index (rationale removed, one line +
  DESIGN link per decision); it is not deleted.
- **Two folded-in moves:** A1 — delete the drifted GraphRAG Cypher from DESIGN §8 (lines 332-345,
  missing `LIMIT`, stale RETURN cols) → pointer to QUERIES §6; A2 — promote the M3-coordination ADR
  decision from `m1-chat-mcp.md` Appendix B → 2-4 lines in DESIGN §6 + back-link.
- **Surprising:** the one *active* philosophy violation is self-inflicted and already caught
  drifting — DESIGN §8's own GraphRAG query broke the "shape-only, link the body" rule that DESIGN
  §5.3 states three sections earlier. And three genuinely design-level invariants (flat
  `ctx`/`input`/`output`, `coalesce` member identity, `Message.role` derivation) live *only* in
  AGENTS.md / QUERIES.md, never in the DESIGN register — AGENTS.md, not DESIGN, was the de-facto
  spec for them.
- **Owner:** `coder` (doc-only). A2 optionally routes a one-line review past the user (promotes a
  design decision). Doc-only ⇒ pytest 110 / query suite 126/126 untouched by construction.
```