# Kaizen agent ontology — graph data model for M8

> **Status:** archived · **Owner:** `graph-dba` · **Tracks:** — (M8)

Design for the data-shape slice of
[`../requirements/kaizen-agent-ontology.md`](../requirements/kaizen-agent-ontology.md) (FR-1…FR-8,
AC-1…AC-7), read in full including its Decision log — relationship names/directions (`PRODUCED`,
`MENTIONS`) and the `author` property's removal are **locked**, not relitigated here. Targets the
substrate M7 actually shipped: one shared `kaizen_team` graph, `author`-partitioned, all 12 team
agents writing to it (`docs/plans/generic-cypher-mcp2.md`, archived, confirmed read for this
note — not the old per-agent `kaizen_<agent>` shape). Supersedes
[`generic-cypher-mcp-graph.md`](./generic-cypher-mcp-graph.md) §2, which explicitly weighed and
rejected an `:Agent` node for the M5 pilot with a stated revisit trigger — this document is that
trigger firing, cited there as prior-decision provenance, not re-derived.

No files are edited, no graph is mutated by this note — design only, same scope discipline as my
own M5 note. Non-goals (left to `architect`, U2, which reads this note by path): which files/agents
get updated and in what order, `cypher-mcp/server.py` code-change specifics beyond the
write-shape/authorization-compatibility analysis below, `cobb`'s distillation-cadence process
changes beyond the query shapes given here, and documentation updates.

---

## 0. Schema

```
(:Agent {agentId})                                          // identity-only — see §1
(:Agent)-[:PRODUCED {sessionId}]->(:KaizenEntry)             // FR-2, locked name/direction — §2
(:KaizenEntry)-[:MENTIONS]->(:Agent)                         // FR-3, locked name/direction, 0..N — §3
```

`PRODUCED` and `MENTIONS` match `falkor-chat/docs/DESIGN.md`'s conventions **exactly**, name and
direction, confirmed by direct read of that file for this note:
`(:StepRun)-[:PRODUCED]->(:Message)` (§6.2, D2 — "PRODUCED, not EMITTED... StepRun→Message
emission is a distinct edge type") and `(:Entity)<-[:MENTIONS]-(:Message)`, i.e.
`(:Message)-[:MENTIONS]->(:Entity)` (§5.1) — content/artifact node points at its referent, creator
points at its artifact. Both patterns transplant onto this graph unchanged: `:Agent` stands in for
`:StepRun`'s and `:Message`'s creator role (`PRODUCED`, agent→artifact) and for `:Entity`'s referent
role (`MENTIONS`, artifact→agent).

`:KaizenEntry`'s own five markdown-sourced fields plus `entryId`/`createdAt` are unchanged from
`generic-cypher-mcp-graph.md` §1 — only `author` is dropped (FR-2, locked) and `sessionId` moves
off the node onto the `PRODUCED` edge (FR-8, new in M8). Node schema after M8, for entries created
post-ship:

| Property | Where it lives now | Change from M5/M7 |
|---|---|---|
| `entryId`, `date`, `fact`, `evidence`, `context`, `suggestedHome`, `createdAt` | `:KaizenEntry` node | unchanged |
| ~~`author`~~ | — | **dropped outright** (FR-2) — no coexisting field, no retrofit |
| `sessionId` | `PRODUCED` relationship | **moved** off the node (FR-8) — new entries only; M7-window entries (created after M7 shipped, before M8) keep it on the node, unmigrated (AC-7) |

**Agent node — identity-only** (requirements doc's one open question, resolved per its own default):
a single property, `agentId`, the same slug already used everywhere else in this system — the
former `author` string value, the MCP tool's `agent` parameter, and `CYPHER_MCP_CURATOR_AGENTS`'
members are all the identical string space (`'graph-dba'`, `'cobb'`, `'analyst'`, …). No `name`,
`kind`, or `role` property — nothing in FR-1…FR-8 or the three stated payoffs (attribution lookup,
provenance tracing, duplicate-spotting) needs more than identity to be reachable as a traversal
target. **`agentId` is also `falkor-chat`'s own established property name for the identical
concept** (confirmed by direct grep of `falkor-chat/docs/QUERIES.md`: `(:Agent {agentId: $agentId,
name: $name, model: $model, ...})` — that graph's `:Agent` carries more properties because it
represents a configured chat participant, a different domain; the identity property name is the
same choice independently, which is a good sign this is the right name, not a coincidence to
special-case). Revisit trigger, same shape as M5 §2's: if `cobb` ever wants to query "which agents
are currently active" or attach kind/role metadata, that is an additive property, not a schema
migration.

---

## 1. The crux: `authorize_write()` cannot authorize *any* new-shape write today

This is the point of this document, stated up front rather than buried. Read in full:
`cypher-mcp/server.py`'s `authorize_write()`, `_author_claims()`, `_kaizen_entry_create_map_spans()`
(lines ~334–381 and ~291–345), and `cypher-mcp/README.md`'s "Writing through this tool" section.

**Today's mechanism recognizes exactly two write shapes, both keyed on the `author` property**:

1. **Author-write**: a `CREATE (<var>:KaizenEntry {...})` clause whose map body contains a literal
   `author: '<value>'` matching the declared `agent` parameter exactly. `_author_claims()` finds
   this by first locating map-literal spans that immediately follow the `CREATE` keyword
   (`_kaizen_entry_create_map_spans()`: `re.match(r"\s*\(\s*[a-zA-Z_]\w*\s*:\s*KaizenEntry\s*\{",
   tail)` — the `KaizenEntry`-labeled node **must be the pattern element immediately following
   `CREATE`**, not reached via a relationship from a prior node), then scanning inside for the
   `author:` literal.
2. **Curator-clear**: the one whitelisted shape `MATCH (var:KaizenEntry {entryId: '...'}) DETACH
   DELETE var`, gated to `CURATOR_AGENTS` (default `cobb`).

If neither matches, `authorize_write()` unconditionally rejects. There is no third path.

**FR-2 drops `author` outright.** A new entry's `CREATE` clause therefore never contains an
`author:` literal, by design. Consequence, independent of anything about `MERGE`:
**`_author_claims()` always returns an empty list for any FR-2-conformant write, which
`authorize_write()` then falls through past `if claims:` straight into the curator-clear check —
which a `CREATE` never matches — and rejects.** This is true for *every* way of shaping the new
entry-creation write, not a corner case: the mechanism's only non-curator "allow" path is finding
an `author:` literal, and FR-2 removes the one thing that path looks for. **Today's
`authorize_write()`, unmodified, rejects 100% of new-shape entry-creation writes once `author` is
gone** — this is not a `MERGE`-vs-`CREATE` nuance, it is the removal of the mechanism's only
recognition signal.

**A second, independent reason, worth stating separately because it would still bite even if
`author` survived**: the natural producer-write shape puts the `KaizenEntry` node **second** in the
pattern, reached via a relationship from the `Agent` node —
`CREATE (a)-[:PRODUCED {...}]->(k:KaizenEntry {...})` — not as the pattern element immediately
after `CREATE`, which is what `_kaizen_entry_create_map_spans()` requires. Even a
hypothetical FR-2-noncompliant design that kept `author` as a redundant node property would still
fail today's scan, because the regex's anchor is "`CREATE` immediately followed by `(var:KaizenEntry
{`", and this shape's first pattern element after `CREATE` is `(a)`, not `(k:KaizenEntry ...)`.

**On `MERGE` specifically** (the open question the requirements doc raises): `_WRITE_KEYWORD_RE`
(used only for the empty-key pre-classification branch, not for authorization itself) already lists
`MERGE` as a write keyword, and since `kaizen_team` already exists post-M7, every new write in this
design routes through the *other* branch (`RO_QUERY` rejects it as a write on an existing graph →
`authorize_write()` runs unconditionally) — the empty-key branch and its `_looks_like_write()`
pre-classification never enter into it. But `authorize_write()` itself has **no code path that
inspects a `MERGE` clause at all** today; it only ever extracts `CREATE ...:KaizenEntry {` spans.
So `MERGE (a:Agent {agentId: '<value>'})` is invisible to the current scanner regardless of what
else the statement contains.

**Recommendation: `authorize_write()` needs new recognized-shape logic — this is not avoidable by
picking a clever write shape.** My concrete recommendation for what to recognize (schema-level
intent; the actual regex/parsing is `cypher-mcp` code, `architect`/`coder`'s call, not designed
here):

- **Producer-write** (§2 below): authorize when the statement is exactly one `MERGE
  (<v1>:Agent {agentId: '<value>'})` clause followed by exactly one
  `CREATE (<v1>)-[:PRODUCED {...}]->(<v2>:KaizenEntry {...})` clause, `<value>` matching the
  declared `agent` — the direct structural analog of today's author-claim check, just anchored on
  the `Agent` node's identity literal instead of a property inside the entry's own map. Not
  curator-gated — any agent may run this for its own `agentId`.
- **Curator MENTIONS-write** (§3 below), **curator producer-edge-resolve**, and **curator
  mention-edge-resolve** (§4.2 below): three new narrow shapes, each the same style of tight,
  single-purpose regex as today's one curator-clear shape, each gated to `CURATOR_AGENTS`.
- The existing curator-clear shape (`MATCH (var:KaizenEntry {entryId:...}) DETACH DELETE var`)
  is **unchanged** — it still correctly expresses "delete the whole node, whatever edges it has"
  (§4.1 below), and continues to leave any `:Agent` endpoints untouched (a `DETACH DELETE` on `k`
  removes only `k`'s own incident edges and `k` itself, never the `Agent` nodes those edges
  pointed at).

Net effect: `authorize_write()` grows from **2** recognized write shapes to **6** (1 producer-write
+ 3 curator-resolve shapes + 1 curator-MENTIONS-write + the unchanged curator-clear). Sizing this
for `architect`: every new shape is the same complexity class as the existing curator-clear regex
(a fixed skeleton, whitespace-collapsed, `agentId`/`entryId` as the only variable parts) — not a
general Cypher parser, consistent with FR-8's own stated trust bar ("well-behaved callers can't do
this by accident," not hardened against a malicious one).

---

## 2. Producer write (FR-2, FR-8)

One statement, both the entry and its producer edge, `sessionId` on the edge:

```cypher
MERGE (a:Agent {agentId: '<agent-slug>'})
CREATE (a)-[:PRODUCED {
  sessionId: '<value of $CLAUDE_CODE_SESSION_ID, or omit this key entirely if unavailable>'
}]->(k:KaizenEntry {
  entryId: '<uuid4>', date: '<YYYY-MM-DD>', fact: '<the fact, one line>',
  evidence: '<what was run/read/observed>', context: '<the task where it surfaced, one line>',
  suggestedHome: 'prompt | knowledge base | project docs | unsure',
  createdAt: '<ISO-8601 write time>'
})
```
called as `mcp__cypher__query(graph='kaizen_team', cypher=<that text>, agent='<agent-slug>')`.

- **`MERGE` for the `Agent` node, `CREATE` for the edge and entry.** `agentId` is a stable identity
  that must be idempotent across an agent's many writes (first note creates the node, every
  subsequent note reuses it — exactly the "an agent's first note creates its node" mechanism the
  task asks about); `entryId` is fresh every time, so `CREATE` is correct and cheaper there — no
  reason to pay `MERGE`'s match cost for a node that can never collide.
- **No `params=` channel** (`cypher-mcp` has none by design — literals substituted into the query
  text, same as every M7 recipe). The angle-bracket placeholders above are substituted by the
  calling agent before the call, not passed as bound parameters.
- This is a **single** `GRAPH.QUERY` call — ordinary multi-clause Cypher (`MERGE` binds `a`, `CREATE`
  consumes it), no special engine behavior needed; nothing in `falkordb-quirks.md` suggests a
  `MERGE`-then-`CREATE` pair sharing a variable behaves any differently than `MATCH`-then-`CREATE`
  does, which this schema's own `falkor-chat` precedent already relies on throughout.
- **DDL provisioning must land first**, mirroring M7's `S0`: `Agent.agentId` needs a supporting
  index + uniqueness constraint (§5) *before* any agent starts calling this recipe — `MERGE` without
  a backing uniqueness constraint is exactly this agent's own standing "duplicate-node bug waiting
  for concurrency" warning. Sequencing that DDL is `architect`'s call; I'm stating the hard
  dependency, not the unit.

---

## 3. MENTIONS write (FR-3, FR-4 — `cobb`, during distillation)

```cypher
MATCH (k:KaizenEntry {entryId: '<entry-id>'})
MERGE (a:Agent {agentId: '<mentioned-agent-slug>'})
MERGE (k)-[:MENTIONS]->(a)
```
called as `mcp__cypher__query(graph='kaizen_team', cypher=<that text>, agent='cobb')`.

- **`MERGE` for the `Agent` node** — the mentioned agent may never have produced an entry itself
  (no node yet), so this write cannot assume the node exists.
- **`MERGE` for the edge itself, not `CREATE`** — `MENTIONS` carries no properties, so `MERGE` on
  the full `(k)-[:MENTIONS]->(a)` pattern (both endpoints already bound) is a safe, free idempotency
  guard against `cobb` double-tagging the same entry/agent pair across two passes — no backing
  uniqueness constraint is needed for this, since the pattern match itself (both endpoints bound,
  no properties to disambiguate) is already the full key.
- **Authorization**: this is genuinely a new curator-write shape, not covered by today's mechanism
  at all — `_kaizen_entry_create_map_spans()` only ever looks at `CREATE`, and this statement has no
  `CREATE` clause and touches no `KaizenEntry` map literal. **Flagging for `architect`, per the
  task's own framing, since this is an authorization-logic decision, not a pure schema one**: my
  recommendation is a fourth recognized shape, gated to `CURATOR_AGENTS` exactly like the existing
  curator-clear regex — narrow to this exact skeleton (`MATCH (var:KaizenEntry {entryId:'...'})
  MERGE (var2:Agent {agentId:'...'}) MERGE (var)-[:MENTIONS]->(var2)`), not a general "curator can
  write anything" carve-out. `cobb`'s existing curator status (already recognized for
  `DETACH DELETE`) is the right authority to extend — this is additional shape recognition, not a
  new class of authorized actor.

---

## 4. Deletion — full node vs. partial edge (FR-6, AC-3, AC-4)

All resolution/deletion actions are `cobb`'s, run during its per-agent distillation pass — the
model established in the requirements doc's Decision log ("cobb reviews one agent's universe at a
time... MENTIONS is a routing mechanism into another agent's review") is that distillation itself,
edge resolution included, is `cobb`'s job throughout; no shape below is run by the producing agent
itself.

### 4.1 Read first: how many edges remain

```cypher
MATCH (k:KaizenEntry {entryId: '<entry-id>'})
OPTIONAL MATCH (:Agent)-[p:PRODUCED]->(k)
OPTIONAL MATCH (k)-[m:MENTIONS]->(:Agent)
RETURN count(DISTINCT p) AS producedEdges, count(DISTINCT m) AS mentionEdges
```

Two `OPTIONAL MATCH`es off the same anchor `k` do fan out against each other (the classic risk
`falkordb-quirks.md` flags for sequential un-collapsed `OPTIONAL MATCH`/`UNWIND` blocks), but the
counts still come out correct here specifically because `PRODUCED` is single-valued per entry by
construction (§2 creates exactly one, ever) — `count(DISTINCT p)` collapses to 0 or 1 regardless of
how many `mentionEdges` rows the cross product produces, and `count(DISTINCT m)` correctly counts
every distinct mention edge across those same rows. This only holds because of that
single-producer invariant; don't reuse this two-`OPTIONAL MATCH`-off-one-anchor shape for a pair of
relationship types that could both be multi-valued without collapsing each to its own `WITH ...
collect(...)` step first, per the general quirk.

`cobb` computes `otherRemaining = producedEdges + mentionEdges - 1` (subtracting the one edge it is
about to resolve in this pass) to choose between §4.2's two shapes:

### 4.2 Delete just the one edge being resolved (`otherRemaining > 0`)

Producer's own pass (FR-6: "always resolved... regardless of how many mention relationships still
point at the note"):
```cypher
MATCH (:Agent)-[p:PRODUCED]->(k:KaizenEntry {entryId: '<entry-id>'})
DELETE p
```

One mentioned-agent's pass, resolving only that agent's edge (AC-3: "only that one relationship is
removed and the note persists"):
```cypher
MATCH (k:KaizenEntry {entryId: '<entry-id>'})-[m:MENTIONS]->(:Agent {agentId: '<mentioned-agent-slug>'})
DELETE m
```

Both are plain `DELETE` on a bound relationship variable (no `DETACH` needed — nothing but the edge
itself is being removed), confirmed workable on this build by `falkordb-quirks.md`'s own note that
"`DELETE` inside a `FOREACH`... works," a strictly harder case than a bare top-level `DELETE`. Two
new curator shapes for `authorize_write()` (§1) — narrow, `entryId`+`agentId` as the only variable
parts, same style as the existing curator-clear regex.

### 4.3 Delete the whole node (`otherRemaining == 0` — this is the last edge)

**Unchanged from M5/M7 — already authorized today, no new shape needed:**
```cypher
MATCH (k:KaizenEntry {entryId: '<entry-id>'})
DETACH DELETE k
```
`DETACH DELETE` removes `k`'s own incident edges (whichever of `PRODUCED`/`MENTIONS` still existed)
and `k` itself; the `Agent` node(s) on the other end of those edges are untouched — `Agent` nodes are
never deleted, by design (§0), regardless of how many entries have been cleared.

---

## 5. Query patterns for `cobb`'s per-agent queue (FR-5, AC-2, AC-5)

"Every note produced by or mentioning agent X," union across both relationship directions:

```cypher
MATCH (a:Agent {agentId: '<agent-slug>'})-[:PRODUCED]->(k:KaizenEntry)
RETURN k.entryId AS entryId, k.date AS date, k.fact AS fact, k.evidence AS evidence,
       k.context AS context, k.suggestedHome AS suggestedHome
UNION
MATCH (k:KaizenEntry)-[:MENTIONS]->(a:Agent {agentId: '<agent-slug>'})
RETURN k.entryId AS entryId, k.date AS date, k.fact AS fact, k.evidence AS evidence,
       k.context AS context, k.suggestedHome AS suggestedHome
```

`UNION` (not `UNION ALL`) is deliberate: it de-duplicates by row, which matters if an entry were
ever both produced by and mentioning the same agent (not expected in practice, but free correctness
at no extra cost). **Not live-verified against this pinned build** — `UNION` is core openCypher and
I have no reason to expect it's unsupported, but `falkordb-quirks.md` doesn't carry a confirming
entry either, so flag it for a quick live check (or `qa-engineer`'s pass) before relying on it,
particularly a trailing `ORDER BY` across the union (omitted above — sort client-side, or confirm
`ORDER BY date` live before adding it back). A verified fallback that avoids `UNION` entirely, using
the same collapse-before-fan-out idiom as §4.1's note:

```cypher
MATCH (a:Agent {agentId: '<agent-slug>'})
OPTIONAL MATCH (a)-[:PRODUCED]->(produced:KaizenEntry)
WITH a, collect(DISTINCT produced) AS producedList
OPTIONAL MATCH (mentioned:KaizenEntry)-[:MENTIONS]->(a)
WITH producedList, collect(DISTINCT mentioned) AS mentionedList
UNWIND (producedList + mentionedList) AS k
RETURN DISTINCT k.entryId AS entryId, k.date AS date, k.fact AS fact, k.evidence AS evidence,
       k.context AS context, k.suggestedHome AS suggestedHome
ORDER BY date
```
(`collect()` drops `NULL`s, so an agent with zero `PRODUCED` or zero `MENTIONS` entries correctly
contributes `[]` to that half, not `[null]`; the final `UNWIND` over an all-empty combined list
correctly yields zero rows for an agent with nothing pending — `falkordb-quirks.md`'s empty-`UNWIND`
warning is about *dropping an unrelated required write* downstream in the same query, which doesn't
apply here since this is a pure read with nothing else to lose.)

**AC-5** ("two notes... mention a third common agent... a query starting from that third agent's
node reaches both") is the same query, unfiltered by producer: the `MENTIONS`-side `MATCH` alone,
run from the mentioned agent's node, reaches every entry mentioning it regardless of who produced
each one — no special-casing needed, it falls out of the schema directly.

---

## 6. Indexing (`Agent.agentId`)

**Recommendation: index + uniqueness constraint, same as every other identity property in this
schema (`entryId`) and in `falkor-chat`'s own convention** ("every entity node has a stable
`{label}Id` property, a range index, and a uniqueness constraint"). At this graph's actual scale
(12–13 `Agent` nodes, however many un-promoted `KaizenEntry` nodes) a full label scan over a dozen
`Agent` rows costs nothing measurable — this is an **integrity** recommendation first, a performance
one only incidentally: `MERGE (a:Agent {agentId: '<value>'})` (§2, §3) is exactly the "`MERGE`
without a backing uniqueness constraint is a duplicate-node bug waiting for concurrency" case this
agent warns about generally, and the constraint is what makes "two `Agent` nodes can't silently
share an `agentId`" an engine-enforced fact rather than a hoped-for one.

```cypher
CREATE INDEX FOR (a:Agent) ON (a.agentId)
```
```
GRAPH.CONSTRAINT CREATE kaizen_team UNIQUE NODE Agent PROPERTIES 1 agentId
```

Index-before-constraint ordering per `falkordb-quirks.md` (`GRAPH.CONSTRAINT CREATE` fails with
"missing supporting exact-match index" otherwise); the constraint keyword is `NODE`, not `LABEL`
(corrected on this build, `falkordb-quirks.md`, verified 2026-08-18); creation is **async** — the
command returns `PENDING` immediately, poll `CALL db.constraints()` for `status = OPERATIONAL`
before treating it as enforced.

**Both statements must run via `redis-cli GRAPH.QUERY` against the container, not through
`mcp__cypher__query`** — confirmed by direct read of `cypher-mcp/README.md`'s "Writing through this
tool" section: schema DDL is rejected by `authorize_write()` the same as any other non-`:KaizenEntry`
write, **with no carve-out for schema statements, even from a recognized curator agent** — this was
live-verified during M7's own `S0` unit ("`CREATE INDEX FOR (e:KaizenEntry) ON (e.entryId)` and
`GRAPH.CONSTRAINT CREATE` are rejected the same as any other non-`KaizenEntry` write... had to fall
back to `redis-cli GRAPH.QUERY`"). This DDL is a new provisioning step analogous to M7's `S0` —
sequencing it (hard predecessor of every producer-write and MENTIONS-write, mirroring `S0`'s role
for `kaizen_team` itself) is `architect`'s call, not designed here.

---

## 7. Coexistence with historical entries (FR-2's no-retrofit rule)

Every query in §5 is **traversal-anchored on `:Agent`** — a historical entry (pre-M8, `author`
property, no `PRODUCED`/`MENTIONS` edges at all) simply has no edge for either query to traverse,
so it is silently absent from both §5 recipes, never an error. This is the correct, intended
behavior under FR-2's no-retrofit decision, not a gap to fix: **these queries do not, and are not
meant to, cover historical entries.**

`cobb`'s pre-existing property-filtered query still works unchanged for that population and remains
the way to review it:
```cypher
MATCH (e:KaizenEntry) WHERE e.author = '<agent-slug>' RETURN e.entryId, e.date, e.fact, e.evidence,
       e.context, e.suggestedHome ORDER BY e.date
```
Practical implication for `cobb`'s distillation process (flagged, not designed — process changes are
this document's own stated non-goal): for as long as any pre-M8 entry remains un-cleared, `cobb`
needs **both** query shapes side by side to see its full queue — the old `author`-filtered read for
legacy entries, §5's traversal for everything created after M8 ships. Once every legacy entry has
been cleared through the existing curator-clear path, the old query permanently returns nothing and
can be dropped.

---

## 8. Footprint note

Negligible, same conclusion as M5 §6 for the same reason (bounded by clear-on-promote, not by a
node-count cap) — the only new element is 12–13 permanent `:Agent` nodes (one property each,
~tens of bytes) that, unlike `:KaizenEntry`, are **never** deleted. **One supernode watch worth
naming, not urgent at current scale**: an agent that gets mentioned unusually often accumulates
incoming `MENTIONS` edges at a rate `cobb`'s distillation cadence controls (each is cleared on that
agent's own next reviewed pass, per §4) — if a backlog were ever allowed to build up across many
agents simultaneously mentioning one popular agent, that node's fan-in grows; not a concern at this
team's actual working-memory scale (single-digit-to-low-dozens of un-promoted entries, per M5 §6's
own estimate), worth re-checking with `GRAPH.PROFILE` only if that scale assumption is ever violated
in practice.

---

## 9. Summary for the implementer (`architect`, U2)

- Schema: `(:Agent {agentId})`, `(:Agent)-[:PRODUCED {sessionId}]->(:KaizenEntry)`,
  `(:KaizenEntry)-[:MENTIONS]->(:Agent)` (§0) — identity-only `Agent` node, locked names/directions
  matched exactly to `falkor-chat`'s `PRODUCED`/`MENTIONS` precedent.
- **The crux (§1): `authorize_write()`'s only non-curator "allow" path is finding an `author:`
  literal inside a `CREATE (...:KaizenEntry {...})` clause — FR-2 removes the one thing it looks
  for, so every FR-2-conformant write is rejected by the mechanism as it stands today, independent
  of `MERGE`/`CREATE` choice.** Six recognized write shapes are needed in total (up from 2) —
  producer-write (§2, any agent, its own `agentId`), curator MENTIONS-write (§3), two curator
  partial-edge-delete shapes (§4.2), the unchanged curator full-`DETACH DELETE` (§4.3) — each a
  narrow fixed-skeleton regex, same complexity class as today's one curator-clear shape, not a
  general parser.
- Producer write (§2): one statement, `MERGE` the `Agent` node + `CREATE` the `PRODUCED` edge
  (carrying `sessionId`) + the `KaizenEntry` node together.
- MENTIONS write (§3): `cobb`-only, during distillation; `MERGE` throughout (idempotent against
  double-tagging); flagged to `architect` as a new curator-authorization-logic decision, with a
  concrete recommendation given.
- Deletion (§4): read-then-decide — count remaining edges first (§4.1), then either delete just the
  one edge being resolved (§4.2, two new shapes) or the whole node once nothing else remains (§4.3,
  unchanged, already authorized).
- Query patterns (§5) for `cobb`'s per-agent queue: `UNION`-based primary recipe (unverified on this
  build, flag for a live check) plus a verified-idiom fallback that avoids `UNION` entirely.
- Indexing (§6): index + uniqueness constraint on `Agent.agentId`, integrity-motivated more than
  performance-motivated at this scale — provisioned via `redis-cli GRAPH.QUERY`, **not** through
  `mcp__cypher__query` (schema DDL is rejected there unconditionally, live-confirmed by M7's `S0`).
- Historical entries (§7): silently, correctly excluded from every new query — `cobb` needs the old
  `author`-filtered read alongside the new traversal for as long as any legacy entry remains
  un-cleared.
- Footprint (§8): negligible; one supernode watch named for a scale this team hasn't reached.
