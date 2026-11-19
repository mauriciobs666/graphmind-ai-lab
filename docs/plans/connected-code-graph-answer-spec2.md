# Connected code graph: canonical answer spec, P1 changes (schema 1.2, `bfs-v1` 1.2)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1) · **Extends:** `docs/plans/connected-code-graph-answer-spec.md`

Revised 2026-11-19: V-both is the shipped flow-fallback table (§5, §1.2: `bfs-v1` `1.2`) and every flag-false answer carries the flow notice (§5.1: schema `1.2`, V20).

This document states **only the changes** P1 makes to `docs/plans/connected-code-graph-answer-spec.md` (v1.3, "v1"
below); anything not named here is as v1 says. It is step P1-00 of `docs/plans/connected-code-graph-p1-answers.md`,
whose scope list it covers in full. It extends rather than edits v1 because S34 executed against v1. The code and
schema edits it requires are P1-45's (§7, §8). The engine behaviour it fixes is P1-22's, P1-23's, P1-25's and
P1-26's.

Decisions it relies on:
- the stakeholder's 2026-11-19 decision on the flow fallback (§5): V-both ships; FR-8 and AC-25's control half are recorded not met for flag-false flow; the answer states the omission (§5.1);
- the P1-39c result `docs/plans/connected-code-graph-p1-flow-fallback-result-ml.md` (§2) and its record `ccg-p1-flow-fallback-v1`;
- index `docs/plans/connected-code-graph-p1.md` D2, D3 and §5;
- the eval protocol `docs/plans/connected-code-graph-eval-protocol-ml.md` ("EP") §3.5, §5.2.1, §10.4, §12.3;
- graph model v1 `docs/plans/connected-code-graph-p0-graph.md` §1.5, §3, and its P1 successor
  `docs/plans/connected-code-graph-p1-graph.md` §1–§3 and §5 (being revised; cited by section);
- graph note `docs/plans/connected-code-graph-graph.md` §6.2, §6.3;
- the method note `docs/plans/connected-code-graph-p1-ml.md` §6.5;
- the stakeholder's 2026-11-09 decision on flow labelling (§5).

CPG: considered, not relevant — a contract specification; no loaded `cpg_*` graph indexes `cpg/`, so every code fact below was read directly from `cpg/ccg/answer.py`, `cpg/ccg/schemas/answer-v1.schema.json`, `cpg/tests/test_answer.py`, `cpg/tests/golden/answer-ac1.json` and `cpg/spikes/vertical-slice/impact.py`.

## 1. Version values

### 1.1 `answerSchemaVersion` is `"1.2"`

`"1.2"` adds one optional root field, `notices` (§5.1), and rule V20; every `"1.1"` and `"1.0"` document stays valid, so by v1 §9 it is a
**minor** bump. `"1.1"` is published (the builder `ccg.engine.answer_build.build_answer` emits it on `main`, from commit `d02de630`), and a
published label is never edited, so the field cannot join `"1.1"`.
- The P1 engine emits `"1.2"`. The validator accepts `"1.0"`, `"1.1"` and `"1.2"` against one schema.
- V17 keeps a `"1.1"` label free of `notices` (§7.2). Every rule this document states for `"1.1"` documents (V17's bound, V18, V19, the `facts_bytes`
  projection) applies to `"1.1"` and `"1.2"` alike; V20 and §5's callee run (V18's `CALLEE_ROWS` and clauses (f), (g)) apply to `"1.2"` only. A `"1.1"` document
  is accepted and rejected exactly as before: a V-both chain (a `HAS_CALL →` exit from `function`) stays a V18 (e) rejection under `"1.1"`.
- `notices` is part of the facts (it is not dropped by `facts_bytes`).

What `"1.1"` added: §2–§3 add optional fields and one hop kind, and relax two rules:
- `Bound.keyPath` is no longer always required.
- V5 now admits the wiring and depends-on `Link` shapes. As v1 wrote V5, it rejected every depends-on chain through
  a link.

V17 (§7.2) keeps a `"1.0"` label free of 1.1 constructs, and V18 and V19, which are tightenings, apply to `"1.1"` and `"1.2"` documents only.
The stored golden `cpg/tests/golden/answer-ac1.json` stays `"1.0"` and unchanged.

v1 §9's rules are otherwise unchanged.

### 1.2 `recipe.algorithm.version`: the value P1 ships and the bump rule (carry-over C6)

**The value is `"1.2"`.** `"1.1"` names the caller-only walk and is published; V-both is a changed definition (§5), so it takes the next minor.
The schema constrains `Recipe.algorithm.version` only by the pattern `^[0-9]+\.[0-9]+$`, and `ccg/answer.py` checks no
value. The golden carries `"1.0"`, the name of the slice's walk (`impact.py`, v1 §11).

**What a version names.** One version names exactly one **walk definition**. The definition is:
- the row table: for each row, the edge type, `dir`, column (impact, contract, depends-on), from-state kinds,
  to-state kind, distance increment, and status/reason labelling, including §5's labelled rows and callee run, §3.3's wiring row and §3.5's `TO_UNIT` row;
- the item rules: which arriving hop makes an item, per column (`symbol`, `config`, `unit`);
- the cost order, which is `(possible hops, distance, length)`, then fewer hops whose own edge is `possible` (§5),
  then the predecessor's stable key;
- the flow-fallback rule (§5);
- `DEFAULT_DEPTH_BOUND` (index D3);
- the `unexplored` counting rule (§4).

The following are **not** part of it, because they cannot change a fact:
- query texts, `CHUNK`, `RESULTSET_SIZE` and `LEVEL_GUARD` (the last two raise and never truncate);
- anything read from the snapshot, `Unit.flowAvailable` values included (§9 R-2).

**Bump rule.**
- Any change to the definition increments MINOR by one.
- MAJOR stays `1` for `algorithm.id` `bfs-v1`. A walk that is not a level-synchronous BFS with a state per entry
  gets a new `algorithm.id`, and that is a schema change (the `id` is a `const`).
- Versions are never reused, and a published entry is never edited. An entry is **published** once an answer
  carrying it has been emitted on `main`. `1.1` is published by `ccg/engine/answer_build.py` (from commit `d02de630`); its fingerprint pin
  (`ALGORITHM_VERSIONS` in `ccg/engine/bfs.py`) is `5cf047c14e43da44d0363707d3036080`, stays in the mapping and is never edited. `1.2` is
  the V-both definition; P1-23d pins its literal from the shipped record. A change to either needs a new algorithm version.

**Mechanism (P1-22 builds it).**
- The walk executes its row table as data. The row table is a record the walk interprets, not a description kept
  beside the code.
- The definition record holds that table, the item rules as data, and one identifier string for each part that is
  code: the cost order, the flow-fallback rule and the `unexplored` rule. A table field no code branches on is not in
  the record (it would version the table without constraining the walk), and a read an adapter issues beyond its
  row's own (the `CONSUMES <-` anti-join reads `ON <-` and `VIA <-`) is named in the record. One example identifier is
  `"costOrder": "possible,distance,length,own-possible,stable-key"`.
- The engine holds `ALGORITHM_VERSIONS`, a mapping from version to a pinned literal fingerprint. The fingerprint is
  `ccg.ids.tuple_id` over the record's canonical JSON. P1-22 chooses the encoding and pins the literal.
- At answer time the engine computes the fingerprint of the record it is running and emits the version that maps
  to it. If no version maps to it, the engine raises before building an answer.
- The check therefore catches a change to the row table, and a change to an identifier. It cannot see a code change
  to the cost order, the fallback or `unexplored` that leaves the identifiers alone. Those are held by P1-22's and
  P1-23's pinned walk tests.
- A row added without a new table entry fails the check. This is P1-22 behaviour 8's test.

**Granularity.**
- P1-22 and P1-23 built the caller-only definition and the builder (P1-39b) published it as `1.1`. The V-both definition (P1-23d) is
  `1.2`, the only version the engine emits from then on; the caller-only table stays a non-shipped candidate table whose fingerprint is `1.1`'s.
- A definition change merged after `1.2` is published is `1.3`, and so on.
- `1.0` names the slice's walk, and the golden keeps it. It is not an engine table entry: no valid 1.1 answer can
  carry it (V17 requires a bound, and the slice's walk has none), and P1-22's equivalence harness compares item
  sets, not answers.

**The validator does not pin version values.** The table lives in the engine. A validator list would make every
definition change an edit to `ccg/answer.py`, which couples the contract code to the walk. The validator keeps
v1's pattern check.

**`function-level-continuation` retires in `1.1`.** §5's labelled rows emit `flow-unavailable:<lang>` in its
place. Its `ENGINE_REASON_CODES` entry and that entry's test are removed by P1-22, not by P1-45 (index C6; v1 §10
item 3).

## 2. `cacheBypass`

EP §10.4 ("Repeats are uncached by a bypass") sets the requirement and its planted inputs. This section supplies
the engine input that requirement names. The input's name, `cacheBypass`, is the value EP §12.3's AC-24
pre-registration records.

- **Input.** A plain-language ask takes `cacheBypass`, a boolean that defaults to `false`. It is surfaced as:
  - the plain-language front's ask entry point (P1-26);
  - an argument of the MCP `ask` tool, and the CLI flag `--cache-bypass` (P1-25).

  When `true`, the interpretation cache is **neither read nor written** for that ask. The cache is left exactly
  as it was, and the interpreter is called.

  A structured request runs no interpretation and has no `provenance`. On a structured request the input is
  accepted and has no effect, and nothing records it.
- **Recorded in `provenance`.** `Provenance` gains the optional field `cacheBypass`, boolean, `const: true`.
  - It is **present exactly when the bypass was engaged**, and absent otherwise.
  - `false` is not representable, so one fact has one encoding in the canonical bytes. This is the same choice v1
    made for `recipe.guards.visitedSeedsAnchor`.
  - `cacheBypass` present requires `cacheHit: false` (schema, §7.1).
- **Not a `cacheKey` input.** The bypass changes whether the cache is consulted, not what the interpreter returns.
  `cacheKey` keeps v1 §3's input list, and it is still computed and reported on a bypassed ask, so a harness can
  see that repeats share a key. A change to that input list would be major (v1 §9).
- **Outside the facts.** `provenance` is outside `facts_bytes` (v1 §6), so `cacheBypass` does not change any
  answer's facts.

## 3. Edge types, hop kinds, binding evidence and wiring items

**In a 1.1 document a hop may not arrive by `DEFINED_BY`, `LAUNCHES`, `DEPENDS_ON` or `FROM_DEP`.** The validator
lists the four in `PROHIBITED_EDGE_TYPES` (V19, §7.2) and checks it on `"1.1"` documents only, so no 1.0 document
changes validity. The schema's `edgeType` enum keeps its 21 names, because removing a value is major (v1 §9). It
gains `TO_UNIT`, and `Hop.kind` gains `Unit` (§3.5).

This leaves EP §5.2.1's class-X rows true as written. One consequence: its planted input (h), a `LAUNCHES` hop that
must be a harness error, can now only be built as a V19-invalid document. P1-32's harness test therefore builds it
without calling `validate`.

### 3.1 What lifts each prohibition

| Edge type | Why no P1 hop | Lifted by |
|---|---|---|
| `DEFINED_BY` | Its target, `SchemaEntry`, is not a hop kind and is not loaded in P1 (`-p0-graph.md` §1.1, §1.6) | The P3 successor spec that adds a `SchemaEntry` hop kind alongside the data rows |
| `DEPENDS_ON`, `FROM_DEP` | `Dependency` is not a hop kind. P1 traverses no dependency row (`docs/plans/connected-code-graph-p1-graph.md` §1) | The P3 successor spec that adds a `Dependency` hop kind and the dependency-bump row |
| `LAUNCHES` | Its target, `File` or `Unit`, is not a hop kind. The launch chain travels as evidence (§3.2) | A successor that adds a `File`/`Unit` hop kind or a compound row. The trigger is the first consumer that needs `LAUNCHES ←` as a step of an item's chain (FR-13 review, P3) |

Lifting a prohibition removes that edge type from `PROHIBITED_EDGE_TYPES` and adds the hop kind. Both are
additive: a minor bump, with EP §5.2.1's row reclassified by its owner.

### 3.2 `Bound`: the binding chain as evidence

Every binding P1 makes is a chain of `BOUND_BY` entries on the `Link`, in chain order: URL, env/Compose, and the
`.mcp.json` launch chain that AC-4 needs. `ord` 0 is the entry nearest the consumer. The graph contract is
`-p1-graph.md` §2 (index D10; P1-17). A `Link` hop's `bound[]` carries those entries, one `Bound` per entry, and
renderers list them by `ord` (P1-24).

`Bound` changes as follows (§7.1):
- **Exactly one of `keyPath` and `launch`.**
  - A `configEntry`-origin entry carries `keyPath`.
  - A `launchFact`-origin entry carries `launch`, a `$ref` to `Launch`, with v1 §7.1's per-kind rules. It has no
    `keyPath`, because the record has none (`-bundle-spec3.md` §1.6).
  - The assembler writes exactly one of the two (`-p1-graph.md` §1, the `launchFact` row), so the field present
    tells the origin.
  - A 1.0 `Bound` keeps `keyPath` (V17).
- **Port evidence: optional `port`, `portOrigin` and `packId`.**
  - These are the `BOUND_BY` edge's properties (`-p1-graph.md` §2, "Port evidence" and "Framework-default port").
  - `port` is an int from 1 to 65535.
  - `portOrigin` is one of `explicit`, `env-default`, `compose`, `pack-default`. It is present exactly when `port`
    is.
  - `packId` matches `^[a-z][a-z0-9-]{0,39}$`, the rule-pack id pattern. It is present exactly when `portOrigin`
    is `pack-default`.
  - Consequence for a renderer: a binding entry shows the matched port and where it came from ("port 8000, from the
    `fastapi` pack's default"). It never re-derives a port from `launch`, which holds the entry's own launch fact,
    not the port the binding matched.
  - These are engine-minted values (enum, pattern, integer), so `DATA_FIELDS` is unchanged.
- **`bound[]` is per site, and `Bound` has no `siteId`.** Every `BOUND_BY` edge carries the `siteId` of the
  site it binds, and `ord` counts per (link, site) (`-p1-graph.md` §1, the `BOUND_BY` row). The engine reads a
  `Link` hop's entries for exactly the site that hop is entered for (`bound_out`, `-p1-graph.md` §5): the
  following `CallSite` in impact, the preceding one in depends-on. So the per-site rule holds by construction of
  the read, as it does for `lifted[]`.

Not changed:
- The `ConfigEntry` **hop** keeps v1's `keyPath` requirement. No 1.1 row ends on, or leaves, a `launchFact`-origin
  entry: `LAUNCHES` is prohibited, and §3.3 lists only `configEntry`-origin entries.
- `AnchorConfig` and `ConfigSubject` keep `keyPath` required.
- F-3/F-4: V12 screens `Bound.launch.{mountPath, service, envVar}` as it screens a hop's `launch`, and screens
  `Bound.keyPath` only where it is present.
- HA-1 does not replay `bound[]` (EP §5.2.1). A wrong entry is caught by P1-17's linker tests and P1-41's fixture
  rows.

### 3.3 Wiring items: binding entries listed as `config` items

AC-2's wiring files must be listed, and EP's `wiring` stratum scores them only as items. Its dev gold is
`config`-subject items with a `keyPath`, and EP §3.5 leaves the choice to P1. **P1 lists them.**
- **Row.** At a `Link` hop the walk reaches by `TO ←` (impact, in `value` or `contract` state), each
  `configEntry`-origin `BOUND_BY` target becomes an item.
  - The edge is `BOUND_BY`, `dir` `->`, to-state `end`, distance increment 1 (a config hop, graph note §6.2's
    distance paragraph).
  - The hop's status and reason are the `Link`'s.
  - This is an impact-column row that the graph note's table lacks; it is part of 1.1's definition (§1.2).
- **Chain shape.** The chain is the `Link`'s chain, then one `ConfigEntry` hop, which is the last hop.
  - The `Link` hop of that chain carries **no `lifted`** (no site was entered).
  - Its `bound` holds **exactly the one entry** the item lists.
  - **Which edge supplies that `Bound`.** One entry may bind several sites, each through its own edge (with its
    own `ord` and port evidence). The item lists the entry **once**. Its `Bound` takes `ord`, `port`,
    `portOrigin` and `packId` from the entry's edge with the lowest `(siteId, ord)`. The rule is fixed so that the
    item's facts do not depend on read order (FR-19).
  - These entries come from the **per-link** read `bound_of_link` (`-p1-graph.md` §5), because the row enters no
    site. It returns every `BOUND_BY` target of the link, ordered so that the rule above is a first-row pick, and
    the engine keeps only the targets that have a `keyPath`.
  - The subject is `{type: config, entryId, path, line, keyPath}` from that entry.
  - `launchFact`-origin entries are not items; they stay `bound[]` evidence.
- **Validator (V5, 1.1).** A `Link` hop is followed either by its `CallSite` (`VIA <-`, v1) or, as the chain's
  last hop, by a `ConfigEntry` hop with `edgeType` `BOUND_BY`, `dir` `->`. In the second case:
  - its `bound` is exactly that entry;
  - it has no `lifted`.

  A 1.0 document has no such chain (V17).
- **Depends-on (V5, every label).** A `Link` hop is entered by `VIA ->` from its `CallSite` and left by `TO ->`
  or `TO_UNIT ->`. Its `lifted` names that preceding site, and its `bound` is that site's (§3.2).
- **Other documents.**
  - EP §3.5's "P1's choice, not yet made" should cite this section (`data-scientist`).
  - EP §5.2.1's `BOUND_BY` row (`Link → ConfigEntry`, class D) becomes replayable under EP's own rule once P1 loads
    the edge. No index text changes.

### 3.6 The `site` hop state: a `Link` a config entry entered

A `ConfigEntry` anchor enters each `Link` it binds by `BOUND_BY <-` in the walk state `site(siteId)` (P1-22), so `VIA <-` lists only
that site's `CallSite`. The hop **reports it as it is**: `state: "site"`, which v1 §5.2 defined as "the walk state's kind on arrival".
- **Why not `value`.** v1 §5.2 and V13 take `(kind, id, state, accessPath)` as the walk's visited-set key. Reporting `site` as `value`
  makes a chain that reaches the same `Link` once from the entry (`site`) and once by `TO <-` (`value`) a V13 cycle, though it is no cycle.
- **No evidence is lost, none is added.** `accessPath` stays absent (V3: present exactly when the state is `path`); the `siteId` is the
  following `CallSite` hop's `id` and the `Link` hop's `lifted[].siteId` (V5), so the state value carries no data field.
- **The enum change is additive**, a minor (v1 §9), and 1.1 is published (§1.2): `Hop.state` gains `site` (§7.1).
- **Shape (V5, 1.1).** On every document, a hop in state `site` is a `Link` hop entered by `BOUND_BY <-` from a `ConfigEntry` hop, and is
  followed by its `CallSite` hop (`VIA <-`), never by a wiring `ConfigEntry` hop and never last; no other hop has state `site`. On a `"1.1"`
  document the converse also holds: a `Link` hop entered by `BOUND_BY <-` from a `ConfigEntry` hop is in state `site`. A `"1.0"` document
  keeps the v1 state of that `Link` and has no `site` hop (V17).
- **Renderers** read `state` only for `path` and `DATA` (`ccg/render/common.py`), so a `site` hop renders as any `Link` hop (P1-24).

### 3.4 `MOUNTED_BY` evidence is not carried

- A mount reaches an answer only through the `Link`'s status and reason, which the linker derived using it.
- No hop carries the mount path or mount site.
- No P1 renderer or AC needs one. A later minor can add `Hop.mounts` on a provider `Symbol` hop.

### 3.5 `TO_UNIT`: crossing into a never-built unit (AC-9b, FR-7a)

A link into a never-built unit has no `TO`; it has a `TO_UNIT` edge to that `Unit` and is
`possible`/`target-not-mapped` (`-p1-graph.md` §1, the `TO_UNIT` row). The crossing is shown as an item.
- **Row (depends-on).** From a `Link` hop entered by `VIA ->`, `TO_UNIT ->` reaches the `Unit`.
  - To-state `end`, distance increment 0 (as `TO`).
  - Status `possible`, reason `target-not-mapped` (registry row on `Link (TO_UNIT)`).
  - In impact and contract mode the same link already appears as a `possible` `Link` hop, so no new row is needed
    there.
- **`Unit` hop.**
  - `Hop.kind` gains `Unit`, with `id` = the `unitId`, `repo` and `unit` (= `id`), and no `path` or `line`.
  - It has no per-kind fields.
  - It is always the chain's last hop.
- **Subject.** `Item.subject` gains `UnitSubject {type: "unit", unitId}`, so the item's subject id is the
  `unitId`. Item order (v1 §6) uses the unit's `rootPath` (from `staleness.units`) in the `subject.path` slot and
  0 in the line slot.
- **`staleness.units`.** The never-built unit **appears** there as `not-mapped`. V9's set equality includes it
  because it is a hop's unit, and V9 also requires a `Unit` hop's entry to be `not-mapped`.
- **Scoring.** EP's H1 has no unit branch, so such an item is never a hit. EP §5.2.1's replay table needs a
  `TO_UNIT` row (`(Link)-[:TO_UNIT]->(Unit)`, label `Unit.unitId`). That is owed by `data-scientist`.

## 4. Depth bound, `unexplored` and build times

### 4.1 The bound (index D3; carry-over C4)

- Every resolved 1.1 answer is bounded at `DEFAULT_DEPTH_BOUND` (`cpg/ccg/engine/config.py`, value 2), and
  `depthBound.bound` records it.
- `Request` gains **no** bound field in 1.1, so the default always applies. A request-side bound would be a new
  `Request` field, and it would have to enter EP §10.4's request equality. No P1 consumer needs one.
- V17 requires `bound` ≥ 2 on every resolved 1.1 answer (EP §15.2 Q1).
- Ambiguous and unresolved answers keep v1's `{"unexplored": 0}`, with no `bound` (V2).
- **The bounded result is everything reachable within the bound.** An item is listed when some chain whose distance
  never exceeds `bound` reaches it, on the best such chain by §1.2's cost order. A `possible` chain within the bound
  is therefore listed, and the item is `possible`, even when a `confirmed` chain reaches it only beyond the bound.
  The result depends on the snapshot and the bound alone, never on route length or arrival order.
- **How the walk keeps it.** A state is a visited-set key (`-p0-graph.md` §3.1), and the walk keeps every *arrival* at it that no other
  arrival dominates, where `a` dominates `b` when `a` has no more possible hops and no greater distance. A state is
  expanded once per kept arrival, and an equal pair is dominated, so a cycle ends. A rule that re-expands on the
  lexicographic `(possible hops, distance)` alone, or on distance alone, makes the result depend on arrival order
  under the bound. Each item's chain is the chain of its own arrival, so the chain's possible hops, distance and
  length recompute to the item's cost.

### 4.2 `unexplored`

`unexplored` is `|E_cut|`, the number of **distinct stored edges** the bound cut. A stored edge is identified by
its type and its two endpoint ids; a `CALLS_NAME` compound is identified by its `CallSite` and `Symbol` ends. An
edge `e` is in `E_cut` when all three of these hold:
1. Some arrival at a state, in the final set of non-dominated arrivals (§4.1), has distance equal to `bound`.
2. A row of the walk definition (§1.2) would traverse `e` from that state. The `DISMISSED` anti-join and §5's exclusions
   apply.
3. That row's distance increment is 1.

Consequences:
- An edge is counted once, however many states reach its source, and whether or not its target is reached within
  the bound some other way. This keeps the count independent of expansion order and deterministic for a fixed
  snapshot.
- An edge with increment 0 (`DATA`, `CONTROLLED_BY`, `HAS_*`) is still followed at distance = `bound`, so it is
  never cut.
- Edges §5 excludes, `DISMISSED` edges and edges no row traverses are not counted.
- The level guard and `RESULTSET_SIZE` raise, so they never produce a count.
- The rendering "reach stopped at distance d with k unexplored edges" (P1-24) reads `bound` and `unexplored`, as v1
  §4.4 says.

### 4.3 `builtAt` and `failedAt` are outside a 1.1 answer's facts

A unit's `builtAt` (and `failedAt` for `stale-kept`) is the build's wall-clock time (`-p1-graph.md` §1, the `Unit`
row). It is not part of the snapshot identity (`-p1-graph.md` §3). Under v1 §6 it sat inside `facts_bytes`. Two
builds of the same commits would then give one `snapshotId` with different facts, against FR-19 and against index
§8 and P1-39 behaviour 3.

**Decision.** For a document labelled `"1.1"`, `facts_bytes` also drops every `staleness.units[].builtAt` and
`failedAt`. The fields stay in the answer and in its canonical bytes, so AC-9's "states when the view was last
refreshed" is unchanged. A `"1.0"` document's `facts_bytes` is v1's.
- **Why not the alternatives.**
  - Defining `builtAt` as the commit time would no longer say when the view was refreshed (AC-9).
  - Putting it in the identity would make every rebuild a new snapshot, against index §8 ("identical snapshots minus
    timestamps").
- **FR-19's reading is unchanged.** Its facts are "the listed impacts, causes, gaps, findings", and a build time is
  none of them. So this is not a stakeholder question.
- **Versioning.** v1 §9 lists a changed `facts_bytes` boundary as major. Scoping the change to the new label keeps
  every stored 1.0 comparison as it was, which is the property §9's rule protects. No 1.1 `facts_bytes` value
  exists yet. This is a decided exception, and 1.1 stays a minor.

## 5. The flow fallback: which hops a flag-false unit makes `possible`

**Stakeholder decisions.**
- **2026-11-09.** Only the flow-dependent call-continuation hops become `possible`. Items
reached by resolved edges stay `confirmed` when no flow-dependent row is on their chain. This is the method
owner's recommendation (`docs/plans/connected-code-graph-p1-ml.md` §6.5). It replaces v1 §5.2's "first hop that
crosses into the unit" sentence.
- **2026-11-19.** The shipped table is **V-both** (`docs/plans/connected-code-graph-p1-ml.md` §3.5's pre-registered rule picked it,
  `docs/plans/connected-code-graph-p1-flow-fallback-result-ml.md` §2). V-flow and V-union do not ship. No walk lists every dev flow item,
  so FR-8 and AC-25's control half are recorded **not met** for flow-dependent impact in a flag-false language, and the answer says so (§5.1).

The one definition P1-23 and P1-23d implement is this section.

**Flag-false unit.**
- A unit is flag-false when its `Unit.flowAvailable` is `false`. The assembler always writes the flag
  (`-p1-graph.md` §1, the `Unit.flowAvailable` row); an absent flag reads as false only as a backstop.
- The flag is set per language (index D2).

**Rows not followed inside a flag-false unit, in any column:**
- `DATA` and `CONTROLLED_BY`;
- the `arg(i)`/`Ret` rows: `CALLS →` into a callee, the index-matched `HAS_PARAM →`, and the `Ret` summary.

No hop whose unit is flag-false has state `path`.
- A call site that a resolved `CALLS ←`, `CALLS_NAME ←` or `VIA ←` reaches is state `path r` in a flag-true unit and
  `value` in a flag-false one (never `path`), so the rule above holds and V3 (`accessPath` only for `path`) is unaffected.
- The argument rows (`arg(i)`) `CALLS →`/`CALLS_NAME →` into a callee are followed only when both the call site's unit and
  the callee's unit are flag-true: the callee's flag is what keeps the callee out of `path`; the caller's matters only for an
  anchor call site in `path` state.

**The closed list of labelled rows.** The walk definition (§1.2) has one state for a function whose callers it
reaches only at call level: `function`. The rows that enter it, and the rows labelled when leaving it, are these
and no others. Both kinds apply only inside a flag-false unit, because a flag-true unit uses the precise rows.

| Column | Rows that enter `function` (to a `Symbol`) | Labelled rows (leaving `function`) |
|---|---|---|
| impact | `HAS_CALL ←`, `HAS_PARAM ←`, `HAS_COND ←` from one of the function's points, outside contract mode (where `HAS_CALL ←` ends in `end`) | `CALLS ←`, `CALLS_NAME ←`, `TO ←`, and `HAS_CALL →` (§5's callee run) |
| depends-on | `CALLS →`, `CALLS_NAME →` into a callee | `HAS_CALL →` |

The unit that gates a depends-on `function` entry is the arriving callee's, the unit of the `function`-state hop it leaves.

**The callee run (impact column, V-both).** From a `function`-state hop at distance `d` the walk also lists the function's **callees**, to the bound:
- `HAS_CALL →` (increment 0) to the function's call sites. The hop is state `end`, labelled, and is not an item.
- `CALLS →` or `CALLS_NAME →` (increment 1) from such a call site to the callee `Symbol`: state `end`, labelled, an item of kind `symbol`. The callee is listed
  while `d` plus its call depth is at most the bound.
- From a callee so reached, the same two steps again (`HAS_CALL →`, then `CALLS →`/`CALLS_NAME →`), still to the bound.

A **callee run** is the maximal chain suffix those rows make. It never re-enters `function`, and the walk never takes a callee's own callers. A function that owns an
affected point in a flag-true unit (state `end`) and an anchor `Symbol` never open a run, because only a `function`-state hop does. The run is not flag-gated: a callee
shares the language, hence the flag, of the function that opened it (`CALLS` stays within a language; `Unit.flowAvailable` is per language). The label's `<lang>` is,
as for every labelled hop, the `lang` of hop *i*−1's unit.

The following rows do not enter the state, so their continuations are not labelled:
- `HAS_RETURN ←`, which enters `value`: a changed return reaches the callers through a resolved call;
- an anchor `Symbol`: a changed function changes its direct callers;
- `PROVIDES ←`, `TO ←` in contract mode, `VIA ←`, and §3.3's wiring row.

A node with no unit, a gated point with no owner, or a unit with no `Unit` row is a data error, fail-closed
(`GraphDataError`; no label or gate is guessed). So is an anchor absent from the snapshot, which P1-39 reports as a build
error and never as an empty (no-impact) answer.

The state is part of the visited key (`-p0-graph.md` §3.1), so a hop's label depends only on its row and the unit
flag. It never depends on the path taken to reach the hop.

**Labelling.**
- A hop on a labelled row is `possible`.
- If its own edge is `confirmed`, its reason is `flow-unavailable:<lang>`, where `<lang>` is the `lang` (in
  `staleness.units`) of the unit of the `function`-state hop it leaves.
- If its own edge is already `possible` (`name-match`, `resolution-unverified`, a `possible` link), it keeps its own
  reason.
- Every other hop keeps its own status and reason. Item status and reason follow v1 §5.1 and V7.
- **A hop's own edge status can therefore be read from the answer.** It is `confirmed` exactly when the hop is
  `confirmed` or its reason is `flow-unavailable:*`. P1-37 reports noise and wrong-possible split by it (P1-36, ML
  note §6.5).
- **Cost order.** After `(possible hops, distance, length)`, ties go to the chain with fewer hops whose own edge is
  `possible`, then to the stable key. So when a caller is reached through two sites, a resolved call and a
  name-matched one, both on labelled rows, the resolved chain wins and the item's reason is
  `flow-unavailable:<lang>`. P1-23's test of this plants the name-matched site so that it **sorts first by
  stable key**. Otherwise the stable key alone picks the resolved site, and the test cannot see the tie-break
  dropped (the mutant it must kill).

**Consequences in P1 (every flag expected false, index §5).**
- These stay `confirmed`:
  - AC-1's item, because no labelled row is on its chain;
  - `apiFetch`, the direct caller of an anchor `Symbol`;
  - an enclosing function of an anchor point;
  - contract consumers' functions;
  - wiring items.
- Callers of a function that holds a changed point are `possible`, `flow-unavailable:<lang>`, and so are their
  callers in turn. So are its callees, and their callees, to the bound (the callee run).
- AC-6's planted kinds keep their edge-level reasons.
- FR-8 and AC-25's control half are **not met** for flow-dependent impact in a flag-false language: the guarded point is not asserted `possible`, and every such
  answer carries the notice of §5.1.
- AC-5's confirmed-precision bar has a population to measure.
- **For P1-22.** Its row and equivalence fixtures set `Unit.flowAvailable` explicitly. The flag is `true` for the
  `arg(i)`/`Ret`/`CONTROLLED_BY` rows, so P1-23's flag gate changes none of those pins. The slice-equivalence item
  sets do not depend on labels.

**What the validator checks (V18, 1.1 only).** The column is read from `interpretation.request.direction`. For every
item and member hop *i* ≥ 1:
- (a) a hop in state `function` is a `Symbol` hop that arrived by an entry row of its column;
- (b) a labelled-row hop leaving a `function`-state hop is `possible`;
- (c) `flow-unavailable:*` appears only on a labelled row leaving a `function`-state hop, and never on
  `CALLS_NAME` (whose own edge is always `possible`);
- (d) its `<lang>` equals the `lang` of hop *i*−1's unit;
- (e) a hop whose predecessor is in state `function` arrives by a labelled row of its column (impact: the four of the table above). This closes the list
  of exits, so no unlisted continuation can yield a `confirmed` item.

The callee run is a **`"1.2"` rule**: under `"1.1"`, `FLOW_ROWS["impact"]` is unchanged and a `HAS_CALL →` exit from `function` is rejected by (e), so the published label's valid set does not widen.
A hop is **in a callee run** (impact column only) when it is a `HAS_CALL →` hop whose predecessor is in state `function` or is a callee `Symbol` hop of a run, or a
`CALLS →`/`CALLS_NAME →` hop whose predecessor is a `HAS_CALL →` hop of a run. (a)–(d) read "leaving a `function`-state hop" as "leaving a `function`-state hop or being in a
callee run", and add:
- (f) after a callee `Symbol` hop of a run the only continuation is `HAS_CALL →` (also in the run); after a `HAS_CALL →` hop of a run the only continuation is `CALLS →` or
  `CALLS_NAME →`. This closes the run's exits as (e) closes `function`'s.
- (g) a callee run occurs only in the impact column, and every item it ends in has a `symbol` subject.

The answer does not carry the flag, so whether `function` occurs only in flag-false units is P1-23's test, not the
validator's.

### 5.1 The flow notice: `notices`

No walk lists every flow-dependent impact in a flag-false language (§5, 2026-11-19), so the answer says what it does not enumerate. The statement is part of the
document, not of a renderer, so a JSON client sees it and a renderer cannot forget it.

**Shape (schema `"1.2"`, §7.1).** The root gains the optional `notices`, an array of `Notice {code, langs}`.
- `code` is an engine-minted enum with one value in P1, `flow-not-enumerated`. It is not a `reasons.yaml` reason code (no item carries it).
- `langs` is a non-empty array of `lang` values (the pattern of `Unit.lang`), without duplicates.
- Absent exactly when there is nothing to state: `[]` is not representable (`minItems` 1), as `recipe.guards.visitedSeedsAnchor` has one encoding. At most one `Notice` per `code`.
- Canonical order: `notices` by `code`, `langs` by code point (`ARRAY_RULES`).
- `notices` is in `facts_bytes`: it is a fact about the answer, determined by the snapshot, so AC-16's equality across clients covers it. It carries no wording and no
  figure, so it adds no `DATA_FIELDS` entry.

**Trigger (the engine's, P1-39d).** A resolved answer, either column, carries `flow-not-enumerated` with `langs` = the set of `Unit.language` values of the **flag-false units that
host a code node the walk visited**: the visited set (`bfs.Walk.visited`), the anchor included, restricted to `Symbol`, `CallSite` and point nodes (`Param`, `Ret`, `Cond`, …;
a point's unit is its owner's). It does not depend on the items: an answer that lists nothing for a flag-false anchor carries it, because that is where silence misleads most.
- `Link`, `ConfigEntry`, `ContractPoint` and `Unit` nodes do not trigger it. A config anchor with a binding does reach a code node (`bound_in` then `via_in:site` visit the bound `CallSite` within the minimum bound), so
  only an unbound config anchor or a contract-point anchor yields no code node, hence no notice.
- An ambiguous or unresolved answer has no walk and no notice.
- **Missing data fails closed (decision).** A visited code node with no unit, or whose unit has no `Unit` row, is a data error (`GraphDataError`), as §5 says for a gated row, so the omission is never silently unstated. A unit's
  language is required only when its flag is false (a flag-true unit with no language is not an error). The extra `units` read adds a `recipe.queries` name to answers that did not read units before (P1-39d updates those pins).
- A flag-true language is never in `langs`.

**Validator (V20, `"1.2"` only, §7.2).** The answer does not carry the flag, so the validator checks what the answer does carry (the posture of V18's last paragraph):
- (a) `notices` appears only on a `"1.2"` document and only when `interpretation.status` is `resolved`;
- (b) one `Notice` per `code`;
- (c) **coverage:** every `<lang>` of a `flow-unavailable:<lang>` reason on any hop or item is in `langs` of a `flow-not-enumerated` notice. A labelled hop proves its unit is
  flag-false, so no answer can carry a labelled hop and omit the notice.

The converse (a notice with no labelled hop, as in an answer that lists nothing) is valid, and the exact trigger is P1-39d's engine test.

**Wording (renderers, P1-24b).** The template lives in `ccg.render.common` and is deterministic text; no model writes it; it reads the same for both columns. V-both lists some flow-dependent items, so the text states a partial list, never an empty one. It must:
- name every language of `langs`, and say that flow-dependent effects there (those that travel through data values, return values, arguments or control conditions) **may be missing from this list**;
- say that in those languages the walk did not follow data, return-value, argument or control-flow edges (call, link and wiring edges were followed), and that an item's absence is not evidence that no such effect exists;
- not say or imply that nothing was listed (no `none`, no `nothing`);
- never match, case-insensitively and as whole words, `safe`, `complete`, `exhaustive`, `full`, `guarantee`, `guaranteed`, or the phrases `no impact`, `all impacts` (`incomplete` and `unsafe` are therefore allowed words; the test runs on the rendered text with `langs` substituted),
  contain no digit outside a language name, and give no rate or count: the measured share is development-set, in-sample, and lives only in the documents that cite it.
Tests render one impact and one depends-on document and assert each statement.

**Where it is shown.** Directly after the counts line and before any item, in every output that shows items, including when there are none:
- `ccg.render.text.render_text` and `ccg.render.markdown.render_markdown`, through a new `ccg.render.common.flow_notice_lines(doc)`;
- `ccg.render.summary_input`: one line `notice flow-not-enumerated: <langs>` (codes only), and the plain-language summary (P1-26) is always followed by the renderer's notice text, not by
  a model's restatement;
- the CLI and MCP `ask` (P1-25) return the canonical document unchanged, so the field is in the JSON; a client that prints text prints `render_text`. No client strips, reorders or
  summarises `notices` away. A notice does not change `--fail-on` exit codes (§9, Q-1).

## 6. v1 text this document replaces

The S50 gate flagged stale v1 text for P1 start (`docs/reviews/connected-code-graph-p0-impl.md`, Pass 9, item 5).
EP v1.4's obligations add more (N-3). This list also covers v1 wording about the engine's reads and hops that
predates graph model v1. Passages not listed stand.

| v1 passage | Now |
|---|---|
| §1 table, row "Function-level continuation is `possible` / `function-level-continuation`" | In 1.1 the same continuation rows are §5's labelled rows, with reason `flow-unavailable:<lang>` and only in flag-false units. `viaData` stands |
| §2 table, `answerSchemaVersion` `"1.0"`; §9 "v1 is `"1.0"`" | `"1.2"` is current; `"1.0"` and `"1.1"` are still accepted (§1.1) |
| §3, bullet "How AC-24 reads `status`" | Replaced by a citation of EP §10.4, the predicate's home (N-3) |
| §3 `provenance` table | Gains `cacheBypass` (§2) |
| §4.1, "The slice's walk (`impact.py`) is `"1.0"`; P1's added rows … bump it (carry-over C6)" | §1.2 |
| §4.1, "…the P1 engine's contract, not something the slice's readers can yet deliver: its reads do not return `Param`/`Ret`/`Cond` `path`/`repo`/`unit` …" | The contract stands. Graph model v1 stores `repoId`/`unitId`/`path` on `Symbol` and `CallSite` only (graph note §6.3), so a point hop's `repo`, `unit` and `path` are its owner's, read by the owner join (`-p1-graph.md` §5). That join, `LIFTED_VIA`'s `unit`/`repo`/`siteId` and the edge `reason` are P1-21's reads (C3). The golden's `recipe.queries` remains a hand-built fixture |
| §4.3, `builtAt`/`failedAt` as facts; §6 "`stamp` and `staleness` are inputs read from the store (`builtAt` is the unit's, not a clock)" | For 1.1, §4.3, a stated exception to v1 §9's major rule |
| §4.4, "P1's default bound must be >= 2, … recorded by S50 …; carry-over C4" and "counting `unexplored` is a P1 engine obligation [frozen, untested]" | §4.1, §4.2 (index D3) |
| §5.2, `state` bullet: "`function` … retires with the `arg(i)` rows"; its list `value`, `path`, `function`, `contract`, `key`, `end` | `function` stays, entered only in flag-false units by §5's entry rows; the list gains `site` (§3.6) |
| §5.2, "A `reason` of the form `flow-unavailable:<lang>` … is carried by the first hop that crosses into the unit …" | §5 |
| §5.2, `Hop.kind` list and `edgeType` vocabulary: "closed in v1; a new edge type is additive" | `Unit` and `TO_UNIT` are added (§3.5). Four names are prohibited on 1.1 hops (§3) |
| §5.1 `subject`; §6 `items` order key | `UnitSubject` is added, with its order key (§3.5) |
| §5.2, `Bound` with `keyPath` required; "A `Link` hop is followed by the consumer `CallSite` hop" (V5) | §3.2, §3.3; the depends-on shape is in §3.3 |
| §7.4 (3), "`bfs-v1` does not follow `CONTROLLED_BY` at all" | True of `1.0`. `1.1` follows it in flag-true units (P1-22 behaviour 3, with the C1 cycle test) and not in flag-false units (§5) |
| §5.4, `other-possible-callees`: "the unresolved other candidates of one resolved call site" | Stands, and is read narrowly: a candidate is an item whose **own arrival** is the candidate hop, i.e. the item's first `possible` hop is `CALLS_NAME`/`resolution-unverified` and nothing follows it but the arrival itself (depends-on: the hop is the chain's last; impact: the hop is followed only by the `HAS_CALL <-` owner hop). In the impact column a candidate reached through an argument row (`CALLS_NAME ->`, then `HAS_PARAM ->`, then the owner's `HAS_PARAM <-`) is therefore not an own arrival: it stays in `items`, `possible`/`resolution-unverified` (fail-open, nothing dropped), while the same item folds in the depends-on column. An item further along a chain through a candidate (a caller of the candidate, the candidate's own callee) is not a candidate of that site: it is never a member, stays in `items`, and is `possible`/`resolution-unverified` by V7 (first possible hop). The fold is a display grouping decided per language by P1-39 (`docs/plans/connected-code-graph-p1-answers.md` P1-39 (5)) |
| §4.2 "Source" ("`:Snapshot` … has no verdict-log properties … the engine mints `0` and the constant"); §6 "the engine mints the verdict-log pair until P4"; carry-over C8 | `:Snapshot` stores `verdictLogOffset` `0` and `verdictLogHash` `EMPTY_VERDICT_LOG_HASH` from P1 (`-p1-graph.md` §3; the assembler writes them). The engine reads them and refuses a snapshot whose stored pair differs, or lacks it, before P4 (`ccg.engine.staleness.stamp`); it mints nothing. `snapshotId` is read from the node as before. C8 narrows to P4's chaining rule for offset > 0 |
| Carry-over C4, C6 | C6 is closed here. C4's rule is §4, and its implementation is P1-22's |

## 7. Schema and validator changes

P1-45 made these edits, and only these, except the rows marked **P1-45b** (the notice and the 1.1-or-later helper, §5.1: `tdd-engineer`, before P1-23d) and **P1-23d** (the V-both rows of V18).
`cpg/ccg/registry/reasons.yaml` and the golden are **not** edited.
`ENGINE_REASON_CODES`, `DATA_FIELDS`, `ARRAY_RULES`, `IDENTITY_KEYS`, `cache_key` and `sampling_hash` are unchanged.

### 7.1 `cpg/ccg/schemas/answer-v1.schema.json`

The file and its `$id` `urn:ccg:answer-v1` stay the same, because 1.1 is a minor of v1.

| Where | Edit |
|---|---|
| root `answerSchemaVersion` | `const: "1.0"` becomes `enum: ["1.0", "1.1", "1.2"]` (**P1-45b** adds `"1.2"`) |
| root `notices` (**P1-45b**) | optional property `{type: array, minItems: 1, items: {$ref: "#/$defs/Notice"}}`; not in `required` |
| `$defs.Notice` (**P1-45b**, new) | `{code: {type: string, enum: ["flow-not-enumerated"]}, langs: {type: array, minItems: 1, uniqueItems: true, items: {type: string, pattern: "^[a-z][a-z0-9]*$"}}}`, both required, closed |
| `$defs.Provenance` | add property `cacheBypass: {type: boolean, const: true}`, not required; add to its `allOf` an `if {required: [cacheBypass]} then {properties: {cacheHit: {const: false}}}` |
| `$defs.Bound` | add properties `launch: {$ref: "#/$defs/Launch"}`, `port: {type: integer, minimum: 1, maximum: 65535}`, `portOrigin: {type: string, enum: [explicit, env-default, compose, pack-default]}`, `packId: {type: string, pattern: "^[a-z][a-z0-9-]{0,39}$"}`. Remove `keyPath` from `required`. Add `oneOf: [{required: [keyPath]}, {required: [launch]}]`. Add `allOf`: `if {required: [port]} then {required: [portOrigin]}`; `if {required: [portOrigin]} then {required: [port]}`; `if {properties: {portOrigin: {const: pack-default}}, required: [portOrigin]} then {required: [packId]} else {not: {required: [packId]}}` |
| `Hop.kind` enum | add `Unit` |
| `Hop.edgeType` enum | add `TO_UNIT` (22 names); none removed |
| `Hop.state` enum | add `site` (§3.6) |
| `$defs.UnitSubject` (new) | `{type: {const: "unit"}, unitId: {pattern: 32-hex}}`, both required, closed; added to `Item.subject`'s `oneOf` |
| root `description` | cite both documents and rules V1–V20; `answerSchemaVersion` "1.0", "1.1" or "1.2" |

### 7.2 `cpg/ccg/answer.py`

| Rule | Change |
|---|---|
| V5 | The column is read from `request.direction`. **Impact:** a `Link` hop is followed by its `CallSite` (`VIA <-`), or, last, by a `BOUND_BY ->` `ConfigEntry` hop, in which case it carries exactly that entry in `bound` and no `lifted` (§3.3). **Depends-on:** a `Link` hop is entered by `VIA ->` from its `CallSite` and left by `TO ->` or `TO_UNIT ->`. In every case, `lifted[].siteId` names the link's site. A `Unit` hop is the last hop, entered from a `Link` by `TO_UNIT ->`, with `unit == id`; `TO_UNIT` arrives only at a `Unit` (§3.5). **`site` state (§3.6):** on every document, a hop in state `site` is a `Link` hop entered by `BOUND_BY <-` from a `ConfigEntry` hop, and is followed by its `CallSite` hop (`VIA <-`), never by a wiring `ConfigEntry` hop and never last. On a `"1.1"` document the converse also holds: a `Link` hop entered by `BOUND_BY <-` from a `ConfigEntry` hop is in state `site`. A `"1.0"` document keeps the v1 state of that `Link` (V17). `_check_hops` gains a `direction` argument |
| V3, V4, order | `KIND_FIELDS["Unit"]` has no fields. A `Unit` hop needs `repo`/`unit` but not `path`/`line`. `SUBJECT_ID["unit"] = "unitId"`. `_item_key` uses the subject unit's `rootPath` and 0 for a `unit` subject |
| V9 | A `Unit` hop's `staleness.units` entry has `buildStatus` `not-mapped` |
| V12 | `_check_f4` screens `Bound.keyPath` only when present, and screens `Bound.launch.{mountPath, service, envVar}` |
| `facts_bytes` | For `"1.1"` and `"1.2"` (**P1-45b**), also drops `staleness.units[].builtAt` and `failedAt` (§4.3); `"1.0"` is unchanged. `notices` stays |
| **V17** (new) | **Version labels.** A `"1.0"` document carries no `provenance.cacheBypass`, no `Bound` without `keyPath`, no `Bound.{launch, port, portOrigin, packId}`, no `BOUND_BY ->` `ConfigEntry` hop, no hop in state `site`, and no `Unit` hop or `unit` subject. A resolved `"1.1"` or `"1.2"` document has `depthBound.bound` ≥ 2 (**P1-45b**). A `"1.0"` or `"1.1"` document carries no `notices` (**P1-45b**). The 1.1-or-later tests (here, V18, V19, `facts_bytes`, the `_check_hops` flag) use one helper over `ANSWER_SCHEMA_VERSIONS`, not `== "1.1"` |
| **V18** (new, 1.1) | §5 (a)–(g), with `FUNCTION_ENTRY` and `FLOW_ROWS` as module constants keyed by direction; **P1-23d** adds `("HAS_CALL", "->")` to `FLOW_ROWS["impact"]`, a `CALLEE_ROWS` constant (`HAS_CALL ->` into a `CallSite`, `CALLS ->`/`CALLS_NAME ->` into a `Symbol`) for the callee run, and clauses (b)–(d) in a run, (f), (g). The run's rows apply to `"1.2"` only; (a)–(e) as before apply to `"1.1"` and `"1.2"`, with `FLOW_ROWS` unchanged for `"1.1"` |
| **V19** (new, 1.1) | No hop has `edgeType` in `PROHIBITED_EDGE_TYPES = frozenset({"DEFINED_BY", "LAUNCHES", "DEPENDS_ON", "FROM_DEP"})` |
| **V20** (new, 1.2; **P1-45b**) | §5.1 (a)–(c): `notices` only on a resolved `"1.2"` document, one `Notice` per `code`, and every `<lang>` of a `flow-unavailable:<lang>` reason is in the `langs` of a `flow-not-enumerated` notice |
| ordering (**P1-45b**) | `ARRAY_RULES` gains `"notices": "code"` and `"Notice.langs": "code point"`; `canonicalize` sorts both |
| public names | add `ANSWER_SCHEMA_VERSIONS = ("1.0", "1.1", "1.2")` (**P1-45b**), `PROHIBITED_EDGE_TYPES`, `FUNCTION_ENTRY`, `FLOW_ROWS`, `CALLEE_ROWS` (**P1-23d**); the `validate` docstring lists V1–V13 and V15–V20 |

V17–V20 report their own rule ids and run over `items` and every `groups[].members`.

## 8. For P1-45 (`tdd-engineer`): test order

All tests are in `cpg/tests/test_answer.py`, run from `cpg/` (v1 §11). As in v1, each rejection test first asserts
that its base validates.

**The existing tests that change**, all expected-value tables:
- `REQUIRED`:
  - `"Bound"` loses `keyPath`;
  - add `"Bound.keyPath": "keyPath"` and `"Bound.launch": "launch"`, the two `oneOf` lists the count walker counts;
  - add `"UnitSubject": "type unitId"`;
  - `INSTANCES` gains the three. `Bound.launch`'s instance is a 1.1 copy of `b_config()` whose first `Bound`
    holds `launch` only and whose `Link` is in state `site` (§3.6), and `UnitSubject`'s is `B-unit`.
- `test_every_object_of_the_schema_is_closed_and_it_names_its_spec`: its `$defs` name set gains `UnitSubject`.
- `ENUMS`: `("Hop", "kind")` gains `Unit`, `("Hop", "edgeType")` gains `TO_UNIT`, and `("Hop", "state")` gains `site` (§3.6; delivered after P1-45, as P1-23's behaviour 0).

Every other existing test stays as it is, including:
- the 1.0 golden-based `flow-unavailable:python` acceptance;
- `test_port_bounds_and_the_allowed_extras_of_a_launch_hop`'s `V3.*keyPath`;
- the `staleness builtAt` case of `FACT_MUTATIONS`. All three are 1.0 documents.

**New bases.**
- `B-1.1`: the golden relabelled `"1.1"`, with `depthBound` `{bound: 2, unexplored: 0}`.
- `B-prov-1.1`: `b_prov()` relabelled the same way.
- `B-chain`: `B-1.1` whose `Link` hop carries three `Bound` entries in an added config unit:
  - `.mcp.json` with `launch {command, argv0: "cypher-mcp/docker-run.sh"}`;
  - `docker-run.sh` with `launch {command, argv0: "docker"}`;
  - a Compose entry with `keyPath` `services.server.ports[0]`, `port` 8000 and `portOrigin` `compose`.
- `B-wire`: `B-chain` plus a wiring item. Its chain is hops 0–2 of the golden; the `Link` copy carries no `lifted`
  and only the Compose entry. Then comes a `ConfigEntry` hop (`BOUND_BY ->`, `end`). The item is `confirmed`, at
  distance 1.
- `B-flow`: a 1.1 impact document in `U1` (python). The sites have `state="value"`.
  - Item A is `SA` → `CALLS ←` site → `HAS_CALL ←` `fb` in `function`. It is `confirmed`, at distance 1.
  - Item B continues A's chain: → `CALLS ←` site, `possible`/`flow-unavailable:python` → `HAS_CALL ←` `fc`. It is
    at distance 2.
  - Item C is `SA` → `CALLS ←` site → `HAS_CALL ←` `fb` in `function` → `TO ←` `Link`,
    `possible`/`flow-unavailable:python` → `VIA ←` site in a javascript unit `U2` → `HAS_CALL ←` `fw`. It is at
    distance 2. The labelled hop has no `unit`, which is what lets (d)'s mutant be seen.
  - Build each chain from fresh hop dicts. Shared dicts alias under `deepcopy`.
- `B-dep`: a 1.0 depends-on document. Its chain is `SA` → `HAS_CALL ->` site (`value`) → `VIA ->` `Link`
  (`candidateCount` 1, one `lifted` entry naming that site) → `TO ->` provider `Symbol` (`end`). The item is
  `confirmed`, at distance 1.
- `B-unit`: a 1.1 depends-on document.
  - Its chain is `SA` → `HAS_CALL ->` site → `VIA ->` `Link` (`possible`/`target-not-mapped`, `candidateCount`
    1) → `TO_UNIT ->` `Unit` hop of a `not-mapped` unit `U9`.
  - The subject is `{type: unit, unitId: U9}`, at distance 1, with `staleness.units` `[U1, U9]`.

**Order, red to green.** Each case names its test, and the P1-45 behaviour it covers is in brackets.
1. `test_bases_validate` and `test_answer_schema_versions`: `B-1.1` validates. `"1.3"` is rejected (`schema`).
   The golden file round-trips byte for byte [5].
2. `test_cache_bypass` [1]:
   - accepted: on `B-prov-1.1`;
   - rejected (`schema`): `false`; `true` with `cacheHit: true`; the misspelt `cacheBypas`;
   - rejected (`V17`): on 1.0 `b_prov()`;
   - `facts_bytes` is equal and `dumps` differs with and without the field.
3. `test_v19_prohibited_edge_types` (parametrized over the four) and `test_v19_controls` [2]:
   - each of the four types on `B-1.1` hop 1 is rejected;
   - the same mutation on the 1.0 golden is accepted;
   - `BOUND_BY` and `RAN` are accepted;
   - mutant to kill: an empty `PROHIBITED_EDGE_TYPES`.
4. `test_bound_shape_and_port_evidence` [2]:
   - `B-chain` validates;
   - rejected (`schema`): `keyPath` added to a `launch` entry; `launch` removed from it; `launch.args`; `port`
     without `portOrigin`; `packId` with `compose`; `pack-default` without `packId`;
   - accepted: `pack-default` with `packId: "fastapi"`;
   - rejected (`V12`): an unmasked 25-letter `launch.service`;
   - rejected (`V17`): `B-chain` relabelled `"1.0"`;
   - mutant to kill: V12 reading `b["keyPath"]` unconditionally (`KeyError`).
5. `test_wiring_items` and `test_depends_on_links` [2]:
   - `B-wire` validates;
   - rejected (`V5`): the wiring `Link` hop carrying a second `bound` entry; an entry hop id not in its `bound`;
   - rejected (`V17`): `B-wire` relabelled `"1.0"`;
   - `B-dep` validates on both labels (with `bound` 2 at 1.1);
   - rejected (`V5`, entry): the `Link` hop entered by `TO` instead of `VIA ->`, with `candidateCount` removed so
     that V3 does not fire first; the `Link` hop preceded by a `Symbol` instead of its `CallSite`;
   - rejected (`V5`, exit): the hop after the `Link` changed to an impact-shaped `VIA <-` `CallSite`;
   - rejected (`V5`, `lifted`): a `lifted` entry naming another site;
   - mutants to kill: the depends-on entry check dropped; the exit check dropped.
6. `test_depth_bound_1_1` and `test_build_times_outside_1_1_facts` [C4, §4.3]:
   - rejected (`V17`): `B-1.1` without `bound`; with `bound: 1`;
   - a 1.1 ambiguous answer and the 1.0 golden need none;
   - on `B-1.1`, changing a unit's `builtAt` leaves `facts_bytes` equal and changes `dumps`;
   - on the golden, the same change changes `facts_bytes`.
7. `test_v18_labelled_rows`, `test_v18_labelled_to_names_the_departing_unit` and
   `test_v18_function_state_exits` [§5]:
   - `B-flow` validates;
   - rejected (c): `flow-unavailable` on A's hop 1, which leaves `SA` in `value`;
   - rejected (d): B's reason `flow-unavailable:javascript`;
   - rejected (b): B's labelled hop made `confirmed`;
   - rejected (a): a `CALLS ←` hop given state `function`;
   - rejected (c): on `B-1.1`, `flow-unavailable:python` on hop 1 (`PROVIDES ←`);
   - rejected (d): item C's reason `flow-unavailable:javascript`;
   - rejected (e): B's hop 3 changed to `PROVIDES ->` `confirmed`, with the item `confirmed`;
   - accepted: 1.1 `b_possible()` with `bound` 3;
   - mutants to kill: (d) reading the hop's own unit instead of hop *i*−1's (item C kills it); (b) dropped; (e)
     dropped; the direction lookup fixed to impact.
8. `test_to_unit_items` [§3.5]:
   - `B-unit` validates;
   - rejected (`V9`): `U9` made `fresh`;
   - rejected (`V5`): the `Unit` hop's edge set to `TO`; its `unit` differing from its `id`; the request direction
     flipped to impact;
   - rejected (`V17`): `B-unit` relabelled `"1.0"`;
   - mutant to kill: V9's `not-mapped` check dropped.
9. `test_algorithm_version_pin` [3]: `"1.1"` is accepted on `B-1.1`; `"1.1.0"` is rejected. This is a characterization
   test, green on arrival.
10. `test_engine_reason_codes_unchanged`: `ENGINE_REASON_CODES == {"function-level-continuation": "possible"}` [4]. This pin is P1-22's to edit.
11. `unclassified_strings`, `derived_data_fields == DATA_FIELDS` and the array probe pass with unchanged expected
    sets.

The mutation check (index §8) covers:
- V5's wiring, depends-on and `Unit` clauses;
- V9's `not-mapped` check;
- V17's two directions;
- V18 (a)–(e);
- V19;
- the 1.1 `facts_bytes` projection.

**Answer-builder staleness tests.** One answer-level test per source of a `staleness.units[]` entry, each with that unit on
no other path and each killed by dropping its source from the builder's input to `staleness_block`: (1) an item's own
unit; (2) a hop's unit on a chain (a hop in a unit no item is in); (3) a `lifted` unit; (4) a `bound` unit; (5) a unit
reached only by a folded group member (so it is absent from `Listing.items`). A source that is missed fails V9 loudly, but
only these tests say which source. A further test covers the N1 candidate rule (§6, v1 §5.4 row): a candidate reached through
an argument row (`CALLS_NAME ->`, `HAS_PARAM ->`, `HAS_PARAM <-`) stays in `items` as `possible`/`resolution-unverified`
in the impact column and folds in the depends-on column, with a control for an own-arrival candidate.

### 8.1 For P1-45b and P1-23d (`tdd-engineer`): added tests

All in `cpg/tests/test_answer.py` unless named; each rejection test first asserts its base validates. **Order: P1-45b, then P1-23d** (§9, R-8).

**P1-45b (the notice, §5.1).** Bases: `B-1.2` (`B-1.1` relabelled `"1.2"`); `B-notice` (`B-flow`, relabelled, plus `notices [{code: flow-not-enumerated, langs: [python]}]`).
- accepted: `"1.2"` accepted, `"1.3"` rejected (`schema`); `B-1.2` without `notices`; `B-notice`; the 1.0 golden and every 1.1 base unchanged;
- rejected (`schema`): `notices: []`; `langs: []`; a duplicate lang; a code other than `flow-not-enumerated`; an extra key on `Notice`; a `lang` with an upper-case letter;
- rejected (`V17`): `B-notice` relabelled `"1.1"`; `"1.0"`;
- rejected (`V20`): `B-flow` as `"1.2"` without `notices` (coverage); `notices` langs `[javascript]` where a hop carries `flow-unavailable:python`; two notices with one code; `notices` on an
  ambiguous answer; `B-1.2` (no labelled hop) **with** `notices` accepted (the converse is valid);
- `facts_bytes`: equal with and without a changed `builtAt` on `B-1.2`, and different with and without `notices`; `dumps` sorts `langs` `[python, javascript]` to `[javascript, python]`;
- the `ARRAY_RULES` coverage probe, `unclassified_strings` and `derived_data_fields == DATA_FIELDS` pass with their sets extended by `notices`/`Notice.langs`/`Notice.code` only;
- the 1.1-or-later helper: a mutant that tests `== "1.1"` in V18, V19, `facts_bytes` or `_check_hops` is killed by a `"1.2"` base (item C of `B-flow`, the `builtAt` case, a V19 type, a wiring chain);
- the docstrings and messages of `ccg/answer.py` that say "1.1" for "1.1 or later" are reworded (its module docstring, `facts_bytes`, `_check_labels`, `_check_flow_rows`, V19's check);
- mutants to kill: V20 (c) dropped; V20 (a)'s status clause dropped; V17's 1.1-no-`notices` clause dropped.
`build_answer` still emits `"1.1"` here and every existing engine test is green unchanged.

**P1-23d (V18's callee run, §5, `"1.2"` only).** Base `B-both`: a `"1.2"` impact document (with `notices [python]`) in python unit `U1`: `SA` → … → `HAS_CALL ←` `fb` (`function`) → `HAS_CALL →` site `s1`
(`end`, `possible`/`flow-unavailable:python`) → `CALLS →` callee `fg` (`end`, same label; item, distance 2) → `HAS_CALL →` site `s2` → `CALLS_NAME →` callee `fh` (`possible`/own
`name-match`; item, distance 3 on `bound` 3). Build every chain from fresh hop dicts.
- accepted: `B-both`; a `CALLS ←` exit still accepted (§5's other three rows);
- **rejected (V18 (e)): `B-both` relabelled `"1.1"`** (the assertion that no `"1.1"` document admits a V-both chain);
- rejected (b): `s1`'s hop `confirmed`; `fg`'s hop `confirmed`;
- rejected (c): `flow-unavailable:python` on the `CALLS_NAME →` hop; on a `CALLS →` hop that follows no `HAS_CALL →` of a run;
- rejected (d): `fg`'s reason `flow-unavailable:javascript`;
- rejected (f): a hop after `fg` other than `HAS_CALL →`; a hop after `s1` other than `CALLS →`/`CALLS_NAME →`;
- rejected (g): the run on a depends-on document; an item of the run with a `config` subject;
- rejected (V10): `fh` at distance 3 on `bound` 2;
- mutants to kill: the run clause dropped from (b); `("HAS_CALL","->")` dropped from the `"1.2"` exit rows (then `B-both` fails (e)); (f) dropped; the run's rows applied to `"1.1"`.
Engine tests (`test_engine_flow.py`): `bfs` holds the V-both constant and `SHIPPED_ROWS`; `algorithm_version(definition(SHIPPED_ROWS))` is `"1.2"` and equals `ALGORITHM_VERSIONS["1.2"]`; the as-built record is still `"1.1"` (the `5cf0…` literal);
`candidate_table("V-both") == SHIPPED_ROWS`; `walk(table=None)` and the builder are **unchanged** (still the caller-only table, label `"1.1"`): P1-39d flips them.

**P1-39d (the flip).** One commit changes `walk(table=None)`/`definition()` to `SHIPPED_ROWS`, `build_answer`'s label to `"1.2"` and adds `notices`. The assertions that move are named: `test_engine_answer.py::test_a_direct_caller_builds_a_valid_resolved_1_1_answer_whose_chain_is_the_walks`
(label, renamed) and `::test_the_recipe_names_the_shipped_definition_the_guards_and_every_query_the_build_ran` (algorithm `"1.2"`; its query-name set gains `units` if the read is new); in `test_engine_answer_v18.py`, the shipped-population label assertion
(`doc["answerSchemaVersion"] == "1.1"`) and `test_a_candidate_table_document_carries_the_unpublished_version_and_the_shipped_one_1_1` (the shipped table is now V-both, so the explicit as-built table is a candidate with `CANDIDATE_VERSION`), with the population's
shipped table inverted (`table=None` is V-both, as-built passes `bfs.ROWS`). The snapshot-row `"1.1"` pins (`engine_answer_helpers.py`'s `(SNAP, 0, EMPTY, "1.1", "1.0")`, `test_engine_staleness.py`, `test_engine_answer.py`'s snapshot rows) are the graph-model version, not answer labels, and stay.

## 9. Risks and open questions

**Risks.**
- **R-1 Facts under an equal stamp.** `Unit.flowAvailable` is snapshot data, and `-p1-graph.md` §3 hashes every
  value into `manifestHash`. This holds once that note is gated. Until then, a flag flip could change facts under an
  unchanged `snapshotId`.
- **R-2 Validator–engine drift** (v1 §10 item 6) grows with V18, which re-derives §5's labels from the rows. The
  shared fixtures are `B-flow` and P1-23's engine tests.
- **R-3 "P1 ships `1.2`"** holds only if the V-both definition of P1-23d is final when the `ac5-gate-1` pre-registration is written; the value (and the
  schema label `"1.2"`) must be in that record before the core freeze (index §7).
- **R-4 EP §12.3's record template writes `"version": "<semver>"`**, but answers emit `MAJOR.MINOR`. The record
  must carry `"1.2"` exactly. This is wording for EP's owner (`data-scientist`).
- **R-6 The notice is conservative by language.** It names every flag-false language the walk visited, not only those where a flow impact exists, so every P1 answer
  that reaches code carries it (all flags are expected false). That is the stated omission, not noise; a flag-true language leaves it.
- **R-7 Tests pinning `"1.1"`.** Only the answer-label and algorithm-version assertions move, in P1-39d (§8.1 names them). `"1.1"` as the snapshot row's graph-model version (`engine_answer_helpers.py`, `test_engine_staleness.py`, the snapshot rows of `test_engine_answer.py`) is a different version space and stays.
- **R-8 No commit emits or accepts a `"1.1"` V-both document.** P1-45b lands the label, helper and V20 while the builder still emits `"1.1"` with the caller-only table; P1-23d adds the table constant, the `"1.2"` pin and the `"1.2"`-only callee rows without changing the default walk or the builder; P1-39d flips table, label and `notices` in one commit.

**Open question Q-1 (stakeholder, non-blocking).** Should `ccg ask --fail-on any` fail on a notice? Recommendation: no. The exit code keeps meaning "items of that status exist"; the
notice is in the JSON and the text. A CI gate that needs the omission is a later, explicit option.

- **R-5 EP §5.2.1 lacks a `TO_UNIT` replay row** (§3.5). A `unit` item never matches gold, so it is never an
  H1–H4 candidate that HA-1 replays, and no score changes. The row is still owed by `data-scientist`, so that the
  allow-list matches the vocabulary.

## Ready to implement

Implement §7 of this document, test-first per §8 (P1-45, `tdd-engineer`; delivered), then §8.1 in this order: P1-45b (label, helper, `notices`, V20), P1-23d (the V-both table constant, the `"1.2"` pin, V18's `"1.2"`-only callee run),
then P1-39d (the engine flip: default table, label, `notices`) and P1-24b (renderers) in parallel (`docs/plans/connected-code-graph-p1-answers.md`). P1-45b and P1-23d both edit `cpg/ccg/answer.py` and `cpg/tests/test_answer.py`, so they are serialised. None touches `reasons.yaml` or the golden.

What P1-45 implemented:
- schema 1.1, accepting `"1.0"` too;
- `provenance.cacheBypass: true`, which requires `cacheHit: false`;
- `Bound` with exactly one of `keyPath` and `launch`, and port evidence;
- the `Unit` hop kind, the `TO_UNIT` edge type and `UnitSubject` (§3.5);
- V5's wiring, depends-on, per-site and `Unit` clauses, and V9's `not-mapped` check;
- the 1.1 `facts_bytes` projection;
- V17 (version labels and the 1.1 bound);
- V18 (§5's closed row list, entries and exits);
- V19 (four edge types never on a 1.1 hop).

The engine rules are for:
- P1-22: §1.2's version table, §4.2's counting, the cost tie-break, explicit flags in fixtures;
- P1-23: §5, §3.3's wiring row, §3.5's `TO_UNIT` row;
- P1-45b: §5.1's schema and V20; P1-23d: §5's callee run and §1.2's `1.2` entry; P1-39d: the default-table flip, §5.1's trigger and the `"1.2"` label; P1-24b: §5.1's wording and placement;
- P1-25 and P1-26: §2's input and §5.1's pass-through.

Nothing blocks P1-45b.
