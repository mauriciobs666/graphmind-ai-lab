# Connected code graph, P1-36: store-side inputs of the noise and oracle tooling (graph)

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** P1-36 (P1)

Closes the producer gap of `docs/reviews/connected-code-graph-p1-36-impl.md` M2/m2: the oracle's Link set, the
adjacency snapshot, the naive over-lister, the Q-NC5 dump and the baseline's hit rule. The model is
`docs/plans/connected-code-graph-p1-graph.md` (1.1) as built; the consumers are `cpg/eval/tools/oracle.py` and
`cpg/eval/tools/noise.py` as committed; the baseline's method is `docs/plans/connected-code-graph-p1-ml.md` §5.1 (3).
Rulings F1-F5 and amendments A1-A4, M1-M2, m1-m3 of `docs/reviews/connected-code-graph-p1-noise-export-ml.md` are folded
in. Nothing here changes the store or the DDL; the consumer edits are §7.

CPG: considered, not relevant — the subject is the `ccg_<system>` store model; no loaded `cpg_*` graph indexes it.

**Evidence.** Every Cypher text below was run on a throwaway `v4.18.11` (own port, synthetic graph of §4, removed
afterwards) through `GRAPH.RO_QUERY`; the worked numbers of §4 were produced by a scratch reference of the §4
contract; the engine distances of §4 rule 3 were read from `cpg/ccg/engine/bfs.py` (`ROWS`) and re-run through
`cpg/tests/test_engine_bfs_pins.py` (8 passed). No held-out data, custodian directory or shared store was read.

## 1. Where it runs and what is secret

- **Role.** One custodian-side script, `cpg/eval/tools/snapshot_export.py` (new), runs every text below as
  **`ccg_raw_<s>`** with `GRAPH.RO_QUERY` on the live key `ccg_<s>` (the role holds `+graph.ro_query` and
  `~ccg_<s>` only, `-p1-graph.md` §7). The engine role is not used (it is the answer path). `mcp__ccg__raw_query` is not
  the transport: its rows enter an agent's context and the export is tens of thousands of rows.
- **Snapshot guard.** The script reads `Snapshot.snapshotId` first and last (`MATCH (s:Snapshot) RETURN s.snapshotId`)
  and refuses unless both equal the gate's `--snapshot-id` (so a refresh during the export is an error, not a mixed file).
- **Paging.** Every list read ends `SKIP $skip LIMIT $limit`, `$limit` = 5,000 (below `RESULTSET_SIZE` 10,000), with an
  `ORDER BY` that is a total order; it stops on a short page. No keyset (`WHERE id > $after`): a range predicate on an
  indexed property is the silent-wrong-result trap of the quirks file. Per-query `timeout` ≤ `TIMEOUT_MAX`.
- **Custody.** The snapshot export and the Link dump are system facts, not held-out-derived, but the script writes them under
  the custodian root like everything else (`-p1-ml.md` §6.4); `baseline-*` files (§3) are held-out-derived. Stdout is
  counts only.

## 2. Link dump for `oracle.py`

`links.jsonl` (provided links) and `links-unprovided.jsonl` (§2 below), one object per line, sorted by `(element, keyNorm)`;
`oracle.py` reads `element` and `keyNorm` and, after §7, `status`.

```
MATCH (c:CallSite)-[v:VIA]->(l:Link) WHERE l.kind = 'http'
MATCH (s:Symbol)-[:HAS_CALL]->(c)
OPTIONAL MATCH (l)-[:ON]->(cp:ContractPoint)
OPTIONAL MATCH (l)-[:TO]->(p:Symbol)
RETURN s.qname AS element, s.path AS path, cp.keyNorm AS keyNorm, v.consumerKey AS consumerKey,
       l.status AS status, l.linkId AS linkId, p.symbolId IS NOT NULL AS hasProvider
ORDER BY element, linkId, c.callSiteId SKIP $skip LIMIT $limit
```

| Field | Value |
|---|---|
| `element` | the **consumer symbol's** `qname`, which is already `path::dotted.qname` (`Symbol.qname` is path-anchored, `-p0-graph.md` BF2): the owner of the `VIA` site, i.e. the symbol the link is one-per (`linker/link.py`). File-grain matching in `oracle.py` splits it on `::`. A module-level site gives `path::<module>`. |
| `keyNorm` | **the provider's key**: `Link ON ContractPoint.keyNorm` (`linker/link.py`: "`Link.key_norm` is the provider's `keyNorm`"; provider form, `{#}`/`{}` placeholders, the mount prefix already in it), as `VERB /path`. This is the join key a runtime request is compared with. |
| `path`, `status`, `linkId`, `consumerKey` | `status` is `Link.status`. **This note's own rule: a link of any status counts as "present as a Link"** (`-p1-ml.md` §6.5 says only "present in the gate snapshot as a `Link`"); `oracle.py` reports its hits split by the best status of a matching link (§7). |

**Consumer-form keys are never `keyNorm`.** `VIA.consumerKey` is the consumer's key and may start with a wrapper
placeholder (`GET /{?}/shop/api/x`, one extra segment, never matched by `oracle.py`'s per-segment rule). It is carried as
`consumerKey` only. A link into a never-built unit has no `TO` and its `ON` key **is** the consumer's key (`linker/link.py`
docstring), so the writer moves rows with `hasProvider` false or `keyNorm` null (keyless provider) out of `links.jsonl` into
`links-unprovided.jsonl` (same fields; `keyNorm` there is the consumer key or null) and prints the two counts, plus the
number of unprovided rows whose key path starts with `/shop/api` or `/{?}/shop/api` (the oracle's scope, so a zero is
visible). Such an edge counts as an oracle miss, the conservative direction; the alternative (an `oracle.py` rule that strips
a leading `{?}`) is lossy, because the placeholder stands for an unknown number of segments. `oracle.py` classifies a miss
as `linked-no-provider`, not `extractor`, when an unprovided row has the same file and the consumer key's segments after
its leading `{?}` run equal the last segments of the runtime path (per-segment, `{?}`/`{}`/`{#}` matching one segment;
classification only, never scoring).
The writer also prints the number of rows whose `keyNorm` contains `{**}`: `oracle.py` treats it as a literal, so a count
above 0 needs a one-line `oracle.py` rule from `data-scientist` before the number is quoted.
Distinct on `(element, keyNorm)` in the writer: keep the best status, then the lowest `linkId`.

**Grain.** The dump is function grain by construction (one element per consumer symbol); `oracle.py` matches file-grain
runtime rows against any link whose element is in the file, and function-grain rows by exact element, where m1 of the
P1-36 review (runtime `co_name` against dotted `qname`) stays `oracle.py`'s open defect. `test_oracle.py` gets one row of
this format as a fixture; the synthetic graph of §4 gives, verified:

```
{"consumerKey":"GET /{?}/shop/api/x","element":"h.ts::h","keyNorm":"GET /shop/api/x","linkId":"L1","path":"h.ts","status":"confirmed"}
```

and the unprovided `L3` (consumer-form key, no `TO`) is omitted and counted.

## 3. Snapshot adjacency export (custodian-side JSONL)

`snapshot-adjacency.jsonl`: one canonical-JSON object per line (`json.dumps(sort_keys=True, ensure_ascii=False)`), the
`meta` line first, then records grouped by `t` in the order below, each group sorted by its sort key. Ids are the
store's 32-hex ids; strings are as stored (`qname`, `path`, `keyNorm` are not screened here: the file is custodian-side).

| `t` | Fields | Query (all `GRAPH.RO_QUERY`, paged where `SKIP` shown) | Sort |
|---|---|---|---|
| `meta` | `snapshotId`, `systemId`, `graphModelVersion`, `exportVersion: 1` | `MATCH (s:Snapshot) RETURN s.snapshotId, s.systemId, s.graphModelVersion` | – |
| `unit` | `unitId`, `repoId`, `language`, `buildStatus`, `flowAvailable` | `MATCH (u:Unit) RETURN u.unitId, u.repoId, u.language, u.buildStatus, u.flowAvailable ORDER BY u.unitId` | `unitId` |
| `symbol` | `symbolId`, `unitId`, `repoId`, `qname`, `path`, `lineStart`, `lineEnd`, `ownedRegions` (flat int list, absent when none) | `MATCH (s:Symbol) RETURN s.symbolId, s.unitId, s.repoId, s.qname, s.path, s.lineStart, s.lineEnd, s.ownedRegions ORDER BY s.symbolId SKIP $skip LIMIT $limit` | `symbolId` |
| `site` | `siteId`, `owner` (symbolId), `unitId`, `path`, `line`, `calleeName` | `MATCH (s:Symbol)-[:HAS_CALL]->(c:CallSite) RETURN c.callSiteId, s.symbolId, c.unitId, c.path, c.line, c.calleeName ORDER BY c.callSiteId SKIP $skip LIMIT $limit` | `siteId` |
| `call` | `site`, `callee` (symbolId), `status`, `reason` | `MATCH (c:CallSite)-[e:CALLS]->(t:Symbol) RETURN c.callSiteId, t.symbolId, coalesce(e.status,'confirmed'), coalesce(e.reason,'resolved-call') ORDER BY c.callSiteId, t.symbolId SKIP $skip LIMIT $limit` | `(site, callee)` |
| `name` | `site`, `candidate` (symbolId) | `MATCH (c:CallSite)-[:CALLS_NAME]->(:Name)-[:RESOLVES_TO]->(t:Symbol) RETURN c.callSiteId, t.symbolId ORDER BY c.callSiteId, t.symbolId SKIP $skip LIMIT $limit` | `(site, candidate)` |
| `cp` | `cpId`, `kind`, `keyNorm` | `MATCH (cp:ContractPoint) RETURN cp.cpId, cp.kind, cp.keyNorm ORDER BY cp.cpId SKIP $skip LIMIT $limit` | `cpId` |
| `provides` | `symbolId`, `cpId`, `status`, `reason` | `MATCH (s:Symbol)-[e:PROVIDES]->(cp:ContractPoint) RETURN s.symbolId, cp.cpId, e.status, e.reason ORDER BY s.symbolId, cp.cpId SKIP $skip LIMIT $limit` | `(symbolId, cpId)` |
| `consumes` | `site`, `cpId`, `status`, `reason` | `MATCH (c:CallSite)-[e:CONSUMES]->(cp:ContractPoint) RETURN c.callSiteId, cp.cpId, e.status, e.reason ORDER BY c.callSiteId, cp.cpId SKIP $skip LIMIT $limit` | `(site, cpId)` |
| `link` | `linkId`, `kind`, `status`, `reason`, `cpId` (the `ON` target, null if none), `provider` (the `TO` symbolId, null if none) | `MATCH (l:Link) OPTIONAL MATCH (l)-[:ON]->(cp:ContractPoint) OPTIONAL MATCH (l)-[:TO]->(p:Symbol) RETURN l.linkId, l.kind, l.status, l.reason, cp.cpId, p.symbolId ORDER BY l.linkId SKIP $skip LIMIT $limit` | `linkId` |
| `via` | `site`, `linkId`, `status`, `reason`, `consumerKey` (absent when null) | `MATCH (c:CallSite)-[v:VIA]->(l:Link) RETURN c.callSiteId, l.linkId, v.status, v.reason, v.consumerKey ORDER BY c.callSiteId, l.linkId SKIP $skip LIMIT $limit` | `(site, linkId)` |
| `entry` | `entryId`, `unitId`, `path`, `keyPath` (absent for a launch entry), `launchKind` (absent for a key entry) | `MATCH (e:ConfigEntry) RETURN e.entryId, e.unitId, e.path, e.keyPath, e.launchKind ORDER BY e.entryId SKIP $skip LIMIT $limit` | `entryId` |
| `bound` | `linkId`, `entryId`, `siteId`, `ord` | `MATCH (l:Link)-[b:BOUND_BY]->(e:ConfigEntry) RETURN l.linkId, e.entryId, b.siteId, b.ord ORDER BY l.linkId, e.entryId, b.siteId, b.ord SKIP $skip LIMIT $limit` | all four |

Edge direction and status, for the consumers of the file:

- **Impact** (the P1 gate direction) is **reverse reach**: from a node, the *owner* of every site that points at it. A site
  points at a symbol by a `call` row (resolved) or a `name` row (name-match candidate); the owner is `site.owner`. Providers
  are `provides.symbolId`; consumers are `consumes.site` mapped to `site.owner`.
- **Depends-on** is the same records read forward: from a symbol, the `call.callee` and `name.candidate` of every `site`
  it owns; from a contract anchor, the `cp`'s `provides` symbols. Nothing is exported twice and no depends-on record
  exists. `naive_over_lister` takes `direction="impact"` only at P1; any other value raises `ValueError` (the held-out
  gate asks impact, EP §5.1).
- **Per-hop edge status** is the stored one with the reader's defaults applied (`call`: `coalesce` to `confirmed`/
  `resolved-call`; `name`: always possible by construction, so no status field); `provides`, `consumes`, `via` carry theirs.
  It does **not** depend on `Unit.flowAvailable`: a hop's own edge is `confirmed` exactly when its stored status is,
  which is the "confirmed-in-flag-true" reading of `docs/plans/connected-code-graph-answer-spec2.md` §5. `unit.flowAvailable`
  is exported so a consumer can still tell which hops a flag-false walk would label.
- **Contract kind providers and consumers** are `provides`/`consumes` joined to `cp.kind`.

**Validation (`read_snapshot`).** `meta` first with `exportVersion == 1` and `graphModelVersion == "1.1"`; every record
has its fields; referential closure: `site.owner`, `call.callee`, `name.candidate`, `provides.symbolId` ∈ symbols;
`call.site`, `name.site`, `consumes.site`, `via.site` ∈ sites; `provides.cpId`, `consumes.cpId`, `link.cpId` ∈ cps;
`via.linkId`, `bound.linkId` ∈ links; `bound.entryId` ∈ entries; no duplicate key per `t`. A violation raises
`ValueError` naming the record type and count. A `CallSite` with no `HAS_CALL` owner is not exported as a `site`
(the join returns no row; `candidate_count` of `-p1-graph.md` §5 has the same case), and no caller can be listed from it:
`read_snapshot` drops the `call`/`name`/`consumes`/`via` rows on such a site and returns their count under `dropped`, which
the writer prints. Any other closure break is an error.

## 4. `naive_over_lister`: the baseline's burden

Module `cpg/eval/tools/baseline.py` (new; stdlib only, pure, no store access) holds `read_snapshot(path) -> dict`
(§3; `{t: [record, ...]}`), `naive_over_lister`, `baseline_recall` (§6), `dominance` (§7) and `nc5_view` (§5).

```python
def naive_over_lister(snapshot: dict, anchors: list[dict], bound: int = 2, *, direction: str = "impact",
                      kind_scope: str = "kind", in_scope=None) -> list[dict]
```

- `anchors`: **one per answer** (never per gold anchor), `{"anchorId", "type": "symbol" | "contract" | "config" | "unresolved",
  "symbolId" | "cpId" | "entryId"}`, the projection of that answer's resolved `anchor` (answer-v1 `AnchorSymbol`/
  `AnchorContract`/`AnchorConfig`). A `unit` anchor arrives as the contract anchor of its probe (EP §5.1); a `unit`
  type in the input is a `ValueError`, so a unit and its probe cannot both enter. An answer whose anchor did not resolve
  arrives as `type: "unresolved"`, and only that (the answer's own `anchor` is absent; EP §5.4 concerns items, not anchors). The CLI takes `bound` from the `ac5-gate-1` record's
  `defaultDepthBound`; the default 2 is for tests only.
- `kind_scope`: `"kind"` (headline, rule 4) or `"key"` (reported beside it, rule 4).
- `in_scope`: `None` (everything in scope) or a callable taking a `symbol` record; production passes
  `lambda s: not path_excluded(s["path"], exclusions)` (`noise.py`, the same filter as the system's frame, `-p1-ml.md` §6.3).
- **Returns**, one dict per anchor in `anchorId` order: `{"anchorId", "type", "count", "countInScope", "countConfirmedOwn",
  "byName": {"resolvedSite", "unresolvedSite"}, "bySeed", "items"}`. `items` is the sorted list of distinct `symbolId`s listed;
  `count = len(items)`; `countInScope` those for which `in_scope` is true; `countConfirmedOwn` those whose own edge status is
  `confirmed` (rule 6); `byName` the number of items that disappear when the name rows of sites that have a `call` row
  (`resolvedSite`: a `resolution-unverified` candidate) or of sites that have none (`unresolvedSite`) are ignored; `bySeed` partitions the items, by precedence, into `kindWideOnly` (in `Kind`, not in `Reach`), `nameOnly` (in
  `Reach` only through name rows), `providerOnly` (reached only from cost-0 provider seeds), `consumerOrConfigOnly` (reached only
  from consumer or config seeds) and `other` (a symbol anchor's callers, or mixed), summing to `count`. Deterministic:
  no clock, no randomness, independent of record order.
- **Unresolved anchor (not an error).** An anchor of `type: "unresolved"` returns
  `{anchorId, type, count: 0, countInScope: 0, countConfirmedOwn: 0, items: [], unresolved: true}`: the system's answer
  exists with burden 0, its gold items stay in `n` and are misses for both (§6). Raises `ValueError` (counts, never ids) for an
  id absent from the snapshot on a resolved answer (a build or export mismatch, never burden 0), `bound < 0`, `direction != "impact"`, a `kind_scope` outside the two, a `unit` anchor type or a duplicate `anchorId`.

**Rules** (ids are symbols unless stated; "owner of site `x`" = `site.owner`):

1. **Callers.** `callers(s)` = owners of every site with a `call` row to `s` **or** a `name` row with `candidate == s`:
   every name-match candidate of every call hop, **including the name rows of a site that is also resolved** (the
   `resolution-unverified` candidates; literal reading, named in the record, counted by `byName`). A pair is counted once.
   A site whose owner is `s` adds nothing new.
2. **Seeds `S0` and their cost**, read from the engine's rows (`cpg/ccg/engine/bfs.py` `ROWS`, pinned in
   `cpg/tests/test_engine_bfs_pins.py`; the AC-1 golden of `docs/plans/connected-code-graph-answer-spec.md` §10 agrees):
   - symbol anchor: `{symbolId}` at 0;
   - contract anchor: the `provides` symbols of `cpId` at **0** (`provides_in`, increment 0), and the **union** of the owners of its
     `consumes` sites and of the `via` sites of every link `ON` it, at **1** (`consumes_in` and `via_in:contract`, increment 1: the golden's `getHealth` is `direct`,
     distance 1). A symbol that does both takes 0;
   - config anchor: the owners of the sites of every `via` row whose `linkId` has a `bound` row for `entryId`, at **2**
     (`bound_in` 1, then `via_in:site` 1, then `owner_of` 0). The review's "a config consumer costs 1" is not what the
     engine does.
3. **Bound 2 counts the same unit as the system: inter-function hops.** `Reach(S0)` = `S0` members with cost `<= bound`, plus
   callers found by reverse call steps (BFS, visited set; a cycle or recursion terminates) while the total cost stays `<=
   bound`: a seed of cost `c` has `bound - c` further steps. Provider or consumer membership is not free beyond its cost above.
4. **Kind-wide contract members.** `CP(anchor)`: symbol anchor, the contract points it provides or whose `consumes` site it
   owns; contract anchor, its own `cpId`; config anchor, the `link.cpId` of its bound links. `kind_scope == "kind"`:
   `Kind` = every `provides` symbol and every `consumes`-site owner of **any** contract point whose kind is in the kinds of
   `CP(anchor)`, unbounded (the literal "every provider and consumer of the same contract kind", the pre-registered
   reading). `kind_scope == "key"`: `Kind` = the providers and consumer owners of the points in `CP(anchor)` only, unbounded.
   A **symbol** anchor acquires its kind through its own contract points: an extension of the pre-registered text that
   raises burden (a script calling one MCP tool receives the whole MCP kind), named in the record and reported by anchor type
   (§7). No contract points: nothing added.
5. **Items** = `(Reach(S0) ∪ Kind) − {anchor symbol}`; only a symbol anchor is removed. Symbols only: no config, unit or wiring item.
6. **Own status.** An item is confirmed-own if (a) a chain of `call` rows with `status == "confirmed"` only, within the cost
   bound of rule 3, from a confirmed seed reaches it (name rows never qualify), or (b) it is in `Kind` through a `provides`/
   `consumes` row with `status == "confirmed"`, or (c) it is a contract or config seed whose own `provides`/`consumes`/`via`
   row has `status == "confirmed"` (a symbol anchor is a confirmed seed). It is reporting only and never changes `count`.

**What the reduction measures.** The baseline is reverse call reach with name rows plus kind-wide members. Against it the
system's reduction is flow-sensitivity plus kind filtering, not "correctness": in a flag-true unit a symbol anchor lists its
direct callers only (precise `arg(i)`/`Ret` rows), and the baseline lists callers to the bound.

### 4.1 Worked graph and expected numbers

Fifteen symbols, `bound` 2 and `kind_scope` `"kind"` unless stated. Calls are `call` rows (all `confirmed`), the rest as stated.

| Records | |
|---|---|
| calls | `B→A`, `C→B`, `D→C`, `F→E`, `R→H`, `S→R` |
| name matches | `B→A` (also resolved), `C→B` (also resolved), `E→A`, `G→B`, `F→K` (site of `F` is resolved to `E`) |
| contract points | `CP1` http `GET /shop/api/x`, `CP2` http `GET /shop/api/y`, `CP3` mcp_tool `tool` |
| provides (confirmed) | `A→CP1`, `P3→CP1`, `P2→CP2`, `M→CP3` |
| consumes | `H→CP1` confirmed, `Q→CP2` possible |
| link / via / bound | `L1` ON `CP1`, via `H` confirmed, bound by entry `CE1`; `L2` ON `CP2`, via `Q` possible; `L3` ON `CP1`, via `Q` possible |
| paths (for `in_scope`) | `H`, `P3`, `Q` end in `.ts`; the rest in `.py` |

| Anchor | `bound`, scope | `items` | `count` | `countConfirmedOwn` | Why |
|---|---|---|---|---|---|
| symbol `A` | 2, kind | `B C E F G H P2 P3 Q` | **9** | **5** (`B C H P2 P3`) | reach `B`, `E` (1), `C`, `F`, `G` (2); `A` provides http, so `Kind` adds `P2 P3 H Q` (`A` removed); `E F G` rest on name rows; `Q`'s `consumes` is possible; `D` is 3. `byName` `{resolvedSite 0, unresolvedSite 3}` |
| symbol `A` | 1 / 0 / 3, kind | 6 / 4 / 10 items (3: the nine plus `D`) | 6 / 4 / 10 | 4 / 3 / 6 | one step fewer, none, one more |
| symbol `A` | 2, key | `B C E F G H P3` | 7 | 4 | `Kind` is `CP1`'s providers and `consumes` owners `A P3 H` only: `P2`, `Q` vanish (a via-only owner is a seed of a contract anchor, not a member of a symbol anchor's key scope) |
| contract `CP1` | 2, kind | `A B C E F G H P2 P3 Q R` | **11** | **7** | providers `A P3` at 0; consumer owner `H` (`consumes_in`) and via owner `Q` (`via_in:contract`, link `L3` ON `CP1`, possible) at 1: reach `B E` (1), `C F G` (2) from the providers, `R` (2) from `H`; `S` (3) is out; `Kind(http)` adds `P2` only. Under "consumers cost 0" `S` would be in (12) |
| contract `CP1` | 3, kind | the eleven plus `D S` | 13 | 9 | |
| contract `CP1` | 2, key | `A B C E F G H P3 Q R` | 10 | 6 | `P2` vanishes; `Q` stays as a via seed. 3, key: the ten plus `D S` = 12, confirmed-own 8 |
| config `CE1` | 2, kind | `A H P2 P3 Q` | **5** | **4** | bound by `L1`, whose `via` site owner `H` is the seed at cost 2, so `R` (3) is out; `Kind(http)` adds `A P2 P3 Q` |
| config `CE1` | 3, kind | the five plus `R` | 6 | 5 | |
| config `CE1` | 2, key | `A H P3` | 3 | 3 | |
| symbol `K` | 2, kind | `F` | 1 | 0 | `F` reaches `K` only by a name row on a resolved site: `byName` `{resolvedSite 1, unresolvedSite 0}` |
| symbol `D` | 2 | – | 0 | 0 | no callers, no contract points |
| symbol `M` | 2 | – | 0 | 0 | the only mcp_tool member is `M` itself |
| symbol `Z`, or `type: "unresolved"` | any | – | 0 | 0 | row with `unresolved: true`, not an error |

`symbol A`, `in_scope` = not `.ts`: `count` 9, `countInScope` **6** (`H P3 Q` dropped). `ValueError`: negative `bound`,
`direction="depends-on"`, a `unit` anchor, a duplicate `anchorId`.

**Tests the implementer must write** (each with the planted input that kills the mutant): the table above (expected sets
derived from the rules, not from the implementation); name rows dropped (`E F G` vanish from `A`); a pair both resolved and
name-matched counted once; `Kind` restricted to the key (`kind_scope`); `Kind` bounded or seeded from reached symbols;
consumer seeds at cost 0 (`S` appears at `CP1`/2), config seeds at cost 1 (`R` appears at `CE1`/2); `<` against `<=` on
`bound`; the anchor not removed; confirmed-own counting a name chain; a call cycle and a self call terminate; shuffled
records give the identical result; one row per answer (a `unit` anchor and its probe contract anchor arrive once, as the
probe); an ownerless site is dropped and counted by `read_snapshot`, not by the function.

## 5. Q-NC5 dump (unbound confirmed links, key with providers in more than one unit)

`noise.py unbound_multi_unit` consumes `{"links": [{keyNorm, bound, status}], "providers": [{keyNorm, unitId}]}`.
`nc5_view(snapshot)` projects it from the export with no new query: `links` = every `link` with a `cpId`, `keyNorm` =
that `cp.keyNorm`, `bound` = at least one `bound` row for the `linkId`, `status` = `Link.status`; `providers` = the
distinct `(cp.keyNorm, symbol.unitId)` over `provides`. `nc5_view` raises if one `keyNorm` occurs under two `cp.kind`s
(the consumer joins on `keyNorm` alone). The **independent check** is this count, which the writer runs and compares
with `unbound_multi_unit(nc5_view(...))` (both must be equal integers; verified on the synthetic graph: `L3` counted,
`L1` bound, `L2` single-unit):

```
MATCH (s:Symbol)-[:PROVIDES]->(cp:ContractPoint)
WITH cp, count(DISTINCT s.unitId) AS nu WHERE nu > 1
MATCH (l:Link)-[:ON]->(cp)
OPTIONAL MATCH (l)-[b:BOUND_BY]->(:ConfigEntry)
WITH l, count(b) AS nb WHERE nb = 0
RETURN count(l) AS unboundMultiUnit,
       sum(CASE WHEN l.status = 'confirmed' THEN 1 ELSE 0 END) AS unboundMultiUnitConfirmed
```

(`sum` returns a float, `0.0`: the writer converts with `int`.) Unbound is the absence of `BOUND_BY`, read with
`OPTIONAL MATCH` and `count`, never `exists()` over a pattern (quirks file). The list form for the report's custodian file,
verified: the same text ending `WITH l, cp, nu, count(b) AS nb WHERE nb = 0 RETURN l.linkId, l.status, cp.kind, cp.keyNorm,
nu ORDER BY l.linkId` returns `[L3, possible, http, GET /shop/api/x, 2]`. It counts **links** (rows), as the P1-36 text
and the gate review's Q3 read it, not keys or units.

## 6. Item-level recall: the baseline's hit rule

*(Final text: `ccg-noise-recall-rule-v1` pins the sha256 of §4-§6 together and of `baseline.py`.)*

**Gold item** (`ccg-key-v1`, as in `cpg/eval/dev/*.jsonl`): `{anchorId, impacted: {type, repoId, path, qname | keyPath}, lines:
[int], counting: bool, ...}`. **System hit** is EP §5.2 H1-H4, scored by the P1-32b harness.

**Baseline hit.** Gold item `g` is a baseline hit iff `g.impacted.type == "symbol"` and some `symbolId` in the baseline's item
list **for `g.anchorId`** (never pooled across anchors) names a snapshot `symbol` record `x` with
`match_keys.h1_symbol_credit(x, g.impacted, g.lines)`: `E(x.qname) == g.impacted.qname` and some `ℓ ∈ g.lines` in
`[x.lineStart, x.lineEnd]` or in one pair of `x.ownedRegions`. The function is imported, not reimplemented. The store's
`ownedRegions` is a flat list `[s0, e0, s1, e1, ...]` and `_regions` reads pairs, so `baseline_recall` pairs it first
(`[[o[i], o[i+1]] for i in range(0, len(o), 2)]`); test: symbol `E` of §4.1 with `ownedRegions [7, 9]` is credited by gold line 8
and not by line 6. EP §5.2 H2-H4 are not applied (the baseline has no status, chain or grouping); a `config` gold item is
never a hit.

**Population (identity, not a count).** The input is the harness's list of scored counting gold items, one JSONL row per item in
`system-hits.jsonl` (custodian): `{anchorId, impacted, lines, hit, keyError}`; `baseline_recall` never reads the gold files.
`itemSetSha256` = sha256 (UTF-8) of the lines `anchorId|<type>:<repoId>:<path>:<qname or keyPath>|<l1,l2,...>` (`lines` sorted
ascending, comma-joined), the list sorted by Python string order (code point) and joined by `\n`.
Both figures are over the same rows by construction, and `score` recomputes the hash from `system-hits.jsonl` and exits 4 when it
differs from the one in `noise-recall.json`. Items under an anchor that did not resolve stay in `n` and are misses for both
(`unresolved: true` rows of §4 add nothing to either burden sum). The **ex-key-error** figure drops the `keyError` items from
both, recomputed on that set with its own `itemSetSha256`.

**Figures** (`noise-recall.json`; counts only in any report):

| Key | Content |
|---|---|
| `system`, `baseline` | `[k, n]` over all scored items (the pre-registered "same hit rule", headline) |
| `systemSym`, `baselineSym` | `[k, n_sym]` over symbol-type items only, so a baseline's structural zero on `config` items is not read as its listing strategy |
| `paired` | the 2×2 of the same items: `both`, `systemOnly`, `baselineOnly`, `neither`; reported beside the two Wilson intervals (`oracle.wilson`), never judged by interval overlap |
| `configGold` | the number of `config` gold items |
| `exKeyError` | the same keys over the reduced set |

**Flag, as ratified:** reduction is flagged iff `baseline_k * system_n < system_k * baseline_n` (exact, cross-multiplied).
**Reported, not a bar:** when `baseline_k * system_n > system_k * baseline_n` the baseline recalls more than the system, the
reduction is not at equal recall and is an upper bound; the figure is called a burden ratio with both recalls beside it, never
"at equal recall". The same two statements are made on the symbol-only pair.

**Baseline variants.** The headline is `kind_scope="kind"` (the pre-registered reading, §4 rule 4). The key-scoped variant is
emitted beside it (`noise-baseline-key.jsonl`, `noise-recall-key.json`, same record, same hash) with no threshold. Which variant
carries the ≥ 5× bar from the next gate is the stakeholder's (S1, §7).

**Named upward or one-sided choices** (each raises the baseline's burden or recall, or fires the flag one way only; the report
prints them): the name rows of already-resolved sites count (`byName`); a symbol anchor acquires a kind from its own contract
points (burden printed by anchor type); the flag fires only for lower baseline recall; `config` gold items are never baseline hits.

**Bound.** The baseline's bound is the `ac5-gate-1` record's `defaultDepthBound`. The generator and `score` refuse unless every
answer's `depthBound.bound` equals it. A sensitivity at bound 1 and 3 is optional and reported only.

**Burden pairing.** In-scope reduction = `Σ countInScope / Σ system in-scope burden`; all-listed = `Σ count / Σ system all-listed
burden`; never crossed. The headline is in-scope. The baseline lists symbols only, so the report prints the number of non-symbol
system items (this asymmetry lowers the baseline, conservative against the reduction).

**Commitment.** The decision record `cpg/eval/decisions/ccg-noise-recall-rule-v1.json` (owner `data-scientist`, committed by `teco`)
precedes any scored gate answer; `baseline.py` and `snapshot_export.py` join the harness files pinned by `ac5-gate-1`
(`-p1-ml.md` §6.1). The generator needs only the snapshot and the anchors; whoever can see held-out answers does not run it before
the record. A rule changed after scoring is "after adaptation".

## 7. Consumer edits, dominance check, pins, order

**`cpg/eval/tools/noise.py`** (tests per the P1-36 review M2):

1. **Reduction pairing, two places.** `score` builds one `base` list from `baseline[a]` and passes it to both
   `reduction(base, bin_, ...)` and `reduction(base, ball, ...)`; `bootstrap` (via `_agg`) divides `g["base"]` by the in-scope
   `g["burden"]`. Both change: the baseline file rows are `{anchorId, type, count, countInScope, countConfirmedOwn, byName,
   unresolved?, bound}`; `score` uses `countInScope` for the in-scope reduction and `count` for the all-listed one, and
   `bootstrap` takes a `(base, burden)` per-anchor vector pair per pairing, each pairing with its own interval.
2. **Refusals (exit 4, counts only):** baseline anchor set differs from the answers'; `0 <= countInScope <= count` fails; an
   `unresolved` row with `count != 0`; `noise-recall.json` with `k > n`, `n <= 0`, or an `itemSetSha256` that differs from
   `system-hits.jsonl`'s; a baseline row's `bound` or any answer's `depthBound.bound` different from the record's
   `defaultDepthBound` (answers lines gain `depthBound`). It prints the number of unresolved rows.
3. **Reports** (counts only): burden and reduction by anchor type, baseline and system; the `paired` 2×2 and its Wilson intervals;
   `byName`; the number of non-symbol system items; the key-scoped variant beside the headline.
4. **Frame uniqueness (m1).** `build_frame` asserts `(anchorId, itemId)` is unique and that every possible item's `reason` is a
   string; exit 4 with the duplicate count. Test: one planted duplicate.
5. **Spot-check agreement (m2).** `agreement` is also reported split into confirmed and possible items; with fewer than 10 spot
   keys that are possible items, the possible-precision note is "agreement not measured on possible items" (the `n = 0` wording
   class); the Wilson interval of the raw agreement is printed.

**`cpg/eval/tools/oracle.py`** (F5): reads `status` from `links.jsonl`; reports hits split by the best status of a matching link
(`confirmed` / `possible`, counts only, no threshold); a miss is `linked-no-provider` under the §2 rule; a fixture row of the §2
format is added to `test_oracle.py`.

**Dominance check (M1), dev anchors only, design time.** `baseline.dominance(system_items, baseline_rows) -> {"A", "B"}` over
**every** system symbol item for each dev anchor, at the same bound, absent from the baseline's `items` counts as a violation.
Class A: the item's chain has no excluded edge (no `->` hop, no `DATA`, `CONTROLLED_BY`, `HAS_PARAM`, `HAS_RETURN`, `HAS_COND`),
reported as `{checked, violations}`; expected 0, any violation is a baseline bug or a distance mismatch. Class B: the item's chain
has an excluded edge, reported as `{checked, violations}` split by the first excluded edge type (`->` callee run, `DATA`, ...), so
the baseline's blind spots are visible rather than excluded by construction. Counts only. A dev dry run of the two
variants' `Σ count / Σ system burden` is a sizing measurement nobody tunes on; a rule change after it is a defect fix only.

**Draw choices pinned in the `ac5-gate-1` record** (accepted as implemented in `noise.py`; code-level facts a verifier needs):
(1) hash input `sha256(f"{seed}|{anchorId}|{itemId}")` hex ascending, `seed = "ccg-noise-v1|" + gateId + "|" + snapshotId`; ids carry
no `|`; (2) `itemId = "<unitId>:<subjectId>"` (symbolId, entryId or unitId by subject type), which changes the order against the
bare id; (3) spot seed `seed + "|spot"` over the labelled keys' `anchorId|itemId`, 20 keys, drawn before any label value is read;
(4) allocation: `min(N_h, 5)` per stratum, the rest of 100 by largest remainder with quotas proportional to `N_h` over **all**
strata (not net of the minima), remainder ties by `(reason, band)` sorted order, a capped stratum's unspent share to the largest
unsampled stratum; (5) a stratum with `N_h <= 5`, and a frame at or under 100, is sampled whole; (6) the frame is unique on
`(anchorId, itemId)`, stratum keys are `(reason, band)` with a string `reason`, and `distance` is the system's reported one.

**Stakeholder question S1 (not blocking gate 1; before the next gate's pre-registration, and before anyone sees the variants'
reductions):** which reduction carries the ≥ 5× bar: (a) kind-wide, the ratified wording; (b) key-scoped; (c) both (the review's
recommendation).

**Order.** (1) `tdd-engineer` builds `baseline.py` (§4-§6; the tests of §4.1) test-first and `snapshot_export.py` (§1-§3) with a
live tier on a throwaway holding the §4.1 graph (the texts of §2, §3 and §5 were verified; they are label and relationship scans
by design, one-shot custodian reads, `GRAPH.EXPLAIN` not part of the verification); (2) the §7 consumer edits; (3) `teco`
writes the record with the two hashes and commits it before any gate answer is scored. P1-36 stays open until (1) and (2) land;
P1-37 may proceed with reduction and Q-NC5 "not computed" only if the ledger says so.
