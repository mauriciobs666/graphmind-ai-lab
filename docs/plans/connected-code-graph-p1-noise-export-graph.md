# Connected code graph, P1-36: store-side inputs of the noise and oracle tooling (graph)

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** P1-36 (P1)

Closes the producer gap of `docs/reviews/connected-code-graph-p1-36-impl.md` M2/m2: the oracle's Link set, the
adjacency snapshot, the naive over-lister, the Q-NC5 dump and the baseline's hit rule. The model is
`docs/plans/connected-code-graph-p1-graph.md` (1.1) as built; the consumers are `cpg/eval/tools/oracle.py` and
`cpg/eval/tools/noise.py` as committed; the baseline's method is `docs/plans/connected-code-graph-p1-ml.md` §5.1 (3).
Nothing here changes the store, the DDL or a consumer's schema, except the two consumer edits of §6.

CPG: considered, not relevant — the subject is the `ccg_<system>` store model; no loaded `cpg_*` graph indexes it.

**Evidence.** Every Cypher text below was run on a throwaway `v4.18.11` (own port, synthetic graph of §4, removed
afterwards) through `GRAPH.RO_QUERY`; the worked numbers of §4 were produced by a scratch reference of the §3
contract. No held-out data, custodian directory or shared store was read.

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

`links.jsonl`, one object per line, sorted by `(element, keyNorm)`; `oracle.py` reads `element` and `keyNorm` and ignores
the rest.

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
| `path`, `status`, `linkId`, `consumerKey` | diagnostics (`status` is `Link.status`; every status counts as "present as a Link", `-p1-ml.md` §6.5). |

**Consumer-form keys are never `keyNorm`.** `VIA.consumerKey` is the consumer's key and may start with a wrapper
placeholder (`GET /{?}/shop/api/x`, one extra segment, never matched by `oracle.py`'s per-segment rule). It is carried as
`consumerKey` only. A link into a never-built unit has no `TO` and its `ON` key **is** the consumer's key (`linker/link.py`
docstring), so the writer **omits** rows with `hasProvider` false or `keyNorm` null (keyless provider) and prints the
two omitted counts. Cost: such an edge counts as an oracle miss, which is the conservative direction; the alternative
(an `oracle.py` rule that strips a leading `{?}`) is lossy, because the placeholder stands for an unknown number of segments.
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
| `name` | `site`, `candidate` (symbolId), `resolved` (bool: the site also has a resolved `CALLS` to it) | `MATCH (c:CallSite)-[:CALLS_NAME]->(:Name)-[:RESOLVES_TO]->(t:Symbol) OPTIONAL MATCH (c)-[x:CALLS]->(t) RETURN c.callSiteId, t.symbolId, x IS NOT NULL AS resolved ORDER BY c.callSiteId, t.symbolId SKIP $skip LIMIT $limit` | `(site, candidate)` |
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
(§3; `{t: [record, ...]}`), `naive_over_lister`, `baseline_recall` (§6) and `nc5_view` (§5).

```python
def naive_over_lister(snapshot: dict, anchors: list[dict], bound: int = 2, *,
                      direction: str = "impact", in_scope=None) -> list[dict]
```

- `anchors`: one per held-out answer, `{"anchorId", "type": "symbol" | "contract" | "config", "symbolId" | "cpId" | "entryId"}`,
  the projection of that answer's resolved `anchor` (answer-v1 `AnchorSymbol`/`AnchorContract`/`AnchorConfig`). A `unit`
  anchor arrives as the contract anchor of its probe (EP §5.1). Ids are the gate snapshot's.
- `in_scope`: `None` (everything in scope) or a callable taking a `symbol` record; production passes
  `lambda s: not path_excluded(s["path"], exclusions)` (`noise.py`, the same filter as the system's frame, `-p1-ml.md` §6.3).
- **Returns**, one dict per anchor in `anchorId` order: `{"anchorId", "count", "countInScope", "countConfirmedOwn", "items"}`.
  `items` is the sorted list of distinct `symbolId`s listed; `count = len(items)`; `countInScope` those for which
  `in_scope` is true; `countConfirmedOwn` those whose own edge status is `confirmed` (rule 6). Deterministic: no clock,
  no randomness, output independent of record order. Raises `ValueError` (counts, never ids) for an unknown `bound < 0`,
  `direction != "impact"`, a duplicate `anchorId`, or an anchor id absent from the snapshot (never burden 0).

**Rules** (all ids are symbols unless stated; "owner of site `x`" = `site.owner`):

1. **Callers.** `callers(s)` = owners of every site with a `call` row to `s` **or** a `name` row with `candidate == s`
   (every name-match candidate of every call hop, whether or not the site is also resolved; a pair is counted once). A site
   whose owner is `s` itself adds nothing new.
2. **Seeds `S0`.** Symbol anchor: `{symbolId}`. Contract anchor: the `provides` symbols of `cpId` plus the owners of its
   `consumes` sites. Config anchor: the owners of the sites of every `via` row whose `linkId` has a `bound` row for `entryId`.
3. **Bound 2 applies to call hops only.** `Reach(S0)` = `S0` plus the callers found in at most `bound` reverse steps (BFS,
   visited set, so a cycle or recursion terminates). Seeds cost 0 (a provider is at distance 0, as in the system's answer);
   provider or consumer membership never costs a hop.
4. **Kind-wide contract members.** `K(anchor)`: symbol anchor, the `cp.kind` of every contract point it provides or whose
   `consumes` site it owns; contract anchor, its own `cp.kind`; config anchor, the `cp.kind` of the `link.cpId` of its bound
   links. `Kind(K)` = every `provides` symbol and every `consumes`-site owner of **any** contract point whose kind is in `K`,
   not restricted to the anchor's key and not bounded (the literal reading of "every provider and consumer of the same
   contract kind"; method flag F2). A kind-less anchor (`K` empty) adds nothing.
5. **Items** = `(Reach(S0) ∪ Kind(K)) − {anchor symbol}`; only a symbol anchor is removed (a contract or config anchor
   is not a symbol, its seeds are listed). The baseline lists symbols only: no config, unit or wiring item.
6. **Own status.** An item is confirmed-own if (a) a chain of `call` rows with `status == "confirmed"` only, at most `bound`
   steps, from a confirmed seed reaches it (name rows never qualify), or (b) it is in `Kind(K)` through a `provides`/
   `consumes` row with `status == "confirmed"`, or (c) it is a contract or config seed whose own `provides`/`consumes`/`via`
   row has `status == "confirmed"` (a symbol anchor is a confirmed seed). Otherwise possible-own. This is only for the split
   of the reduction (`-p1-gates.md` P1-36); it never changes `count`.

### 4.1 Worked graph and expected numbers

Twelve symbols, `bound` 2 unless stated. Name-match rows are `name`, calls are `call` (all `confirmed`), the rest as stated.

| Records | |
|---|---|
| calls | `B→A`, `C→B`, `D→C`, `F→E` |
| name matches | `B→A` (also resolved), `C→B` (also resolved), `E→A`, `G→B` |
| contract points | `CP1` http `GET /shop/api/x`, `CP2` http `GET /shop/api/y`, `CP3` mcp_tool `tool` |
| provides (confirmed) | `A→CP1`, `P3→CP1`, `P2→CP2`, `M→CP3` |
| consumes | `H→CP1` confirmed, `Q→CP2` possible |
| link / via / bound | `L1` ON `CP1`, via `H` confirmed, bound by entry `CE1`; `L2` ON `CP2`, via `Q` possible; `L3` ON `CP1` via `Q` possible |
| paths (for `in_scope`) | `H`, `P3`, `Q` end in `.ts`; the rest in `.py` |

| Anchor | `bound` | `items` | `count` | `countConfirmedOwn` | Why |
|---|---|---|---|---|---|
| symbol `A` | 2 | `B C E F G H P2 P3 Q` | **9** | **5** (`B C H P2 P3`) | reach: `B`, `E` (1), `C`, `F`, `G` (2); `A` provides http so `Kind` adds `P2 P3 H Q` (`A` itself removed); `E`, `F`, `G` rest on name rows, `F` only through `E`; `Q`'s `consumes` is possible; `D` is distance 3 |
| symbol `A` | 1 | `B E H P2 P3 Q` | 6 | 4 | `C F G` need 2 steps |
| symbol `A` | 0 | `H P2 P3 Q` | 4 | 3 | the kind-wide members only |
| symbol `A` | 3 | the 9 plus `D` | 10 | 6 | `D→C` is the third step |
| symbol `A`, `in_scope` = not `.ts` | 2 | as the first row | 9 | 5 | `countInScope` **6** (`H P3 Q` dropped) |
| contract `CP1` | 2 | `A B C E F G H P2 P3 Q` | **10** | **6** | seeds `A P3` (providers) and `H` (consumer); reach adds `B E C F G`; `Kind(http)` adds `P2 Q` |
| config `CE1` | 2 | `A H P2 P3 Q` | **5** | **4** | `CE1` is bound by `L1`, whose `via` site is `H` (seed, confirmed); `H` has no callers; `Kind(http)` adds `A P2 P3 Q` |
| symbol `D` | 2 | – | 0 | 0 | no callers, provides and consumes nothing |
| symbol `M` | 2 | – | 0 | 0 | the only mcp_tool member is `M` itself |

The unknown anchor `symbol Z`, a negative `bound`, `direction="depends-on"`, a duplicate `anchorId`: `ValueError`.

**Tests the implementer must write** (each with the planted input that kills the mutant): the table above; name rows dropped
(`E`, `G` vanish); a pair both resolved and name-matched counted once (`B` is 1 item); `Kind` restricted to the anchor's key
(`P2`, `Q` vanish); `Kind` bounded or seeded from reached symbols; the `<` against `<=` on `bound` (rows `bound` 1 and 3);
the anchor not removed (symbol `A`); confirmed-own counting a name chain; a call cycle (`A→B→A`) and a self call
terminate; a shuffled record order gives the identical result; an ownerless site is dropped and counted by `read_snapshot`,
not by the function.

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

## 6. Item-level recall: the baseline's hit rule, and the consumer edits

**Held-out gold item** (`ccg-key-v1` schema, as in `cpg/eval/dev/*.jsonl`; no held-out file was read):
`{anchorId, impacted: {type, repoId, path, qname | keyPath}, lines: [int], counting: bool, ...}`.

**System hit** is EP §5.2 H1-H4, scored by the P1-32b harness. **Baseline hit**, fixed here for `baseline_recall(baseline_items,
gold, symbols) -> (k, n)`:

> Gold item `g` is a baseline hit iff `g.impacted.type == "symbol"` and some `symbolId` in `baseline_items[g.anchorId]`
> names a snapshot `symbol` record `x` with `E(x.qname) == g.impacted.qname` **and** some line `ℓ ∈ g.lines` lies in
> `[x.lineStart, x.lineEnd]` or in one pair of `x.ownedRegions`.

That is `match_keys.h1_symbol_credit(x, g.impacted, g.lines)` imported, not reimplemented (`E` strips `#k` and truncates
at `<anon>#N`). One trap: the store's `ownedRegions` is a **flat** int list `[s0, e0, s1, e1, ...]` while
`h1_symbol_credit` reads pairs, so `baseline_recall` pairs it first (`[[o[i], o[i+1]] for i in range(0, len(o), 2)]`);
the test is symbol `E` of §4.1 (`ownedRegions [7, 9]`) credited by gold line 8 and not by line 6 outside its span.

- **Population.** `n` is the same for both figures: every held-out gold item with `counting: true`; `k_system` from the
  harness, `k_baseline` from this predicate; `noise-recall.json` is `{"system": [k, n], "baseline": [k, n]}`. The rule is over
  items, not anchors or clusters. The baseline's items are those of `naive_over_lister` at `bound` 2.
- **What differs from H1-H4, and why.** H2 (status) is vacuous: the baseline lists no status. H3 (own chain) and H4 (listed
  individually) are dropped: the baseline has no chains and lists every item. A `config` gold item is never a baseline hit
  (the baseline lists symbols only, §4 rule 5); the writer prints the number of `config` gold items so that effect is
  visible. The relaxations of H2-H4 can only raise the baseline's recall, which makes the comparator stronger and the
  "baseline recall is lower" flag rarer; the `config` rule works the other way.
- **Commitment.** Before any gate answer is scored, a decision record `cpg/eval/decisions/ccg-noise-recall-rule-v1.json`
  (owner `data-scientist`, committed by `teco`) carries the sha256 of this section and of `baseline.py`; `baseline.py` and
  `snapshot_export.py` join the harness files pinned by the `ac5-gate-1` record (`-p1-ml.md` §6.1). A rule changed after
  scoring is "after adaptation".

**Consumer edits** (`cpg/eval/tools/noise.py`, with the tests the P1-36 review asks for in M2):

1. `noise-baseline.jsonl` rows are `{anchorId, count, countInScope, countConfirmedOwn}`. The in-scope reduction divides the
   system's in-scope burden by `Σ countInScope`, the all-listed reduction by `Σ count`; today one figure feeds both (a
   baseline counted with scope against a system counted without it, or the reverse). `countConfirmedOwn` is reported only.
2. `score` exits 4 (counts, never ids) when the baseline's anchor set differs from the answers', `0 <= countInScope <= count`
   fails, a recall pair has `n <= 0` or `k > n` or `k < 0`, or the two pairs' `n` differ; it never reads a missing anchor as 0.

## 7. Method rulings needed, work order

`data-scientist` rules before `ccg-noise-recall-rule-v1` is recorded (each is a method choice, not a graph fact):

- **F1** the §6 predicate (H1 plus "listed", config never a hit, population `counting: true`).
- **F2** the literal "same contract kind" reading of §4 rule 4 (kind-wide, unbounded, kinds taken from the anchor only). A
  key-scoped or reach-seeded variant is a smaller, fairer baseline and a lower reduction; the choice moves the 5× figure.
- **F3** which burden pairs which reduction (§6 edit 1); and that the baseline counts symbols only while the system also lists
  config and unit items.
- **F4** `bound` 2 is the plan's; if the pre-registered default bound of `ac5-gate-1` differs, the baseline is not
  recomputed at it unless ruled.
- **F5** §2: a link with no provider (or a keyless one) is omitted from the oracle's Link set and counts as a miss, and
  `possible` links count as present.

Order: (1) `data-scientist` rules F1-F5; (2) `tdd-engineer` builds `baseline.py` (§4, §5, §6; tests of §4.1) test-first
and `snapshot_export.py` (§1-§3) with a live tier on a throwaway holding the §4.1 graph (the texts of §2, §3 and §5 are
the ones verified here; `GRAPH.EXPLAIN` of the paged reads is not part of the verification and they are label and
relationship scans by design, one-shot custodian reads); (3) the `noise.py` edits of §6 and the `test_oracle.py` fixture of
§2; (4) `teco` commits the record before the gate answers are scored. P1-36 stays open until (2) and (3) land
(review Q1); P1-37 may proceed with reduction and Q-NC5 "not computed" only if the ledger says so.
