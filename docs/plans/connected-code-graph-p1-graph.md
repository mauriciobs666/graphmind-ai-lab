# Connected code graph: graph model 1.1 for P1 (N-GRAPH)

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** C-1001 (P1) · **Version:** 3 · **Extends:** `docs/plans/connected-code-graph-p0-graph.md` · **Reviews:** `docs/reviews/connected-code-graph-p1-graph.md`

Revision 2026-11-10: revised in place after review Pass 3 and the applied plan fixes.

This note states **only the change against** graph model v1 (`docs/plans/connected-code-graph-p0-graph.md`, "v1"),
which itself extends `docs/plans/connected-code-graph-graph.md` (the "graph note"). Anything not named here is as v1
says. It is the contract for P1-18 (assembler), P1-19 (loader, DDL, ACL) and the query texts of P1-21 (reader),
and answers the six questions of `docs/plans/connected-code-graph-p1.md` §9, row N-GRAPH. It is a successor file
because v1 has been executed against (root `AGENTS.md`, collision rule 5).

Inputs it agrees with: `docs/plans/connected-code-graph-answer-spec2.md` (§3 hop kinds and `Bound.launch`, §5 flow
fallback, §9 R-2), `docs/plans/connected-code-graph-bundle-spec5.md` (§1 version rule, §6.3, §7, §8 "Graph note" row),
`docs/plans/connected-code-graph-bundle-spec3.md` §1.6, §2.2, §3.2, §5.15–§5.18, §8.5, §9, and
`docs/plans/connected-code-graph-rule-pack-spec.md` §1.6 (`launch` block), §1.8, §4.3–§4.4. §9 lists what this note
needs from the answer side and the edits other documents owe.

CPG: considered, not relevant — a graph-model note; no loaded `cpg_*` graph indexes `cpg/`, and the subject is the
`ccg_<system>` store, read from the slice loader `cpg/spikes/vertical-slice/load.py` and `impact.py` directly.

**Evidence.** The DDL, edge groups, reader texts and ACL users below were run on throwaway `v4.18.11` instances
(module `41811`), and `bundleHash` on the stored slice bundles; the facts pinned are the tests of §8.

## 1. Model deltas (question 1)

`graphModelVersion` is **`"1.1"`** (a new `:Snapshot` property, §3). Rows are additions to v1 §1.2 unless they say
"replaces".

| # | Element | 1.1 rule |
|---|---|---|
| G1 | `ConfigEntry`, `configEntry` origin | `(:ConfigEntry {entryId, repoId, unitId, path, line, keyPath, valueHash})`. `entryId` is the record id `["configEntry", repoId, path, keyPath]` from the raw `keyPath`; `keyPath` is stored masked (v1 §1.5 P2-8). `unitId` is the bundle header's (every config file belongs to a unit, P1-44). **Not carried:** `format`, `bindValue`, `example`, `bindParts`. No `File`→`ConfigEntry` edge: `path` is denormalized. |
| G2 | `ConfigEntry`, `launchFact` origin | `(:ConfigEntry {entryId, repoId, unitId, path, line, launchKind, protocol?, port?, envVar?, mountPath?, service?, argv0?})`, per-kind exclusivity as `-bundle-spec3.md` §1.6. `entryId` is the record id `["launchFact", repoId, path, kind, ordinal]`; the type tag keeps it disjoint from G1 ids under the one `UNIQUE` constraint. **No `keyPath`** (the record has none; `-answer-spec2.md` §3.2). The assembler writes exactly one of `keyPath` and `launchKind`, so a reader tells the origins apart by which is present. Not carried: `args`, `default`, `defaultHash`, `origin`, `ordinal` (v1 §1.5 G-1 stands: no role-relevant argument pairs). |
| G3 | `Dependency` | As v1. "Nodes only" means **no traversal row** before P3 (`-answer-spec2.md` §3, V19); `DEPENDS_ON` is still loaded, because P1-18's equivalence test reproduces the slice, which loads it. |
| G4 | `Unit` | `(:Unit {unitId, repoId, rootPath, language, buildStatus, flowAvailable, builtCommit?, builtTree?, builtAt?, bundleSchemaVersion?, failedAt?, failReason?})`. `fresh` carries the four header fields from its bundle; `stale-kept` carries them from the **last good** bundle plus `failedAt` (RFC 3339 UTC `Z`) and `failReason`; `not-mapped` carries none of them and has `failReason`. **`failReason` is value-free** (`-bundle-spec5.md` §6.3, §8): one code of the closed set `FAIL_REASONS` in `cpg/ccg/staleness.py` (created by P1-18, whose assembler refuses any other code; P1-20 adds `unit_staleness` there and sets reasons only from the constant): `manifest:never-built`, `build:shallow-clone` (D1), `extract:failed`, `extract:timeout`, `bundle:invalid`, `bundle:schema-minor`, `bundle:bind-key-check`, `bundle:audit-missing` (P1-18 (7)). A new code is an edit of that constant and of this row. Never exception text, source text or a configuration value; never indexed. A not-mapped unit's `unitId` comes from the manifest's unit discovery (P1-03). |
| G5 | `Unit.flowAvailable` | **Always present, boolean**, written by the assembler from `cpg/ccg/registry/flow-availability.yaml` (index D2) by `Unit.language`. A language with no row (Kotlin; `config`, `markdown`) is written `false`. The loader refuses a `Unit` row without it, so no reader coalesces an absent flag (`-answer-spec2.md` §5's "missing counts as false" stays a fail-closed backstop). |
| G6 | `MOUNTED_BY` (replaces v1 D5's props) | `(:Symbol)-[:MOUNTED_BY {ord, mountPath}]->(:CallSite)`. `mountPath` is the normalized mount path the pack's `mounts` or `mountPaths` rule read (rule-pack §1.6), an ungated path rendered through F-4's screen. For a **unit-level** mount (`mountPaths`, e.g. FastMCP's `/mcp` at `app.py:543`) every provider of that unit whose kind the rule serves gets one edge to the mount call site. `ord` by (`mountPath`, line). |
| G7 | `BOUND_BY` (replaces the graph note's `{ord}`) | `(:Link)-[:BOUND_BY {siteId, ord, port?, portOrigin?, packId?}]->(:ConfigEntry)`. **Per consumer site**, as `LIFTED_VIA` is (v1 D1): `siteId` is the `callSiteId` of the `VIA` site the chain binds, `ord` is 0-based per (link, site) in chain order. One edge per (link, site, entry), at the entry's lowest `ord`. Target is a `ConfigEntry` only (a mount is `MOUNTED_BY`, G6). `port`, `portOrigin`, `packId`: §2. **No hash of any kind is an edge property.** |
| G8 | `LAUNCHES` | `(:ConfigEntry)-[:LAUNCHES]->(:File)` and `(:ConfigEntry)-[:LAUNCHES]->(:Unit)`, one edge per target the linker resolves for a launch or Compose entry (§2). Loaded as a fact; **not traversed by `bfs-v1` 1.1** (`-answer-spec2.md` §3, V19). |
| G9 | `TO_UNIT` | `(:Link)-[:TO_UNIT]->(:Unit)` for a link into a never-built unit (graph note §6.4 FR-7a row): the `Link` has no `TO`, its `status`/`reason` are `possible`/`target-not-mapped`, the edge has no property. Its answer row is `-answer-spec2.md` §3.5. |
| G10 | `Pack` (new label) | `(:Pack {packId, version, contentHash})`, one node per pack the build ran, no edges, no index (§3). |
| G11 | `Param`, `Ret`, `Cond` | **No denormalization.** A point's `repo`, `unit` and `path` are its owner's, read by the owner join (`-answer-spec2.md` §6, the §4.1 row; §5 `owner_of`). |
| G12 | `Link` | Props as v1 (`linkId, kind, status, reason, matcher`). `paramMap` is **not stored** in 1.1: no P1 row reads it and a map is not a FalkorDB property value. The assembler counts it in `dropped["link-param-map"]`; the row that needs it adds it as a flat list, as `ownedRegions` is. |
| G13 | `header` | Not carried, beyond v1 BF13: `bindKeyCheck`. `bundleSchemaVersion` is carried per unit (G4) and as the snapshot maximum (§3). |

**Bindings by parts.** A link bound by `bindParts` is written exactly as one bound by clear fields: same `Link`
and `VIA` props, same `BOUND_BY` targets and props, no new reason (`-bundle-spec5.md` §7). Nothing derived from
`bindParts` (neither a hash nor its clear `host`) becomes a node property, edge property or index.

## 2. Bindings, O-9 and the framework-default port (question 2)

**One representation for every binding (URL, env/Compose, `.mcp.json` launch chain; index D10).** For each consumer
site the linker binds, it writes the chain it used as `BOUND_BY {siteId, ord}` edges from the `Link` to each
`ConfigEntry`, `ord` 0 being the entry nearest the consumer (`-answer-spec2.md` §3.2). Each chain entry whose target
the linker resolved also has its `LAUNCHES` edge (G8).

| Binding | `BOUND_BY` targets, by `ord` | `LAUNCHES` |
|---|---|---|
| URL in config (rule-pack §4.4) | the consumer URL entry (`.mcp.json` `mcpServers.<s>.url`); the port's source entry; the command launch that names the unit (one edge when they are the same entry) | command launch → the unit (`targetArg`) |
| Env/Compose (P1-17 (5)) | the consumer's base-URL entry; the Compose `services.<svc>.environment.<VAR>` entry; the port entry; the `services.<svc>.build.context` entry | `build.context` entry → the resolved unit (system-wide unit index; the context may lie in another repo) |
| `.mcp.json` launch chain (P1-17 (6), AC-4) | the server's `command` launch entry; the script's `exec`/`docker run` launch entry; the container's `CMD`/`ENTRYPOINT` launch entry | each entry → the next target: script `File`, then the unit |

**Port evidence.** The port a binding matched lives on the `BOUND_BY` edge of the entry that supplied it, as `port`
(int 1–65535) and `portOrigin` ∈ {`explicit`, `env-default`, `compose`, `pack-default`} (rule-pack §1.6's origins).
It is the port's only graph home (G2 carries neither `args` nor an env default's text); it is written only when the
linker read it through the gates.

**Framework-default port (O-9's second half).** A pack default applies only to a visible `command` launch whose
`command` matches and whose `args` hold no port flag (rule-pack §1.6 `frameworkDefaults`). That command launch is the
graph element: its `BOUND_BY` edge carries `port`, `portOrigin: 'pack-default'` and `packId` (a loaded
`Pack.packId`). No pack-fact node is introduced. `packId` is present iff `portOrigin` is `pack-default`.

**O-9 ruling (the reason on a bound site and its `Link`, any kind, `mcp_tool` included).**
- `VIA.reason` is `binding-confirmed` **iff** that `VIA` is `confirmed` and the link has ≥ 1 `BOUND_BY` edge with
  that site's `siteId`; a confirmed site with no binding keeps `key-match`; a `possible` site keeps the reason P1-17
  assigns (`example-config`, `port-shared`, …).
- `Link.status` stays best over its sites (v1 D2). `Link.reason` is `binding-confirmed` iff some `confirmed` site is
  `binding-confirmed`; otherwise it is the reason of the site that sets the best status, ties broken by lowest
  `siteId`.
- Why: for `mcp_tool` the key carries no server identity (bundle spec §6), so when a site has a binding, the binding,
  not the key, selected the provider; a rule over stored edges is a checkable invariant (§8). Rule-pack spec §4.4's
  "`confirmed`, `key-match` … with `BOUND_BY`" reads as `binding-confirmed`.
- `reasons.yaml`'s `binding-confirmed` row gains `VIA` in its `on` list (P1-17's edit).

## 3. Snapshot identity, pack set, schema version (question 3)

```
(:Snapshot {snapshotId, systemId, graphModelVersion: '1.1', bundleSchemaVersion, vocabVersion,
            manifestHash, packSetHash, codeHash, auditSetHash, builtAt?,
            verdictLogOffset: 0, verdictLogHash: 'c482b1cf54631b37eb4572dc86a68fb1'})
(:Pack {packId, version, contentHash})
```

- **`snapshotId`** keeps the answer spec v1 §4.2 form, `tuple_id(["snapshot", systemId, manifestHash])`.
- **`manifestHash`** = lowercase hex SHA-256 of the canonical JSON (`json.dumps(obj, ensure_ascii=False,
  separators=(",", ":"), sort_keys=True)`, UTF-8) of
  ```
  {"graphModelVersion": "1.1", "packSetHash": <packSetHash>, "codeHash": <codeHash>,
   "units": [[unitId, buildStatus, bundleHash|"", flowAvailable, failReason|""], ...]}
  ```
  `units` sorted by `unitId`, one row per `Unit` node, not-mapped units included, `flowAvailable` the JSON boolean.
- **`bundleHash`** (per unit; a `manifestHash` input only, **never stored**, since it is computed over `bindParts`
  and `bindKeyCheck`, which never become graph properties) = lowercase hex SHA-256 over the bundle's lines, each
  re-serialized with `ccg.bundle.canonical_line`, the header line first with the `-bundle-spec3.md` §8.5 exempt
  fields removed (today `builtAt` and `sandboxAuditIds`; the rule cites §8.5, so a later exempt field follows it).
  Bundles are canonical by §8.1, so this equals hashing the stored bytes minus the two fields. It covers every
  record, `builtCommit`, `builtTree`, `bundleSchemaVersion`, `tools` (Joern, frontend, normalizer, extract-script
  hash), `flowProjected`, `fieldDepthCap`, `vendoredDirs` and `skippedFiles`: any extractor or normalizer change that
  changes facts changes the bundle, hence the stamp. Checked on both stored slice bundles: an exempt-only edit keeps
  the hash, a one-character `tools.normalizerVersion` edit changes it.
- **`codeHash`** = SHA-256 hex over the source files of the modules that turn bundles and packs into rows, sorted by
  relative POSIX path, each contributing `path + "\0" + bytes + "\0"`, plus `cpg/ccg/registry/reasons.yaml`. The
  list is one constant, `ASSEMBLY_CODE` in `cpg/ccg/assemble/snapshot.py` (expected:
  `ccg/{assemble,linker,packrt}/**/*.py`, `ccg/{ids,keynorm,rulepack,bundle,valuehash,sysname}.py`). A test runs the
  assembler on a fixture bundle set in a fresh `sys.executable -I` subprocess and fails if any `ccg.*` module then in
  `sys.modules` is missing from `ASSEMBLY_CODE` (lazy imports included, independent of test order), so the list
  cannot fall behind the import graph. `ccg/engine/**` (covered by
  `recipe.algorithm.version`), `ccg/front/**` and the store are not in it. `flow-availability.yaml` is not either:
  its effect enters through the hashed `flowAvailable` values, so a registry edit that changes no loaded flag
  changes no `snapshotId`, and one that flips any loaded unit's flag does (`-answer-spec2.md` §9 R-2).
- **`packSetHash`** (O-10) = `tuple_id(["packSet", id₁, version₁, contentHash₁, id₂, …])`, packs sorted by `packId`;
  `contentHash` = SHA-256 hex over the pack directory's files by the `codeHash` byte rule. The `Pack` nodes carry the
  same triples for raw clients (`MATCH (p:Pack) RETURN p.packId, p.version, p.contentHash ORDER BY p.packId`).
- **`bundleSchemaVersion`** (`-bundle-spec5.md` §8 (4)): **the highest minor loaded**, minors compared as integers;
  the per-unit value is on `Unit` (G4) and inside `bundleHash`. A bundle of a major other than `1` is refused for its unit (`bundle:invalid`).
- **`auditSetHash`** = SHA-256 hex of the sorted, de-duplicated `sandboxAuditIds` of every loaded bundle joined by
  `\n`. Provenance **outside** the identity (audit ids are fresh per run, §8.5); amends v1 BF15's "`manifestHash`
  covers it".
- **`verdictLogOffset`/`verdictLogHash`** are stored with the P1 values `0` and `EMPTY_VERDICT_LOG_HASH`; the engine
  reads them from the node and fails if they differ from those values before P4.
- **`builtAt`** = the maximum `builtAt` of the `fresh` units, absent when none is fresh; outside the identity and,
  for 1.1, outside the facts (`-answer-spec2.md` §4.3). `vocabVersion` is the rule-pack core vocabulary version the packs declare (one
  value; a mix is refused).
- **Empty system.** The assembler refuses a snapshot in which no unit is `fresh` or `stale-kept`: there is nothing to
  answer from, and `bundleSchemaVersion` would have no value.

## 4. DDL and load groups

`cpg/ccg/store/ddl_v1.cypher` gains, in its existing order discipline (part 1 before the load, part 2 after):

```
// PART 1 additions (20 indexes in all)
CREATE INDEX FOR (n:ConfigEntry) ON (n.entryId);
CREATE INDEX FOR (n:ConfigEntry) ON (n.path);
// PART 2 addition (12 constraints in all)
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE ConfigEntry PROPERTIES 1 entryId;
```

`ConfigEntry.entryId` is EP §5.2.1 HA-1's lookup and every `BOUND_BY`/`LAUNCHES` load anchor; `ConfigEntry.path` is
the config-anchor read (§5). `Pack` and `Snapshot` stay unindexed. The file's "Deferred" comment keeps
`SchemaEntry`/`Mark`/`Finding`/`LogTemplate`, the `READS`/`WRITES`/`DISMISSED` relationship indexes and the full-text
index. `cpg/tests/test_ddl_live.py` checks the file against this section plus v1 §1.6.

**Loader tables** (port of `load.py`): `ID_PROP` gains `ConfigEntry: entryId` and `Pack: packId`; `NODE_ORDER` gains
`ConfigEntry` before `Link` and `Pack` last before `Snapshot`. The edge groups are a **closed list**; a row in any
other group fails the build before any write:

- v1's 29 groups (`load.py` `EDGE_ORDER`), unchanged;
- `(Link, BOUND_BY, ConfigEntry)`, `(ConfigEntry, LAUNCHES, File)`, `(ConfigEntry, LAUNCHES, Unit)`,
  `(Symbol, MOUNTED_BY, CallSite)`, `(Link, TO_UNIT, Unit)`.

**Excluded in P1, known refusals:** `(CallSite|ConfigEntry)-[:PROVIDES]` (v1 D4 admits them), `(ConfigEntry)-[:CONSUMES]`,
and `(ConfigEntry)-[:WRITES]`/any `READS`/`WRITES` (`-bundle-spec3.md` §9.1). No P1 pack emits them; a pack that does
fails the build in P1-16/P1-41 with the group named, and the phase that adds a producer adds the group here.

Edge statements keep the slice form (`UNWIND $rows AS r MATCH (a:<L> {<id>: r.s}) MATCH (b:<L> {<id>: r.t})
CREATE (a)-[e:<T>]->(b) SET e = r.p`), so every group's endpoints are matched by an index that exists before the
load. `BATCH_MAX`, the per-query `timeout`, zero retries and the created-count check are v1 §4's.

## 5. Reader queries, carry-over C3 (question 4)

The texts P1-21 implements, named by the recipe pattern `^[a-z_]+(:[A-Za-z]+)?$`. Rules for all: one statement,
parameters only (`$ids` a list of ≤ `CHUNK` ids, halved on a full reply; `$pairs` a list of `[a, b]` string pairs),
run with `GRAPH.RO_QUERY` as `ccg_engine_<s>`; a reply of `RESULTSET_SIZE` rows raises; no `OR`, no `IS NULL OR`, no
function around an indexed property, no endpoint label on a relationship-index anchor (quirks file). D8 defaults are
coalesced in the query, nowhere else. A `<Label>` in a name is a fixed token from the set the row gives, chosen by the
engine, never caller text; `<idProp>` is `callSiteId` for `CallSite` and `pointId` for `Param`, `Ret`, `Cond`.

**Anchors** (v1 §1.3–§1.4; the three location reads run in that order):

| Name | Text |
|---|---|
| `repo_exists` | `MATCH (r:Repo {repoId: $repo}) RETURN r.repoId` |
| `file_by_path` | `MATCH (f:File {path: $path}) WHERE f.repoId = $repo RETURN f.fileId, f.unitId` |
| `skipped_by_path` | `MATCH (s:SkippedFile {path: $path}) WHERE s.repoId = $repo RETURN s.unitId, s.reason` |
| `symbols_of_file` | `MATCH (f:File {path: $path})-[:DEFINES]->(s:Symbol) WHERE f.repoId = $repo RETURN s.symbolId, s.qname, s.lineStart, s.lineEnd, s.ownedRegions, s.depth` |
| `symbol_by_qname` | `MATCH (s:Symbol {qname: $qname}) WHERE s.repoId = $repo RETURN s.symbolId, s.unitId, s.path, s.lineStart, s.lineEnd, s.ownedRegions` |
| `symbol_by_name` | `MATCH (s:Symbol {nameNorm: $name}) RETURN s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart` |
| `config_by_path` | `MATCH (e:ConfigEntry {path: $path}) WHERE e.repoId = $repo RETURN e.entryId, e.unitId, e.line, e.keyPath, e.launchKind` — the engine drops rows with a `launchKind` (a launch entry is never an anchor, `-answer-spec2.md` §3.2); a query filter would need `IS NULL` |
| `cp_by_key` | `MATCH (cp:ContractPoint {keyNorm: $key}) RETURN cp.cpId, cp.kind, cp.keyNorm` |
| `cp_keys` | `MATCH (cp:ContractPoint) WHERE cp.kind = $kind RETURN cp.cpId, cp.keyNorm` — **label scan by design**: the client-side prefix/template match needs every key of the kind (graph note §3.4), and `ContractPoint` is small (61 in the slice) |
| `snapshot` | `MATCH (s:Snapshot) RETURN s.snapshotId, s.verdictLogOffset, s.verdictLogHash, s.graphModelVersion, s.bundleSchemaVersion` — label scan of one node |
| `units` | `UNWIND $ids AS i MATCH (u:Unit {unitId: i}) RETURN u.unitId, u.repoId, u.rootPath, u.language, u.buildStatus, u.builtCommit, u.builtAt, u.failedAt, u.failReason, u.flowAvailable` |

**Owner join (C3 item 1).**

```
owner_of:<Label>   UNWIND $ids AS i MATCH (s:Symbol)-[:<HAS_REL>]->(p:<Label> {<idProp>: i})
                   RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, s.lineEnd, s.ownedRegions
```

with (`<Label>`, `<HAS_REL>`) ∈ {(`CallSite`, `HAS_CALL`), (`Param`, `HAS_PARAM`), (`Ret`, `HAS_RETURN`), (`Cond`,
`HAS_COND`)}. A point's own `line` comes from the expansion that reached it.

**Expansions, impact direction (`←`).** Each returns the arriving edge's `status` and `reason` (C3 item 3):

| Name | Text |
|---|---|
| `calls_in` | `UNWIND $ids AS i MATCH (c:CallSite)-[e:CALLS]->(s:Symbol {symbolId: i}) RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, coalesce(e.status, 'confirmed'), coalesce(e.reason, 'resolved-call')` |
| `name_in` | `UNWIND $ids AS i MATCH (c:CallSite)-[:CALLS_NAME]->(:Name)-[:RESOLVES_TO]->(s:Symbol {symbolId: i}) OPTIONAL MATCH (c)-[r:CALLS]->(s) WITH i, c, s, r WHERE r IS NULL OPTIONAL MATCH (c)-[d:DISMISSED]->(s) WITH i, c, d WHERE d IS NULL OPTIONAL MATCH (c)-[x:CALLS]->() RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, count(x)` — the compound name hop (graph note §6.1 item 6): the resolved pair is suppressed, `DISMISSED` anti-joined (absent in P1; an absent type is not an error), and `count(x) > 0` means the site is resolved, so the candidate is `possible`/`resolution-unverified`, else `possible`/`name-match` |
| `data_in:<Label>` (`Param`, `CallSite`) | `UNWIND $ids AS i MATCH (u)-[d:DATA]->(p:<Label> {<idProp>: i}) RETURN i, labels(u)[0], coalesce(u.callSiteId, u.pointId), u.line, d.fromPath, d.toPath, coalesce(d.status, 'confirmed'), coalesce(d.reason, 'intra-flow')` |
| `controlled_in` | `UNWIND $ids AS i MATCH (p)-[:CONTROLLED_BY]->(k:Cond {pointId: i}) RETURN i, labels(p)[0], coalesce(p.callSiteId, p.pointId), p.line` |
| `to_in` | `UNWIND $ids AS i MATCH (l:Link)-[:TO]->(s:Symbol {symbolId: i}) RETURN i, l.linkId, l.kind, l.status, l.reason, l.matcher` |
| `via_in` | `UNWIND $ids AS i MATCH (c:CallSite)-[v:VIA]->(l:Link {linkId: i}) RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, v.status, v.reason, v.consumerKey` |
| `consumes_in` | `UNWIND $ids AS i MATCH (c:CallSite)-[e:CONSUMES]->(cp:ContractPoint {cpId: i}) RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, e.status, e.reason` |
| `on_in` | `UNWIND $ids AS i MATCH (l:Link)-[:ON]->(cp:ContractPoint {cpId: i}) RETURN i, l.linkId, l.kind, l.status, l.reason, l.matcher` |
| `provides_in` | `UNWIND $ids AS i MATCH (s:Symbol)-[e:PROVIDES]->(cp:ContractPoint {cpId: i}) RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, e.status, e.reason` |
| `bound_in` | `UNWIND $ids AS i MATCH (l:Link)-[b:BOUND_BY]->(e:ConfigEntry {entryId: i}) RETURN i, l.linkId, l.kind, l.status, l.reason, b.siteId, b.ord` (config change → the sites it binds) |

**Expansions, depends-on direction (`→`):**

```
calls_out              UNWIND $ids AS i MATCH (c:CallSite {callSiteId: i})-[e:CALLS]->(s:Symbol) RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, coalesce(e.status, 'confirmed'), coalesce(e.reason, 'resolved-call')
name_out               UNWIND $ids AS i MATCH (c:CallSite {callSiteId: i})-[:CALLS_NAME]->(:Name)-[:RESOLVES_TO]->(s:Symbol) OPTIONAL MATCH (c)-[r:CALLS]->(s) WITH i, c, s, r WHERE r IS NULL OPTIONAL MATCH (c)-[d:DISMISSED]->(s) WITH i, c, s, d WHERE d IS NULL OPTIONAL MATCH (c)-[x:CALLS]->() RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, count(x)
has_out:CallSite       UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_CALL]->(p:CallSite) RETURN i, p.callSiteId, p.line, p.calleeName
has_out:Param          UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_PARAM]->(p:Param) RETURN i, p.pointId, p.line, p.index, p.name
has_out:Ret            UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_RETURN]->(p:Ret) RETURN i, p.pointId, p.line
has_out:Cond           UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_COND]->(p:Cond) RETURN i, p.pointId, p.line, p.code
data_out:<Label>       UNWIND $ids AS i MATCH (p:<Label> {<idProp>: i})-[d:DATA]->(t) RETURN i, labels(t)[0], coalesce(t.callSiteId, t.pointId), t.line, d.fromPath, d.toPath, coalesce(d.status, 'confirmed'), coalesce(d.reason, 'intra-flow')      <Label> ∈ {CallSite, Ret, Cond}
controlled_out:<Label> UNWIND $ids AS i MATCH (p:<Label> {<idProp>: i})-[:CONTROLLED_BY]->(k:Cond) RETURN i, k.pointId, k.line, k.code      <Label> ∈ {CallSite, Ret, Cond, Param}
consumes_out           UNWIND $ids AS i MATCH (c:CallSite {callSiteId: i})-[e:CONSUMES]->(cp:ContractPoint) RETURN i, cp.cpId, cp.kind, cp.keyNorm, e.status, e.reason
provides_out           UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[e:PROVIDES]->(cp:ContractPoint) RETURN i, cp.cpId, cp.kind, cp.keyNorm, e.status, e.reason
via_out                UNWIND $ids AS i MATCH (c:CallSite {callSiteId: i})-[v:VIA]->(l:Link) RETURN i, l.linkId, l.kind, l.status, l.reason, l.matcher, v.status, v.reason, v.consumerKey
to_out                 UNWIND $ids AS i MATCH (l:Link {linkId: i})-[:TO]->(s:Symbol) RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart
to_unit_out            UNWIND $ids AS i MATCH (l:Link {linkId: i})-[:TO_UNIT]->(u:Unit) RETURN i, u.unitId, u.repoId, u.rootPath, u.buildStatus      (`-answer-spec2.md` §3.5)
```

**`Link` evidence (C3 item 2) and the derived N.** Per (link, consumer site), except `bound_of_link`:

```
lifted_of        UNWIND $pairs AS q MATCH (l:Link {linkId: q[0]})-[r:LIFTED_VIA]->(t) WHERE r.siteId = q[1]
                 RETURN q[0], q[1], r.ord, labels(t)[0], coalesce(t.callSiteId, t.symbolId), t.repoId, t.unitId, t.path,
                        r.line, r.urlPath, r.verbPath, r.viaSiteId, r.viaLine, coalesce(t.calleeName, t.qname)
                 ORDER BY q[0], q[1], r.ord                                   ($pairs = [linkId, siteId])
bound_out        UNWIND $pairs AS q MATCH (l:Link {linkId: q[0]})-[b:BOUND_BY]->(e:ConfigEntry) WHERE b.siteId = q[1]
                 RETURN q[0], q[1], b.ord, e.entryId, e.repoId, e.unitId, e.path, e.line, e.keyPath, e.launchKind,
                        e.protocol, e.port, e.envVar, e.mountPath, e.service, e.argv0, b.port, b.portOrigin, b.packId
                 ORDER BY q[0], q[1], b.ord                                   ($pairs = [linkId, siteId])
bound_of_link    UNWIND $ids AS i MATCH (l:Link {linkId: i})-[b:BOUND_BY]->(e:ConfigEntry)
                 RETURN i, e.entryId, b.siteId, b.ord, e.repoId, e.unitId, e.path, e.line, e.keyPath, e.launchKind,
                        b.port, b.portOrigin, b.packId
                 ORDER BY i, e.entryId, b.siteId, b.ord                       (-answer-spec2.md §3.3 wiring items)
candidate_count  UNWIND $pairs AS q MATCH (c:CallSite {callSiteId: q[0]})-[v:VIA]->(l:Link {linkId: q[1]})
                 MATCH (s:Symbol)-[:HAS_CALL]->(c) MATCH (s)-[:HAS_CALL]->(:CallSite)-[v2:VIA]->(l2:Link)
                 WHERE v2.consumerKey = v.consumerKey RETURN q[0], q[1], count(DISTINCT l2)   ($pairs = [siteId, linkId]; v1 D6, self included)
```

Each `bound_out` row is a `Bound` of `-answer-spec2.md` §3.2, port fields included. For `bound_of_link` the engine
keeps, per (`i`, `entryId`), the first row with a `keyPath` (launch-origin entries are not items): the order makes
spec2 §3.3's lowest-`(siteId, ord)` rule a first-row pick. The `keyPath` filter stays in the engine: a `WHERE` on `e`
moves the anchor to a `ConfigEntry` label scan (verified), the unfiltered text starts at the `Link` index scan.
`LAUNCHES` and `MOUNTED_BY` have no reader text in 1.1 (no hop carries them; spec2 §3.4). These texts give a raw
re-run of `recipe.queries` every hop's id, edge `status`/`reason`, `repo`/`unit`/`path`/`line` and per-site lifted
rows, enough to derive the item set (C3, AC-16).

## 6. `Link` answer size and the memory re-estimate (question 5)

**Answer size** (answer spec v1 §10 item 5). The bytes driver is per-site evidence copied into every item: a contract
item's chain carries one `Link` hop with that site's `lifted[]` and `bound[]`. P1-19 reports the graph-side drivers on
the loaded proving repo, and P1-39 the bytes:

1. Sites per link and per contract point:
   `MATCH (c:CallSite)-[:VIA]->(l:Link) WITH l, count(c) AS n RETURN max(n), avg(n), percentileDisc(n, 0.95)`;
   `MATCH (cp:ContractPoint)<-[:ON]-(:Link)<-[:VIA]-(c:CallSite) WITH cp, count(c) AS n RETURN cp.keyNorm, n ORDER BY n DESC LIMIT 10`.
2. Evidence per (link, site): `MATCH (l:Link)-[r:LIFTED_VIA]->() WITH l, r.siteId AS s, count(r) AS n RETURN max(n), avg(n)`;
   the same over `BOUND_BY`.
3. P1-39: for every `ContractPoint`, the canonical bytes of its contract-mode answer at the default bound
   (`len(ccg.bundle.canonical_line(answer))`), reported as p50/p95/max and the top 10 by bytes.

Prior (slice, not a measurement of P1): 11 links, one site each, 4–5 `LIFTED_VIA` rows, 0 `BOUND_BY`. **Decision
rule:** an answer above **1 MiB** of canonical bytes, or a contract-answer p95 above **256 KiB**, is reported to
`architect` as a size finding with the top items; P1 does not change the shape (v1 §10 item 5).

**Memory re-estimate for P1-19** (the ~18 GB peak of graph note §6.4), on the loaded proving system:

1. `INFO memory` `used_memory` before the first `CREATE INDEX` on the empty build key and after every constraint is
   `OPERATIONAL`; the delta is the snapshot's cost. `GRAPH.MEMORY USAGE <build key>` beside it (integer MB, a coarse
   cross-check, slice report §2.5).
2. Per language: load the language's units alone into scratch build keys (as `load.py:measure_unit_density` does).
   For `python`, `javascript` and `shell` report MiB, nodes and edges per `File` node of `kind` `code`/`script`; for
   `config` and `markdown`, per unit and per `ConfigEntry`/`File`. Report the share of `Link`, `ConfigEntry` and `DATA`.
3. Projection: `snapshot₁₀ₖ = 10,000 × Σ share_lang × MiB/file_lang` over code languages, plus the measured
   config/markdown share scaled by the same file ratio (a language P1 does not build uses python's density, flagged);
   `peak = snapshot₁₀ₖ × (1 live + 1 build + k overlays) × 1.2`, k = 2 (FR-23 default).
4. Report the proving system's own peak (live + build, k = 0 in P1) against `ccg-falkordb`'s `maxmemory` (1536 MB).

Priors to recompute: the slice's python unit measured 0.287 MiB per `File` node (`used_memory` delta,
`docs/test-reports/connected-code-graph-p0-slice-report.md` §2.5), so `snapshot₁₀ₖ` ≈ 2.9 GB and peak ≈ 13.9 GB, under
the ~18 GB estimate. The proving scope holds 331 Python and TS/JS tracked files (`git ls-files`, before the census
exclusions), so
its code part is ≈ 100 MB per snapshot and live + build fits the instance by an order of magnitude.

## 7. ACL users and the system-name whitelist (question 6)

**User names are per system: `ccg_<role>_<s>`, role ∈ {`build`, `verdict`, `engine`, `raw`}** (stakeholder decision,
2026-11-09). One instance hosts several systems (P1 builds the proving system and `cpg/fixtures/system1`), and each
system's credentials are created, rotated and stored independently (`~/.config/ccg/keys/<s>.acl` through
`ccg.atrest.write_acl`/`read_acl`, read by clients via `CCG_CREDENTIALS_DIR`; P1-02).

**The four users** (the `ccg_build_<s>` line replaces the graph note §2 Q10 / §6.4 `~ccg_<system>*` form):

```
ACL SETUSER ccg_build_<s> on ><secret> resetkeys resetchannels -@all
    +graph.query +graph.ro_query +graph.explain +graph.constraint +graph.config|get +rename +exists +info
    %W~ccg_<s> ~ccg_<s>__*
    (~ccg_<s>:verdict_lock +set +get +del +eval)
    (~ccg_<s>__build_* ~ccg_<s>__ov_* +del +expire)
ACL SETUSER ccg_verdict_<s> on ><secret> resetkeys resetchannels -@all +graph.query +info ~ccg_<s>
    (~ccg_<s>:verdict_lock +set +get +del +eval)
ACL SETUSER ccg_engine_<s> on ><secret> resetkeys resetchannels -@all +graph.ro_query +graph.explain +info ~ccg_<s> ~ccg_<s>__ov_*
ACL SETUSER ccg_raw_<s>    on ><secret> resetkeys resetchannels -@all +graph.ro_query +graph.explain +info ~ccg_<s> ~ccg_<s>__ov_*
```

- **The build credential holds the live key write-only (`%W~ccg_<s>`).** `RENAME <build> ccg_<s>` needs only write on
  the destination, so the swap works; every other path to the live key needs read on it and is refused: `GRAPH.QUERY`,
  `EXPIRE ccg_<s> 0`, `DEL`, an `EVAL`'d `DEL`, and `RENAME ccg_<s>` away (onto `__build_*` or `__ov_*`, where the
  selector's `+del` would then reach it). `+del` and `+expire` exist only in the selector over build and overlay keys,
  so the loader removes its own failed build key (P1-19 (5)) and P3's runner expires overlays. Nothing in P1 or the
  proposal expires or reads the live key with this credential (graph note Q4: overlays and build keys only).
- **Residual, by design:** `RENAME <build> ccg_<s>` and `RENAME <ccg_s__ov_*> ccg_<s>` overwrite the live graph with
  whatever the source key holds — the swap's own class; `EXISTS ccg_<s>` answers (harmless). The control is P1-02's
  credential custody; live is rebuildable from bundles. The pre-`RENAME` replay onto the build key runs as
  `ccg_build_<s>`; any post-`RENAME` work on the live key (v1 row 9's remainder
  replay and offset check, a no-op in P1) runs as `ccg_verdict_<s>`, the only credential that reads and writes the live
  key; P4 inherits that. `ccg_verdict_<s>` can rewrite live content but cannot `DEL`, `EXPIRE`, `RENAME` or touch build
  keys. The guarded-`EVAL` keep-`prev` swap form of v1 row 11 is **withdrawn** (it needs read on live; zero snapshots
  are kept).
- `+graph.config|get` lets the loader read `TIMEOUT_MAX` (v1 §4) while `GRAPH.CONFIG SET` is refused. No user holds
  `+graph.delete`, `+graph.list` or `+graph.profile`. All four hold `+info` (`falkordb-py`'s constructor).
- The grant matrix these lines produce is the test of §8 P1-19 (c). With the pattern alone, `ccg_shop___build_1` —
  the build key of a system named `shop_` — **is** granted to `ccg_build_shop`, which is why the whitelist forbids a
  trailing separator.

**The whitelist.** A system name `s` is valid iff it matches

```
^[a-z0-9]+(?:[-_][a-z0-9]+)*$        length 1–48
```

lowercase ASCII letters and digits, single `-` or `_` separators, none leading or trailing, no `__`. Isolation: for
systems `s ≠ t`, a key of `t` (`ccg_t`, `ccg_t__…`, `ccg_t:verdict_lock`) matches `ccg_s` or `ccg_s__*` only if `t`
begins with `s__` (no `__` allowed) or `t` = `s` + `_` (no trailing `_` allowed); the review's exhaustive probe over
every name to length 5 found 0 cross-system matches. It also excludes every glob metacharacter and `:`. User names
inherit it: `ccg_<role>_<s>` is unique per (role, system), because a role is a fixed token without `_`.

**Its home:** `cpg/ccg/sysname.py`, the one module holding `SYSTEM_NAME_RE`, `check_system_name(name) -> str`
(raises `SystemNameError(ValueError)`), `ROLES = ("build", "verdict", "engine", "raw")`, and the only key and user
constructors: `live_key(s)`, `build_key(s, snapshot_id)`, `overlay_key(s, ov_id)`, `lock_key(s)`, `acl_user(role, s)`
(→ `ccg_<role>_<s>`), `acl_user_re()` (the pattern a drift check uses, built from `ROLES` and `SYSTEM_NAME_RE`, so no
second regex exists). Each constructor calls `check_system_name`; a test fails if `ccg_` key or user text is
formatted in any Python file under `cpg/ccg/` or `cpg/clients/` other than `sysname.py`. `cpg/infra/` is out of the
sweep (its regex literal and test fixtures are deliberate). P1-03's `cpg/tests/test_sysname.py` covers it as a
**same-verdict** test, not string equality (ERE has no `(?:`): over every role × valid name to length 5 over
`{a, b, _, -}`, malformed names (`ccg_build`, `ccg_raw_`, `ccg_admin_x`, `__`, trailing `_`), `default`, and a 48- and
a 49-character system name, the script's check and `acl_user_re()` plus the 48-character bound agree on every name.
Neither regex holds the length, so the script checks it separately.

## 8. Tests this note requires

- **P1-18 (assembler).** (a) Exactly one of `keyPath`/`launchKind` on every `ConfigEntry` row; no `args`, `default`,
  `defaultHash`, `bindValue`, `bindParts`, `format`, `example`, `bindKeyCheck` key in any row (a scan of all keys).
  (b) `Unit.flowAvailable` present on every `Unit` row, from the registry; a language with no row gives `false`.
  (c) **O-9 invariants**, checked on `VIA` and `Link` rows: `VIA.reason == 'binding-confirmed'` iff the `VIA` is
  confirmed and has a `BOUND_BY` with its `siteId`; `Link.reason == 'binding-confirmed'` iff some confirmed site is.
  Planted case: one link, site A confirmed unbound, site B possible with a chain → link `confirmed`/`key-match`.
  Mutants: dropping the `siteId` condition; deriving `Link.reason` from "any `BOUND_BY`". (d) Every `BOUND_BY` row has
  `siteId` equal to a `VIA` site of its link; one edge per (link, site, entry); `packId` iff `portOrigin ==
  'pack-default'`, naming a loaded `Pack`. (e) `manifestHash` changes when only `header.tools.normalizerVersion`
  changes in one bundle (mutant: drop `bundleHash`), when one unit's `flowAvailable` flips, when a pack's bytes change
  and when an `ASSEMBLY_CODE` file changes; it does **not** change for a `flow-availability.yaml` edit that flips no
  loaded unit, nor between builds that differ only in `sandboxAuditIds` and `builtAt`. (f) The `ASSEMBLY_CODE`
  import-coverage test of §3. (g) Mixed `"1.0"`/`"1.1"` bundles give `Snapshot.bundleSchemaVersion == "1.1"` and
  per-unit values; a `"2.0"` bundle is refused; a system with no fresh or stale-kept unit is refused. (h) A link bound
  by `bindParts` and the same link bound by clear `bindValue` give identical rows. (i) `failReason`: a failing bundle
  carrying a planted `bindValue` and an exception message yields a `failReason` that is a member of
  `FAIL_REASONS` and contains neither; a code outside the set is refused by the assembler.
- **P1-19 (loader, live on a throwaway).** (a) `db.indexes()` equals §4 + v1 §1.6; 12 constraints `OPERATIONAL`.
  (b) A row in an edge group outside §4's list — including the "excluded in P1" groups — fails the build before any
  write. (c) The §7 grant matrix, `<s>` = `shop`: `ccg_build_shop` writes, indexes and constrains `ccg_shop__build_1`,
  `DEL`s/`EXPIRE`s build and overlay keys, `RENAME`s build → live, reads `TIMEOUT_MAX`, and holds the lock (`SET NX PX`,
  `GET`, compare-and-delete `EVAL`); it is refused `GRAPH.QUERY`/`RO_QUERY`, `EXPIRE … 0`, `DEL` and `RENAME`-away on
  `ccg_shop`, `GRAPH.CONFIG SET`, `GRAPH.DELETE`, `GRAPH.LIST`, and every key of `shop2`. `ccg_verdict_shop` holds
  the lock (`SET NX PX`, `GET`, compare-and-delete `EVAL`), reads the live `Snapshot` offset and patches it in one
  `GRAPH.QUERY`, and is refused `GRAPH.RO_QUERY`, `EXPIRE`, `DEL`, `RENAME` of live and build keys; `ccg_engine_shop` reads live through `falkordb-py` and is refused
  `GRAPH.QUERY`. The name `shop_` is refused by `check_system_name`, and the test shows the pattern alone would grant
  `ccg_shop___build_1`.
- **P1-21 (reader).** (a) Each text of §5 is the one in the code, character for character. (b) `GRAPH.EXPLAIN` of every
  expansion and anchor text contains `Node By Index Scan` and no `Label Scan`, except `cp_keys` and `snapshot`;
  **positive control:** `MATCH (s:Symbol) WHERE toLower(s.qname) = $q RETURN s.symbolId` must show `Node By Label
  Scan` (verified). The harness supplies parameter values: a `{path: $path}` text explained without them returns
  `Missing parameters`. (c) `name_in`/`name_out` on a resolved site with two same-name candidates return only the
  non-resolved one with `count(x) == 1`, and on an unresolved site all candidates with `0`. (d) `candidate_count`
  returns 2 for a site with two `VIA` edges of equal `consumerKey`. (e) `bound_out` for (link, site A) returns only
  A's chain when the link has a second site with its own chain (verified), and its entry columns map onto valid
  `Bound` objects for a mixed chain. (f) `bound_of_link` over a link whose entry binds two sites returns it once per
  edge in `(entryId, siteId, ord)` order, and the engine's pick is the lowest `(siteId, ord)` row with a `keyPath`.

## 9. Agreements other documents owe

The answer-side rulings (`-answer-spec2.md` §3.2 port fields, §3.5 `TO_UNIT`, §4.3 `builtAt`), the index's
`auditSetHash` exemption and the step texts §7 names are applied. Remaining (`architect`, exact wording):
- `docs/plans/connected-code-graph-p1-linking.md` P1-19 Files: in "the lock and the `RENAME build live` run as
  `ccg_build_<s>`", insert after "lock": ", the pre-`RENAME` replay onto the build key".
- `docs/plans/connected-code-graph-p1-extraction.md` P1-48 (2): replace "a test asserts it equals
  `ccg.sysname.acl_user_re()`" with "a test asserts it gives the same verdict as `ccg.sysname.acl_user_re()` and the
  48-character bound over the name set of `-p1-graph.md` §7".
