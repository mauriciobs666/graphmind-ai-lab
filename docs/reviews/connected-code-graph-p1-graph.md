# Review: connected code graph, graph model 1.1 for P1 (N-GRAPH)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Pass 1 (2026-11-09)

### Scope & verdict

**Reviewed:** `docs/plans/connected-code-graph-p1-graph.md` (uncommitted, first version). I checked it against
the six N-GRAPH questions (`docs/plans/connected-code-graph-p1.md` §9), D2/D8/D10 and §8 of the index, the
steps P1-03, P1-17, P1-18, P1-19, P1-21, P1-39 and P1-48, v1 (`-p0-graph.md`), the graph note §6.2, answer
spec v1 §6 and `cpg/ccg/schemas/answer-v1.schema.json`, `cpg/ccg/answer.py`, `cpg/ccg/registry/reasons.yaml`,
the slice loader `cpg/spikes/vertical-slice/load.py`, `-bundle-spec3.md` §2.2, §3, §8.5 and §9.1, and the
uncommitted P1-48 `cpg/infra/ccg-falkordb.sh`. `-answer-spec2.md` and `-bundle-spec5.md` are ungated, so I
read their claims as unverified inputs. `cpg/REVIEW.md` does not exist; root `REVIEW.md` applies.

**Live re-run:** a labelled throwaway `falkordb/falkordb:v4.18.11` (module `41811`, Redis 8.6.3) on
`127.0.0.1:6397`, removed afterwards. Ports 6379 and 6390 were not touched. Probe scripts are in the session
scratchpad and are not committed.

**Verdict: needs changes.** There is one blocker (F1): the snapshot identity does not cover the extractor
toolchain. Several majors are contract gaps between this note and `-answer-spec2.md` (F3–F5); each needs one
owner to change one document. Most of the note checks out: the DDL counts, the index-scan plans, the ACL
matrix and the whitelist's sufficiency all reproduced.

CPG: considered, not relevant — no loaded `cpg_*` graph indexes `cpg/`. The subject is a graph-model note
whose claims I checked against the slice loader source, the answer schema and validator, and a live throwaway
FalkorDB.

### Findings

#### F1 — blocker — `manifestHash` does not identify the extractor toolchain, so one `snapshotId` can stand for two different fact sets (§3)

*Verified by reading.* The `units` rows carry `[unitId, buildStatus, builtCommit, builtTree,
bundleSchemaVersion, flowAvailable]`. The bundle header also has `tools` (`joern`, `frontendVersion`,
`normalizerVersion`, `extractScriptHash`), plus `flowProjected`, `fieldDepthCap` and `vendoredDirs`
(`-bundle-spec3.md` §2.2). All of these decide facts, and none of them is hashed. `codeHash` cannot fill the
gap: it covers only `cpg/ccg/**/*.py`, while D8 and P1-01 put every extractor, normalizer, Joern script and
SCIP indexer under `cpg/extractors/` (P1-04…P1-15 Files lists).

So when a fixed normalizer or projection (P1-09) re-extracts the same commits, the facts change and the stamp
does not. That is the R-2 failure class (`-answer-spec2.md` §9) that §3 says it closes.

The rule also over-covers. `codeHash` includes `cpg/ccg/registry/*.yaml`, and that includes
`flow-availability.yaml` itself. A registry edit that changes no loaded flag therefore *does* change
`snapshotId`, which contradicts §3's own sentence and leaves test (e) without a check of that half. `front/`,
`answer.py` and `atrest.py` are also hashed, though they decide no graph fact.

**Suggested fix:**
- Add a per-unit `bundleHash` to each `units` row: SHA-256 of the bundle's canonical bytes with the two
  §8.5-exempt header fields (`builtAt`, `sandboxAuditIds`) removed. The bytes are deterministic by spec3 §8.4,
  and this covers `tools`, the flags and every record in one value.
- Narrow `codeHash` to what turns bundles into rows: `cpg/ccg/{linker,assemble,packrt,keynorm,ids}` plus
  `reasons.yaml`. `flow-availability.yaml` is already covered by the hashed flags.
- Add a test to P1-18 (e): change only `header.tools.normalizerVersion` in one bundle and `manifestHash` must
  change. The mutant that drops `bundleHash` turns it red.

#### F2 — major — `BOUND_BY` is per link, but bindings are per call site. The O-9 rule then mislabels links, and its `VIA` half cannot be checked (§2, G7)

*Inferred, traced across documents.* A `Link` is one per (consumer symbol, provider, key) (v1 BF6), and
several call sites reach it through `VIA`. Each site can be bound by a different chain: two calls in one
function, one reading `A_URL` and one reading `config.b_url`. `LIFTED_VIA` carries `siteId` for exactly this
reason (v1 D1). `BOUND_BY {ord, port?, portOrigin?, packId?}` has no `siteId`, which causes three problems:

1. Two chains interleave under one `ord` sequence. `bound_out … ORDER BY b.ord` cannot separate them.
2. **The rule mislabels a link.** Suppose site A is `confirmed`/`key-match` with no binding, and site B is
   `possible`/`port-shared` with a chain. The Link is `confirmed` (best over sites, v1 D2) and has a
   `BOUND_BY`, so §2 makes it `binding-confirmed`. That credits the binding with a confirmation it did not
   give.
3. "Each `VIA` follows the same rule for that site's own binding" cannot be evaluated over the graph. §8 (c)
   checks only `Link` rows. `reasons.yaml` also scopes `binding-confirmed` to `[Link]`, not `VIA`.

A related gap: a pack-default command launch is both "the port's source entry" and "the command launch". §2
does not say whether that gives one edge or two.

**Suggested fix:**
- Add `siteId` to `BOUND_BY`, as D1 did for `LIFTED_VIA`.
- State O-9 per site: `VIA.reason = binding-confirmed` iff that `VIA` is `confirmed` and has ≥ 1 `BOUND_BY`
  with its `siteId`. `Link.reason = binding-confirmed` iff a `confirmed` site of the link is
  `binding-confirmed`.
- Rule one edge per (link, site, entry), at the entry's lowest `ord`.
- Extend §8 (c) to `VIA` rows, with the A/B example above as its planted case. Add `VIA` to the registry row's
  `on`.
- Carry `siteId` into `Bound` (see F3).

#### F3 — major — `bound_out` does not return "exactly a `Bound`": `Bound` has no field for `port`, `portOrigin` or `packId` (§5)

*Verified.* `$defs.Bound` in `cpg/ccg/schemas/answer-v1.schema.json` is
`additionalProperties: false` with `{ord, entryId, repo, unit, path, line, keyPath}`. `-answer-spec2.md` §7.1
adds only `launch`.

This hits command launches hardest. A `command` launch may not carry `port` (the `Launch` per-kind rule), so
for an explicit `--port` or a pack default the edge is the only home of the port. That port is the evidence
§2 exists to keep, and the answer can never show it. P1-21 test (e) can pass only if the engine silently
drops the three columns.

**Owner:** `architect` (`-answer-spec2.md` §3.2 / §7.1).
- Option 1: add optional `Bound.port`, `Bound.portOrigin` and `Bound.packId` (plus `Bound.siteId`, F2). This
  is additive, V12-neutral, with `packId` iff `pack-default`.
- Option 2: rule that the port evidence stays graph-only. This note then drops "exactly" and states the
  projection.

This note must cite whichever is chosen.

#### F4 — major — `builtAt`: rebuild determinism, FR-19 and wall-clock `builtAt` cannot all hold

*Verified by reading.* Three statements conflict:

- `facts_bytes` keeps `staleness`, and `staleness.units[].builtAt` comes from `Unit.builtAt` (answer spec v1
  §6; `cpg/ccg/answer.py:508`).
- `Unit.builtAt` is the bundle header's wall-clock time, which is exempt from determinism (spec3 §8.5).
- P1-39 (3) requires byte-identical `facts_bytes` "across rebuilds from the same commits". The index §8
  requires identical snapshots across such rebuilds.

This note keeps `builtAt` out of the identity (§3). A rebuild therefore keeps the `snapshotId` but changes
the facts, so "same snapshot, same facts" (FR-19) breaks, and P1-39 (3) cannot pass unless bundles are
reused. Exactly one of the three must give.

**Owner:** `architect` (`-answer-spec2.md`, P1-39), with the stakeholder if FR-19's reading changes. The
options:
- (a) `staleness.units[].builtAt` leaves the facts (a minor bump).
- (b) `Unit.builtAt` becomes the commit time of `builtCommit`, which is deterministic. The rendered meaning
  shifts.
- (c) `builtAt` enters `manifestHash`. Index §8 and P1-39 (3) are then reworded to "same commits, same facts
  minus the stamp and `builtAt`".

This note follows whichever is chosen.

#### F5 — major — `TO_UNIT` has a reader text but no answer representation (G9, §5 `to_unit_out`)

*Verified.* `Unit` is not a `Hop.kind`, and `TO_UNIT` is not in the `edgeType` enum (schema).
`-answer-spec2.md` §3 neither adds nor prohibits it. V9 (`answer.py:669`) requires `staleness.units` to
equal exactly the units of items and chains. A never-built target unit is on no hop, so it cannot be listed
with its `not-mapped` reason, and that reason is what P1-39 (1) and FR-7a ask for.

**Owner:** `architect` (`-answer-spec2.md` §3). Either define the row (the depends-on walk ends at the `Link`
hop, which carries `target-not-mapped`, and V9's need set gains the `TO_UNIT` target), or prohibit it so
`to_unit_out` is not read in 1.1. This note then marks `to_unit_out` accordingly.

#### F6 — major — `+expire` on the live key lets the build credential delete it, so "DEL on live refused" guards nothing (§7)

*Verified live.* As `ccg_build_shop` with the §7 line:
- `DEL ccg_shop` and an `EVAL`'d `DEL` both return `NOPERM`, as the note says.
- `EXPIRE ccg_shop 0` returns `1`, and `EXISTS ccg_shop` is then `0`.
- `RENAME ccg_shop ccg_shop__ov_9` is allowed (`~ccg_shop__*`).

v1 row 10 called this exact outcome a defect ("flat `+del` lets `ccg_build` … delete the live graph"). P1-19
(c)'s "DEL of the live key refused" is a check that cannot fail on the path that matters.

**Suggested fix (verified):** move `+expire` into the build-key selector, `(~ccg_<s>__build_* +del +expire)`.
On the throwaway this gives `EXPIRE ccg_alt 0` → `NOPERM` and `EXPIRE`/`DEL` of `ccg_alt__build_1` → `OK`,
and `RENAME build live` still works. First confirm that no loader or swap path `EXPIRE`s the live key; the
proposal's G7 says only that the build credential is the one that may.

Then add "`EXPIRE` live 0 refused" to P1-19 (c), and name the remaining residual in §7: `RENAME` of the live
key away, and `GRAPH.QUERY` writes on live.

#### F7 — major — §5 is not the complete query set: the depends-on texts for `CALLS_NAME →` and `CONSUMES →` are missing

*Verified by reading.* Graph note §6.2's depends-on column traverses the `CALLS_NAME`+`RESOLVES_TO` compound
("as `CALLS`", so `→`) and `CONSUMES →` (the wiring-error hypothesis). P1-22 (2) implements those rows "as
written". §5 lists `calls_out … to_unit_out` with no `name_out` and no `consumes_out`. P1-21 (a) ("each text
of §5 is the one in the code") then fails, or the engine runs texts that were never published.

Two smaller gaps in the same section:
- `controlled_out:<Label>` names no label set. From the edge groups it is `CallSite`, `Ret`, `Cond`, `Param`.
- `has_out`'s "same shape" returns `p.pointId`, which is wrong for `CallSite` (`callSiteId`).

**Suggested fix:** add the two texts, with the same `DISMISSED` anti-join for `name_out`, and the explicit
label sets.

#### F8 — major — the per-system ACL names collide with P1-48's drift check, and that check also rejects names the whitelist allows

*Verified by script.* The uncommitted `cpg/infra/ccg-falkordb.sh:117` has
`ACL_ALLOWED_RE='^(default|ccg_build|ccg_verdict|ccg_engine|ccg_raw_[a-z][a-z0-9_]*)$'`:
- Every `ccg_build_<s>`, `ccg_verdict_<s>` and `ccg_engine_<s>` user from §7 counts as drift.
- So do `ccg_raw_fx-job` and `ccg_raw_1sys`, which the whitelist allows.

The stakeholder decision (Open question 1) sets which form `status` encodes. Either way, the user-name part
must be the whitelist itself, not a second, looser regex.

**Owner:** `devops` (P1-48), after the decision. For option A, this ERE passed `default`, `ccg_build_shop`,
`ccg_raw_fx-job` and `ccg_engine_1sys`, and flagged `ccg_build`, `ccg_build_shop_`, `ccg_build_a__b`,
`ccg_admin_shop` and `ccg_raw_`:

```
^(default|ccg_(build|verdict|engine|raw)_[a-z0-9]+([-_][a-z0-9]+)*)$
```

#### F9 — major — no step creates the whitelist's home, and two step texts still carry the old rule

*Verified by grep.*
- `cpg/ccg/sysname.py` appears in no step's Files list.
- P1-03 (1) still specifies "none of `*?[]\:`, no `__`". That rule admits `shop_`: my probe found 462
  cross-system key matches under it.
- P1-19 (2) plants `shop2` and `shop__x` but not `shop_`, and says "the four `ccg_*` users".
- The proposal §3.5 (`docs/plans/connected-code-graph.md`) states the weaker rule.
- The "no `ccg_` key text formatted elsewhere" test has no owner.

A `tdd-engineer` who implements P1-03 from its own text writes the weak rule.

**Owner:** `architect`, or `teco` if routing the edit. Put `sysname.py` and the format-sweep test in P1-03's
Files list, since P1-03 is the first importer. Make P1-03 (1) and P1-19 (2) cite `-p1-graph.md` §7. The
proposal wording follows the stakeholder decision.

#### F10 — minor — `Unit.failReason` omits `-bundle-spec5.md` §8's fifth agreement (G4)

*Verified by reading.* The spec5 §8 "Graph note" row requires `Unit.failReason` to be value-free (§6.3).
G4 says "free text" and §8 has no test for it. `failReason` is a graph property, and `raw_query` returns it
unscreened.

**Suggested fix:** state "value-free (spec5 §6.3): a rule id or reason code, never exception text". Add a
P1-18 test that plants a `bindValue` in a failing bundle and asserts that value is absent from
`failReason`.

#### F11 — minor — loose ends of the two `ConfigEntry` shapes (G1, G2, §5)

*Verified by reading.*
- G2 does not say where a launch-origin `entryId` comes from. By spec3 §3 it is the record id
  `["launchFact", repoId, path, kind, ordinal]`, and its tag keeps it from colliding under the shared
  `UNIQUE`.
- `config_by_path` returns launch-origin rows too, but `AnchorConfig` requires `keyPath` and a launch entry
  is "never an anchor" (`-answer-spec2.md` §3.2). The engine must drop rows with `launchKind`, and the note
  should say so, since a query filter would break the no-`IS NULL` rule.

#### F12 — minor — the closed edge list silently excludes producers that v1 and spec3 name

*Inferred.* v1 D4 allows `(:CallSite|:ConfigEntry)-[:PROVIDES]` and spec3 §9.1 has packs add
`(:ConfigEntry)-[:WRITES]`. §4's closed list fails the build on both. That is acceptable for P1, but it should
be stated as a P1 exclusion, so a pack that emits them is a known refusal rather than a surprise in
P1-16/P1-41.

#### F13 — minor — two §8 checks have no named positive control

*The EXPLAIN behaviour was verified live.*
- P1-21 (b) "contains `Node By Index Scan`, no `Label Scan`" needs a planted non-indexed text, such as
  `toLower(s.qname) = $q`, that must show a label scan.
- `GRAPH.EXPLAIN` of a `{path: $path}` text without a `CYPHER path=… repo=…` prefix returns
  `Missing parameters`. The `UNWIND $ids` texts do not. The test harness must supply parameter values.

#### F14 — nit — loose ends

- The empty-set cases are undefined: `Snapshot.builtAt` with no `fresh` unit, and `Snapshot.bundleSchemaVersion`
  with every unit `not-mapped`.
- The memory method's step 2 divides by code/script `File`s, which `config`/`markdown` units do not have.
- Index §8's "identical snapshots minus timestamps" should also exempt `auditSetHash` (owner: `architect`).
- G3 restates v1's `DEPENDS_ON` shape where a citation would do.

### Growth-stops check

This is the first version of the document, so there is no rework growth to classify. One restated fact (G3)
is listed under F14. The note states the design without finding→resolution tables. Because v1 has been
executed against, a successor with `Extends:` is the correct form, and v1's header carries the matching
`Extended by:`.

### What's solid (re-run on the throwaway)

- **DDL:** v1's 18 indexes plus the two `ConfigEntry` indexes give 20 (`db.indexes()`). 12 constraints reach
  `OPERATIONAL`.
- **Index-scan claim:** I checked a sample of 18 texts: `bound_in`, `bound_out`, `owner_of:Param`,
  `data_in:Param`, `data_out:CallSite`, `controlled_in`, `controlled_out:Param`, `to_in`, `via_in`,
  `to_unit_out`, `lifted_of`, `mounted_by_out`, `candidate_count`, `name_in`, `units`, `config_by_path`,
  `file_by_path` and `symbol_by_name` each start at a `Node By Index Scan`. `cp_keys` is a label scan, as
  stated.
- **`name_in` semantics:** with one site resolved to `a` and one unresolved site, the resolved pair is
  suppressed, the other candidate comes back with `count(x)=1`, and the unresolved site comes back with `0`.
  The absent `DISMISSED` type is not an error.
- **ACL matrix:**
  - Granted: `ccg_shop`, `ccg_shop__build_1`, `ccg_shop__ov_1`.
  - Denied: `ccg_shop2`, `ccg_shop_`, `ccg_shopx__build_1`, `ccg_shop_x__build_1`, `ccg_shop-__build_1`.
  - `RENAME` build→live is allowed and `RENAME` onto `ccg_shop2` is denied.
  - `+graph.config|get` returns and `SET` is `NOPERM`.
  - `GRAPH.DELETE` and `GRAPH.LIST` are `NOPERM`.
  - Lock: `SET NX PX`, `GET` and the compare-and-delete `EVAL` work.
- **The `shop_` hole is real:** `ccg_shop___build_1` can be read, written, renamed and expired. `DEL` is not
  granted on it, because `__build_*` does not match `___build_`.
- **Whitelist:** the probe ran every name up to length 5 over `{a, b, _, -}`, every key form (live,
  `__build_`, `__ov_`, lock) and every pattern of the four users. It found **0** cross-system matches among
  the 230 names the regex accepts.
  - The regex is sufficient, and stricter than it needs to be: `a-`, `-a`, `_a`, `a--b` and `a_-b` are
    collision-free but rejected. That is acceptable.
  - The minimal safe rule is "no `__`, no trailing `_`". `a--b` and `a_-b` are rejected, harmlessly.
- **Other checks:** `EMPTY_VERDICT_LOG_HASH` = `c482b1cf54631b37eb4572dc86a68fb1` matches
  `ids.tuple_id(["verdictLog"])`. The memory priors (0.287 MiB/file, ≈ 2.9 GB, ≈ 13.9 GB) match the slice
  report §2.5. v1's 29 edge groups match `load.py` `EDGE_ORDER`.
- **Design choices that hold:** O-10 as `Pack` nodes plus `packSetHash`; `auditSetHash` kept out of the
  identity; `flowAvailable` as a required boolean; `paramMap` dropped with a `dropped[]` count.

### Open questions

1. **Stakeholder: per-system ACL user names (`ccg_<role>_<s>`) or the proposal's fixed names (`ccg_build`,
   `ccg_verdict`, `ccg_engine`, plus `ccg_raw_<s>`)?** P1 already builds two systems on one instance: the
   proving system and `cpg/fixtures/system1` ("built as separate systems", index §2).
   - **Option A — per-system names (the note's proposal).** `create_users` for one system cannot overwrite
     another's grants, and each system's `.acl` file (P1-02 (e), already per-system) holds its own four
     passwords.
     - **Cost:** the proposal §3.5 wording changes; P1-19 (2) text changes; P1-21's `ccg_engine` becomes
       `ccg_engine_<s>`; P1-48's `status` regex, script header, README line and test change (F8).
   - **Option B — fixed names, plus a refusal of a second system per instance.** No proposal edit.
     - **Cost:** the fixture system and the proving system cannot share `ccg-falkordb`, so P1-40/P1-41 run
       only on throwaways or after a teardown, and the refusal itself needs a test. The P1-48 regex stays
       as it is, apart from the F8 charset bug.
   - **Option C — fixed names with key patterns accumulated across systems.** One credential then spans
     every system, which voids the isolation that the whitelist proof exists for. Not recommended.
   - **Recommendation: A.** It is the only option where the §7 isolation proof means one credential per
     system. Its cost is text edits in four places, and none of them is code yet: P1-48's regex is
     uncommitted.
2. **For `architect`:** F3, F4 and F5 each need a ruling in `-answer-spec2.md`, which is under concurrent
   review. Those rulings decide the final shape of `bound_out`, `units` and `to_unit_out` here.

## Pass 2 (2026-11-09)

### Scope & verdict

**Reviewed:** the note at Version 2, revised in place and uncommitted. I re-ran the checks on a second
labelled throwaway (`falkordb/falkordb:v4.18.11`, `127.0.0.1:6398`, removed afterwards; 6379 and 6390 were
not touched). I recomputed `bundleHash` independently on the two stored slice bundles
(`cpg/.ccg-work/vertical-slice/s14a`). I did not judge §9 A1–A3; they are owed by `-answer-spec2.md`.

**Verdict: approve with suggestions.** No blocker or major remains on the graph side. The six new findings
(N1–N6) are minors.

CPG: considered, not relevant — no loaded `cpg_*` graph indexes `cpg/`; I checked against the slice
bundles, `cpg/ccg/` source and a live throwaway.

### Pass 1 dispositions

- **F1 — fixed.** I hashed the bundles two ways: (a) every line re-serialized with my own canonical JSON;
  (b) the stored bytes with only line 1 rewritten.
  - The two ways agree on both bundles (`0fb89974…`: 85,071 lines; `c2341bca…`: 2,775 lines). Every
    stored line is already canonical.
  - The hash is unchanged when only `builtAt` and `sandboxAuditIds` change.
  - It changes on a one-character `tools.normalizerVersion` edit, a `builtCommit` edit and a non-header
    record edit.
  - `codeHash` now excludes `flow-availability.yaml`, which matches §3's claim and §8 (e).
- **F2 — fixed.** I ran it live on one link with a confirmed unbound site A and a possible site B that has a
  two-entry chain:
  - `bound_out` for `(L, A)` returns no rows; for `(L, B)` it returns `e1`, `e2` by `ord`.
  - The `VIA` invariant query finds 0 violations. It finds exactly site A once I plant A's reason as
    `binding-confirmed`.
  - A `BOUND_BY` with an orphan `siteId` is caught by the §8 (d) check.
  - The plan starts at a `Node By Index Scan`.
- **F3 — graph side fixed.** The engine reads `b.port`/`portOrigin`/`packId` and does not emit them until
  A1 is ruled. `Bound` needs no `siteId`, because `bound_out` reads per site, and that matches A1.
- **F4 — graph side consistent.** `bundleHash` excludes `builtAt` and `manifestHash` does not carry it, so
  rebuilds of the same commits keep `snapshotId`. The remaining tension sits entirely in A2.
- **F5 — graph side fixed.** `to_unit_out` is read only under A3's first option.
- **F6 — fixed, by the author's stronger design.** Details are in the next section. The residual is
  acceptable.
- **F7 — fixed.** `name_out` ran live: `c1` (resolved to `a`) → `b` with `count(x)=1`, and `c2` → `a` and
  `b` with `0`. Its plan is index-anchored. `consumes_out`, the per-label `has_out` and the label sets are
  present.
- **F8 — fixed.** The uncommitted `cpg/infra/ccg-falkordb.sh:121` `ACL_USER_RE` uses per-system names. The
  user-name probe is under "The per-system user names".
- **F9 — routed, not yet landed.** §9 lists the edits, but the step texts are unchanged:
  - `-p1-extraction.md:90` (P1-03, weak rule);
  - `-p1-linking.md:139` (P1-19, "the four `ccg_*` users", `shop__x` and no `shop_`);
  - `-p1-answers.md:50` (`ccg_engine`);
  - `connected-code-graph.md:636`.

  This is not a defect of the note, but `teco` must route these edits before P1-03, P1-19 or P1-21 is
  dispatched.
- **F10 — fixed** (value-free `failReason`; test (i)). See N5.
- **F11 — fixed** (G1/G2 `entryId` tuples; the `config_by_path` drop rule).
- **F12 — fixed** (§4 "Excluded in P1").
- **F13 — fixed.** The `toLower(s.qname)` control shows `Node By Label Scan` (I re-ran it), and the harness
  note is present.
- **F14 — fixed except one item.** These are done: the empty system is refused, `builtAt` is absent when
  no unit is fresh, memory step 2 is fixed, and G3 is trimmed. Not done: index §8's "minus timestamps" also
  needs to exempt `auditSetHash`. That is `architect`'s wording, and §9 does not list it.

### F6 re-verification: `%W~ccg_<s>` for the build user

I ran `ACL DRYRUN` for every attempt below, and executed the important ones for real.

**Refused on the live key:**
- `GRAPH.QUERY`, `GRAPH.RO_QUERY`, `GRAPH.EXPLAIN`;
- `GRAPH.CONSTRAINT CREATE` and `DROP`;
- `EXPIRE … 0` (executed);
- `DEL`;
- `RENAME` live → `__ov_*` / `__build_*` (executed).

**Refused as commands:** `PEXPIRE`, `UNLINK`, `RENAMENX`, `COPY`, `GRAPH.COPY` (both directions),
`RESTORE … REPLACE`, `MOVE`, `DUMP`, `TYPE`, `TTL`, `GRAPH.DELETE`.

**Script paths:** through `EVAL` on the lock key, an inner `RENAME` live-away, `DEL` and `EXPIRE` all fail
with "ACL failure in script". An inner `GRAPH.QUERY` is "not allowed from script".

**Other keys:**
- `ccg_shop2` and `ccg_other` are refused, as both target and destination.
- `SET ccg_shop__x` is refused (no `+set` outside the lock selector).

**Allowed:**
- `RENAME` build → live (executed: live then held the build key's node);
- `RENAME __ov_*` → live;
- `EXPIRE`/`DEL` on build and overlay keys;
- the lock (`SET NX PX`, `GET` and compare-and-delete in one `EVAL`);
- `EXISTS ccg_shop`.

**Residual: acceptable.** Overwriting live by `RENAME` is the swap itself. Producing live is the build
credential's purpose, so a stolen credential is equivalent to a poisoned build, and P1-02's custody is the
control. Live is rebuildable from bundles, since zero snapshots are kept. Two items to add to §7's
residual sentence: `RENAME __ov_*` → live is the same class, and `EXISTS` on live is allowed (harmless).

### The per-system user names (option A)

The probe ran all 230 whitelist-valid names up to length 5 over `{a, b, _, -}`, crossed with the four roles:
- That gives 920 users, all distinct: `acl_user` is injective.
- §9's regex accepts exactly role × valid name. Its only matches outside the generated set are
  `ccg_build_x_a`-style strings, and each one parses as a valid (role, system) pair.
- The bash ERE and the Python regex agree on the whole universe, including `ccg_admin_*`, `ccg_build`,
  `ccg_raw_` and the malformed names.
- The revised key patterns, including the `__build_*`/`__ov_*` selector, still give 0 cross-system matches.

### New findings

#### N1 — minor — `Unit.bundleHash` is a graph property computed over `bindParts` and `bindKeyCheck` (G4, §3)

*Verified by reading.* `-bundle-spec5.md` §6.3 says neither may "become a graph … property", and its §8 (2)
row says "no `bindParts` hash". A whole-bundle digest gives no per-field oracle, but it breaks the letter of
that rule. No reader needs it: the `units` text does not return it.

**Fix:** use `bundleHash` only as a `manifestHash` input and drop it from `Unit`. Alternatively, get
spec5's owner to rule explicitly that a whole-bundle digest may be stored.

#### N2 — minor — the `ASSEMBLY_CODE` import-coverage test (§3, §8 (f)) is order-dependent as written

*Verified.* After importing only `ccg.bundle`, the `ccg.*` modules in `sys.modules` are `ccg`, `ccg.bundle`
and `ccg.ids`. Once another test has imported `ccg.answer` and `ccg.atrest`, they are also present, together
with `ccg.keynorm`. In a pytest session the check therefore goes red for modules outside the list.
Importing also misses function-level (lazy) imports.

**Fix:** run the check in a fresh `python -I` subprocess, *after running* the assembler on a fixture
bundle. Positive control: a planted lazy import of an unlisted `ccg` module must turn it red.

#### N3 — minor — the swap's credential split is not among §9's P1-19 edits

*Inferred from the P1-19 text.*
- §7 requires post-`RENAME` work to run as `ccg_verdict_<s>`.
- `%W~` makes v1 row 11's guarded-`EVAL` kept-`prev` form impossible for the build user, because it needs
  read on live. Nothing is lost with zero kept snapshots, but P1-19 (1) still names that form.
- P1-19's Files still give `swap(admin, system, build_id)`. A swap run as admin bypasses everything §7
  verifies.

**Fix:** add the following to §9's P1-19 edit: the swap uses `ccg_build_<s>` for lock + `RENAME` and
`ccg_verdict_<s>` afterwards, and never admin; the guarded-`EVAL` form is withdrawn.

#### N4 — minor — the format-sweep test contradicts §9's own placement of the regex

*Verified by grep.* §7 sweeps for `ccg_` key or user text "anywhere else in … `cpg/infra/`". §9 puts the
ERE literal in `ccg-falkordb.sh`, and the uncommitted file has it at `:121`. The sweep scope also contains
`infra/tests/test-ccg-falkordb.sh` (13 hits, deliberate fixtures) and `infra/README.md` (`ccg_smoke`).

**Fix:** restrict the sweep to the Python under `cpg/ccg/` and `cpg/clients/`; the regex-equality test
already covers infra. Otherwise, give the sweep an explicit allowlist with each hit's owner.

#### N5 — minor — the "closed" `failReason` code list has no home (G4)

*Verified by reading.* The list ends in "…", and test (i) checks only the shape `^[a-z-]+:[a-z0-9-]+$`.
`failReason` is now an input to `manifestHash`, so its values matter.

**Fix:** name the one constant or registry file that holds the codes (and its owning step, which looks like
P1-20's `ccg/staleness.py`). Test membership, not only shape.

#### N6 — minor — growth-stops: rework growth

*Verified with `wc -w`: 4,602 words before, 5,610 after (+1,008).*

**New scope:** the per-system naming decision (§7's opening paragraph, §9's consequential edits).

**Rework:** everything else, in answer to Pass 1. Most of it is necessary content that was missing (two
query texts, tests, the exclusion list). Three passages are trimmable:
- §7's **Verified** bullet re-lists the matrix that §8 P1-19 (c) cites "as stated there". Keep one home.
- §9 A1–A3 restate the problem descriptions of review F3–F5. One line each, citing the finding and the
  ruling needed, would do.
- The **Live evidence** paragraph names container ports and history.

The revision line is one dated line, as required.

## Pass 3 (2026-11-10)

### Scope & verdict

**Reviewed:** the note after graph-dba's third revision. It is uncommitted and down to 5,474 words (−136).
The live checks ran on a labelled throwaway (`falkordb/falkordb:v4.18.11`, `127.0.0.1:6399`), removed
afterwards; 6379 and 6390 were not touched.

**Verdict: approve with suggestions.** N1–N6 are resolved. The three new findings (P3-1 to P3-3) are
minors: each is a wording edit in §9 or §8, and none changes the graph model.

CPG: considered, not relevant — no loaded `cpg_*` graph indexes `cpg/`; I checked the claims by reading and
on a live throwaway.

### Pass 2 dispositions

- **N1 — fixed.** `bundleHash` is now an input to `manifestHash` only. G4 no longer lists it, and a grep
  finds no `u.bundleHash` or `Unit.bundleHash` in any `-p1-*.md`. §8 (e)'s mutant is still meaningful.
- **N2 — fixed, with one nit.** The fresh-subprocess wording is in §3.
  - Nit: the test must launch `sys.executable -I`, not a `python` found on `PATH`.
  - Evidence: `cpg/.venv/bin/python -I` imports `ccg.bundle` from another directory, while the system
    `python3 -I` raises `ModuleNotFoundError: No module named 'ccg'`.
- **N3 — fixed for the post-`RENAME` half; the pre-`RENAME` half is P3-1.** I checked `ccg_verdict_shop`
  live with §7's line:
  - it gets the lock with no admin: `SET NX PX`, `GET` and compare-and-delete `EVAL` all work;
  - it reads the live `Snapshot` offset and patches and advances it in one `GRAPH.QUERY`;
  - `GRAPH.RO_QUERY` is refused (the remainder check must use `GRAPH.QUERY`);
  - `DEL`, `EXPIRE` and `RENAME` of live are refused, and so is any access to `ccg_shop__build_1`.

  The guarded-`EVAL` form is withdrawn, consistently in §7 and §9.
- **N4 — fixed, but the equality test is P3-2.** The sweep now covers Python under `cpg/ccg/` and
  `cpg/clients/` other than `sysname.py`, and `cpg/infra/` is out of it.
- **N5 — fixed.** `FAIL_REASONS` (eight codes) is in `cpg/ccg/staleness.py`, and test (i) checks
  membership and refusal.
  - The producers I found map onto it:
    - P1-20's per-unit `try`: `extract:*`, with `extract:failed` as the catch-all;
    - not-mapped units: `manifest:never-built`;
    - D1: `build:shallow-clone`;
    - spec3 §8.6, a minor above the reader's: `bundle:schema-minor`;
    - spec5 §5.4: `bundle:bind-key-check`;
    - P1-18 (7): `bundle:audit-missing`.
  - Nit: §3's "a major other than `1`" refusal names no code. Rule it `bundle:invalid`, or a whole-build
    refusal.
- **N6 — fixed.** §8 P1-19 (c) now holds every cell of the Pass 2 matrix, plus `TIMEOUT_MAX`. The
  `EVAL`'d-`DEL` refusal survives in §7's prose; the script-path checks (Pass 2) were never in the matrix.
  This revision cut net words (−136). The revision line is one dated line, but the header still says
  `Version: 2` (nit: bump it).

### New findings

- **P3-1 — minor — the pre-`RENAME` replay has no credential (§9, P1-19 Files edit).** The proposal's swap
  is "under the lock, replay every log entry past that offset **onto the new key**, then `RENAME`"
  (`docs/plans/connected-code-graph.md`, B6 section). `ccg_verdict_<s>` cannot reach build keys (verified
  above). §9's sentence names only "the lock and the `RENAME`" for `build_conn` and "post-`RENAME`" work for
  `verdict_conn`.

  **Fix:** add "the pre-`RENAME` replay onto the build key runs as `ccg_build_<s>`". Add verdict's lock and
  live read and patch to §8 P1-19 (c); the swap depends on both. It is a no-op in P1, but P4 inherits the
  wording.
- **P3-2 — minor — §9 asks for an equality between two regexes that can never be string-equal.**
  `acl_user_re()` is built from `SYSTEM_NAME_RE`, which uses `(?:…)`. Bash ERE has no `(?:`, so a string
  comparison is `False`, as checked. The live script's `ACL_USER_RE` also lacks §9's `default|` alternative.
  Length 1–48 sits in neither regex.

  **Fix:** define the test as *same verdict over a generated universe*: the Pass 2 probe of 920 valid users
  plus malformed ones, and a 48- and a 49-character name. The Pass 2 probe already showed that bash and
  Python agree on such a universe.
- **P3-3 — minor — two step-text edits in §9 name no target text.**
  - The P1-48 edit adds a sentence but does not say it replaces `-p1-extraction.md:41`'s "the four
    `ccg_*` ACL users are the P1 steady state".
  - The proposal §3.5 edit does not say which sub-bullets it replaces (`ccg_build` … `ccg_raw_<system>`,
    lines 621–639; keep the lock rationale).

  The other replacements match present text verbatim, which I checked: P1-03 (1) at
  `-p1-extraction.md:90`, P1-19 (1) "(a guarded `EVAL` only where a kept `prev` is wanted; …", P1-19 (2)
  "the four `ccg_*` users" at `-p1-linking.md:139`, and P1-21's "`CypherReader` bound to the `ccg_engine`
  credential" at `-p1-answers.md:49–50`.

**Same-revision cross-check:**
- `bundleHash` is never stored, yet it is still the `manifestHash` input and the target of §8 (e)'s mutant:
  consistent.
- The per-site `bound_out` and A1's "no `siteId`": consistent.
- `failReason` is in `manifestHash`, so `FAIL_REASONS` must be deterministic codes, which they are;
  `failedAt` is in A2.
- The sweep's carve-out of `cpg/infra/` versus the equality test: see P3-2.
- §7's verdict-only post-`RENAME` work versus the proposal's replay onto the build key: see P3-1.
