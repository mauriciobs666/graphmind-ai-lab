# Review: connected code graph, canonical answer spec v1 (S33)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P0)

## Scope and verdict

**Reviewed:** `docs/plans/connected-code-graph-answer-spec.md` (441 lines, committed) as the S33 plan gate. Baseline: the S33 and S34 rows of `docs/plans/connected-code-graph-p0-2.md` (§3, §4.2), `docs/plans/connected-code-graph-eval-protocol-ml.md` (EP §5.1-§5.2, §10.4, §14, §15.2 Q1), `-bundle-spec3.md`/`-spec4.md`, `-p0-graph.md`, `-graph.md` §6.2, `cpg/ccg/{bundle,ids,keynorm}.py`, `cpg/ccg/registry/reasons.yaml`, `cpg/spikes/vertical-slice/{impact,render,lookup,load,assemble}.py`, the s14a bundles, root `REVIEW.md`.
**Not reviewed:** the live (`-m live`) suite (the ccg-falkordb store needs auth and the tests create build keys, so I did not run it); the `data-scientist` consult the S33 row requires (it is not in this review).

**Verdict: needs changes** (1 blocker, 5 majors). The shape is sound, the EP walk (H1-H4) closes, and the golden AC-1 values are right. But the F-4 validator rule cannot accept its own masked output, the determinism rule leaves arrays unordered, and the F-12 and F-3 "checks" are partly unable to fail or have no owner.

CPG: considered, not relevant — a contract specification; no loaded `cpg_*` graph indexes `cpg/`, and every claim was checked against the documents, the code and the slice outputs.

## Findings

### B1 (blocker, verified) The masking literal does not parse under the grammar the validator uses, so §7.3 and §11 test 9 cannot go green

§7.3 splits `keyPath` with `bundle._keypath_segments`, masks a failing segment to `[#]`, and requires V12 to accept the result ("accepted after `mask_identifier`"; "a segment already stored as `[#…]` … passes the screen unchanged"). I ran the function: `_keypath_segments` raises `BundleError ... does not parse (§5.17)` on `a.[#].b`, `a.[#…].b`, `[#]`, `server.[#]` and `[#…].x`. Bracket segments are only `[digits]` or `["quoted"]`, and a bare name excludes `[` (`bundle.py:441`). So (1) V12 applied to any masked or P2-8-stored `keyPath` raises or, under the spec's own "does not parse, fail closed" rule, rejects; (2) the fail-closed output `[#]` is itself unparseable, so the validator rejects what the engine emits; (3) the "idempotent" claim is about `screen_fails` on one segment, which is never reached. S34 would have to re-derive the decision the spec exists to make. Open item 1 treats this as "one constant"; it is structural. Also `spring:server.port` (a `config_key` `keyNorm`, `<namespace>:<keyPath>`, bundle spec3 §6) tokenizes to `spring:server`, `port`: the namespace is glued to the first segment, so a mask there would erase the namespace.
**Fix.** In §7.3 define V12's tokenizer: strip a `config_key` `keyNorm`'s `<namespace>:` first; treat a masked segment `[#]` and the stored `[#…]` (whichever S17 writes; accept both) as a third segment kind, "masked, not screened", before calling `_keypath_segments`' regexp; give three literal vectors (input, masked output, V12 result). Name the mutant: V12 using `_keypath_segments` unmodified must fail test 9. This also settles your ruling on open item 1: the `[#]` default stands **only** with that tokenizer; S34 need not wait for S17's literal because both forms are accepted.

### M1 (major, verified) F-12: two of the three checks cannot fail, and the `CONTROLLED_BY` half has no test anywhere

- §7.4(1) says an answer from a walk without the guard "cannot validate". `visitedSeedsAnchor: const true` is a self-declaration: any producer writes `true`. It checks nothing about the walk.
- §7.4(2) "chain is a simple path in the visited-set key". The chain is rebuilt from `best[key].pred` pointers (`impact.py:Result.chain`), `best` is keyed by that very key, so a real chain cannot repeat it; the rule only rejects hand-built documents. Fine as a footprint, but it is not evidence the walk is cycle-safe.
- Only (3) (anchor never an item) can fail on real output.
- §7.4 says the guard is tested by "the traversal-guard tests `-p0-graph.md` §5 lists". I read §5: they are a two-point `DATA` cycle, the `<` vs `<=` re-expansion, the 10,000-row raise and the 64-level raise. None is a cond-to-cond `CONTROLLED_BY` cycle, and `bfs-v1` does not follow `CONTROLLED_BY` at all (`impact.py` docstring, "Not followed"). F-12's stated subject (`-spec2` F-12: cycle-safe over `CONTROLLED_BY`) therefore has no test owner.
**Fix.** Drop the "cannot validate" claim (keep the declaration as a record). In the P1 carry-over list (M4) add: a planted cond-to-cond cycle that passes through the anchor, asserting termination and no item equal to the anchor; mutant: anchor not seeded into `visited`. Keep (3) as the validator check.

### M2 (major, verified by reading) "Every array ordered" is false: `scope` arrays and others have no rule

§6's table omits `interpretation.request.scope.repoIds` and `scope.paths` (EP §4.2 gold: `{"repoIds":[…]}` / `{"paths":[…]}`, compared as part of equality), so two clients can emit different bytes for one request, which breaks FR-19 and AC-16 at `facts_bytes`. Also unstated: duplicate options in `options` (sorted by bytes but uniqueness not required), uniqueness of a `ConfigSubject` `entryId` (the "ties are impossible" sentence rests on "a symbol appears once"), and what `canonicalize` does with NFC/strip (`ccg.bundle.canonical_line` is plain `json.dumps`, `bundle.py:64`; it normalizes nothing, so V1's "strings NFC" must be reject-or-normalize, unsaid). `dumps` = canonicalize then validate, but canonicalize needs `rootPath` from `staleness.units`; a document with a missing unit (test 10) raises a bare `KeyError` instead of `AnswerError`.
**Fix.** Add rows: `scope.repoIds`/`scope.paths` sorted by code point and unique; `options` unique; `entryId` unique across items and members; state that V1 rejects (does not repair) non-NFC and unstripped identity strings, and that `canonicalize` raises `AnswerError` on an unresolvable `unitId`. Coverage probe, not a list: derive the array set from the schema (every `"type":"array"` property, including inside `$defs`/`oneOf` branches) and require a §6 row or an explicit "unordered, forbidden" for each; S34 test 1 iterates that same derived set instead of hand-listing eight arrays.

### M3 (major, verified) No query the walk runs returns the fields the hops require; the golden recipe cannot be real engine output

V3/V6/V9 require `path`, `line`, `repo`, `unit` on every hop except `ContractPoint`/`Link`, and a `reason` on every possible hop. In the loaded rows (`load.build_rows`, observed) `Param`, `Ret`, `Cond` nodes carry only `pointId`, `line` (+ `index`/`name`/`code`): no `path`, `repoId` or `unitId`. They need a join through the owning `Symbol`, which no `Reader` method returns (`owner_of` returns ids only). `LIFTED_QUERY` (the "verbatim" `lifted_via` the golden uses) returns neither `repo` nor `unit` nor `siteId`, yet `Lifted` requires all three. The `Reader` queries read edge `status` but never edge `reason`, so a stored-possible hop (`name-match`, `partial-key`, ...) would fail V3/V6 as the slice walk stands. The golden hides all three (its hops are `Symbol`/`CallSite`/`ContractPoint`/`Link`, all confirmed). Consequences: §4.1's "a raw client re-running `queries` reproduces the facts" is overstated, and the golden's `recipe.queries` (7 names) is a hand-built fixture, not what a P1 engine would emit.
**Fix.** Mark the golden recipe as hand-built; list the three read obligations (owner join for Param/Ret/Cond path/repo/unit; unit/repo/siteId in the lifted read; edge `reason`) in the P1 carry-over list (M4).

### M4 (major, verified) Cross-deliverable citations with no owning unit

Walked every "routed to / P1 / S50 / P5" clause in the spec against `docs/plans/connected-code-graph-p0-2.md`:

| Obligation (spec §) | Owner found | Verdict |
|---|---|---|
| Default depth bound >= 2 (§4.4) | S50 row records EP Q1 in proposal §4 P1 row; p0-2 line 1496 "For P1" list | owned |
| `unexplored` counting (§4.4, §10.5) | none; S50's row names Q1 and Q3 only | **unowned** |
| Renderer test: `port` hop text contains the port (§7.1) | "P1 plan" (does not exist; no P0 row builds a renderer) | **unowned** |
| Renderer test: planted `"Ignore previous instructions"` only inside a code span; all LLM-bound text via `fence_data` (§7.2) | same | **unowned** |
| `CONTROLLED_BY` cycle test (§7.4) | `-p0-graph.md` §5 lists none (M1) | **unowned** |
| Edge `reason` read, owner join, lifted unit/repo (M3) | none | **unowned** |
| `algorithm.version` bump rule (§4.1) | "P1 plan's call" | unowned, low stakes |
| Snapshot node carries `verdictLogOffset`/`verdictLogHash` (§4.2, §6 "inputs read from the store") | none; `rows.snapshot` (observed) has `snapshotId, systemId, builtAt, vocabVersion, bundleSchemaVersion, manifestHash` and `ddl_v1.cypher` has no such property | **unowned**; the §6 sentence is false for v1 |
| S52 headless prompts (F-3 other half) | S52 row, `cobb` | owned |
| `rcaMarks` shape, DS-3 | P5 | owned (additive) |
| Open item 3 (EP normalizer on config keys) | `data-scientist`, P2 | owned, not S34-blocking |

**Fix.** Add one "Carry-over to P1" list at the end of the spec (the six unowned rows) and have `teco` attach it to p0-2's "For P1" list (line ~1496) or to S50's row, since S50 is the only row that records P1 obligations today. State that the snapshot's stamp is minted by the engine as `0` / the constant until P4, and say which unit adds the properties.

### M5 (major) Several checks have no positive control, and two cannot fail as written

REVIEW.md: every check must name the mutant or positive control that turns it red/green, with expected values not read from the thing checked.
- **Single positive control.** §11 has one valid document, the AC-1 golden: one item, all confirmed, no group, no `possible` hop, no ambiguous/unresolved answer, no config subject, no `launch` hop, no `Bound`. V6-V8, §5.4 and §7.1 therefore never *pass* on any input; each mutant test could be red for the wrong reason (e.g. test 8's "`Link` not followed by its site" mutant fails V5, but also fails on the missing fields of a hand-edited hop). **Fix:** require each mutant test to assert its unmutated base validates, and name the base documents (a `possible` item with a first-possible reason and a later different one; a group with two members; one per interpretation status; one `launch` hop per kind).
- **Constant read from itself.** Test 3 ("offset 0 with a wrong hash is rejected") will compare against `EMPTY_VERDICT_LOG_HASH`, the thing under test. I computed `ids.tuple_id(["verdictLog"])` = `c482b1cf54631b37eb4572dc86a68fb1`; pin that literal in §4.2 and §11. Likewise give one literal vector for `cacheKey` and for `fence_data` (§7.2 gives none; the order "escape, then truncate to 500 code points" is unstated, so a cut inside a `\u{HEX}` escape is possible).
- **Schema-classification guard (F-3, V14).** "Every string-typed property is engine-minted or `x-ccg-data`" is a schema walker. Root `REVIEW.md` requires one planted bare string per composition axis. The schema will use `oneOf`/`$defs`/`$ref` (anchors, subjects), array `items`, nested objects; a walker that skips `oneOf` or `$ref` passes while a string hides there. **Fix:** S34's test plants an unclassified string at each axis (nested object, array item, `$ref` target, `oneOf` branch) and shows each turns the test red; `DATA_FIELDS` path notation (e.g. `Hop.qname` def-qualified) must be fixed in the spec because P1's renderer reads it.

### m1 (minor) `Hop.state` enum is missing from §9's additive list; `function` is an `impact.py` internal

§9 lists additive enum values for `Hop.kind`, `edgeType`, `Group.kind`, `ambiguityKind`, not `state`. The `state` vocabulary is closed in v1 and includes `function`, which `impact.py` adds (FUNCTION, its sixth) beyond `-p0-graph.md` §3.1's five (`value`, `path`, `contract`, `key`, `end`). When P1's `arg(i)` rows replace function-level continuation, the vocabulary changes. Add `state` to the additive list, and cite where `function` is defined (p0-graph §3.3 does not name the state).

### m2 (minor) "`cacheHit` is the only field that may differ" over-claims

For an uncached plain-language repeat the interpretation can legitimately differ (EP §10.4's flip rate over 5 uncached repeats; ML §3.2 item 4). Narrow §6 to "for one fixed `interpretation.request`"; the DS-5 consult is the same issue.

### m3 (minor, verified) F-4 for `http` keys: cite the shared implementation

`cpg/ccg/keynorm.py:mask_http_segments` is the one shared step-8a implementation (its docstring cites EP2 §1.1). §7.3 says only that an `http` `keyNorm` "is already screened" and tests "an `http` `keyNorm` with a failing literal segment is rejected" without saying how the validator splits `VERB /a/b` (verb kept unscreened, `{}`, `{**}`, `{?}`, `{#}` kept). Cite it; the check is `mask_http_segments(k, "{#}") == k` or `... "{?}" == k`.

### m4 (minor) `unresolved` and DS items that change shape

(a) Mixed case undefined: a question naming one mapped and one not-mapped file has no status (a silent drop of the mapped anchor would violate EP §10.4's "no silent single interpretation"). (b) `not-mapped` means a skipped file in `lookups` but a never-built unit in `staleness.buildStatus`; rename one (e.g. `lookups[].state: "skipped"`) or say so. (c) DS-4 (which AC-24 category scores `unresolved`) and DS-5 (is `request` in `facts_bytes`) change the schema if answered other than the default, which would be a major bump after S34 builds; the S33 row requires the `data-scientist` consult (p0-2 §3 row, §9.2 gates table). Run DS-4/DS-5 before dispatching S34; DS-1..3 and item 3 need not block it.

### m5 (minor) Golden values left open that I could pin

`snapshotId`: "any 32-hex" leaves the file non-reproducible. From the s14a bundles via `load.build_rows` I observed `baa84a25a28c1bf05a44b58f870e2fb9` (systemId `p0slice`). `candidateCount`: observed `1` (`render.candidate_count` over the loaded `VIA` rows; the spec says "S34 records the value"). Pin both. The golden also depends on gitignored `cpg/.ccg-work/vertical-slice/s14a` (present); S34 should not regenerate it from there at test time.

### m6 (minor) Names S34 must guess

`mask_identifier(field_kind, ...)`: the `field_kind` vocabulary is not listed (`keyPath`, `config_key`, `env_var`, `http`, `mountPath`, `service`, `envVar`?) and `http` is "carried unchanged" (no-op or raise?). §7.2's data-field list and its own rule disagree: `launch.protocol` (unconstrained string in bundle spec3 §1.6), `repoId`, `contractKind` are neither engine-minted nor in the list. The registry code `flow-unavailable:<lang>` is marked "on: impact items" (item-level) but V6/V7 admit reasons only via a possible hop; say which hop carries it.

### Open items (§10), my view

- **Item 4 (your ruling: `function-level-continuation` in `answer.py` `ENGINE_REASON_CODES`, registry unchanged, disjointness test): agreed.** The disjointness test can fail (plant the code in a registry copy), the registry has no description column a renderer would miss, and the code is `possible` so V6's status equality holds. Two conditions: record the deviation from `-spec3` §7.1 ("engine codes live in the one list") in the spec as a decided exception, and note the code retires with the `arg(i)` rows.
- **Item 1: see B1.** The default stands only with the extended tokenizer.
- Items 2, 3, 5, 6, 7, 8: acceptable as decided defaults; 5 is in M4.

## What is solid

- **EP walk H1-H4 closes on the schema's fields:** H1 reads `subject.qname`/`lineStart`/`lineEnd`/`ownedRegions` (flat list per `-p0-graph.md` D7), H2 `status`, H3 `chain[0].id` and `chain[-1].id`, H4 membership in `items` or `groups[].members`; E9's empty verdict log is `stamp.verdictLogOffset == 0`. Interpretation anchors carry `qname` (S33 row, EP §15.2 Q1).
- **Golden AC-1 values re-derived (ran):** I built rows from the s14a bundles (`assemble` + `load.build_rows`) and ran `impact.walk` from `ContractPoint a359de518acc3d443ff5efa5597f458f` in contract mode: one item, `getHealth` `8ae57f349ffed0533e458ee5341f2936`, distance 1, 0 possible hops, `confirmed`, 4 hops (5 chain entries), `viaData` false, chain `PROVIDES <-`, `TO <-`, `VIA <-`, `HAS_CALL <-` with the provider `a6c2648709f43a0bad2211a1b8bbcd52` and link `8d1cc712eb5d40263a96835e3d5803a4` as in §11. Bundles confirm spans (16-23; provider 954-986), unit ids, `builtAt`/`builtCommit` and `rootPath`s, and the four `liftedVia` ords (17, 107 with `viaLine` 108, 73, 118). `contract_point_id("http","GET /shop/api/health")` begins `a359de51`. `impact` non-live suite: 22 passed, 1 deselected.
- **The additions are consistent with the frozen specs:** `dir`/`edgeType` vocabulary matches graph note §6.2; `candidateCount` on `VIA` matches D6 and `render.py`; `LIFTED_VIA` fields match D1; `lookups` reasons match `-p0-graph.md` §1.3; `launch` matches bundle spec3 §1.6 per-kind exclusivity and `-p0-graph.md` §1.5; `snapshotId` formula matches `load.py:312`; `cacheKey` matches ML §3.2 item 2 and `tuple_id` accepts the new tag (ran). `rcaMarks: []` meets p0-2 §4.2's name requirement without inventing a shape.
- Bytes rule is correct: `canonical_line` is `json.dumps(sort_keys, compact, ensure_ascii=False)` + `\n` (read `bundle.py:64,70`); `facts_bytes` excluding only `provenance` is a clean boundary.
- Growth-stops: first gate of a new document, no prior revision to compare; upstream decisions are cited in §1 rather than restated (only the visited-set key in §7.4 and the distance rule in §5.1 are restated, both short). Nit: "Ready to implement" and the done-condition map duplicate §11.

## Open questions

None for the stakeholder. For `teco`: dispatch the `data-scientist` consult (DS-4, DS-5) before S34 (m4c).

## Outside the reviewed artifact (observed, not a finding against it)

`MemoryReader.from_rows` in `cpg/spikes/vertical-slice/tests/test_live_slice.py:119-134` transposes the `TO` and `VIA` tuples relative to `to_in`/`via_in`; on real rows it returned no items for the contract walk until I swapped them in my scratch script. Used by value-mode tests only (lines 1193, 1198); S34's "hand-built from the slice's BFS result" must not use `from_rows` for contract mode.

## Stopping condition for the re-gate

Re-gate returns **approve** iff all of R1-R6 hold; a failing one is **needs changes**. New findings below major do not block.

- **R1.** Run on the spec's own vectors: for each of `server.[#].port`, `a.[#…].b`, `[#]`, `spring:server.port` (masked and unmasked forms), the tokenizer §7.3 describes yields no exception and the stated V12 result; the 25-character alphanumeric segment is rejected unmasked and accepted masked. (I run it, as I did B1.)
- **R2.** Derive every array property from the schema description in §2-§5 (a fresh sweep, not the old list); each has a §6 row or an explicit forbidden-unordered note. `scope.repoIds` and `scope.paths` are present.
- **R3.** §7.4 contains no "cannot validate" claim for the declaration, and names a `CONTROLLED_BY` cycle test with its owner and mutant.
- **R4.** Every "routed to / P1 / S50" clause maps to a row or list line that exists in `docs/plans/connected-code-graph-p0-2.md` (a grep I run), or to a listed carry-over; zero unowned in M4's table.
- **R5.** §11 names a base document that validates for each mutant test, the literal `c482b1cf54631b37eb4572dc86a68fb1` (or the value `ids.tuple_id(["verdictLog"])` gives when I re-run it), and the axis-planted classification test.
- **R6.** The golden values in §11 still re-derive as above (I re-run `derive`), with `snapshotId` and `candidateCount` pinned.
