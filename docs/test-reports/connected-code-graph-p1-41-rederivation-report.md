# Connected code graph — P1-41 re-derivation of the impact rows (dispatch A)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1-41)

CPG: considered, not relevant — the subject is the `ccg` answer-engine walk over the fixture manifests, derived from the fixture source and the walk rules; no `cpg_<component>` graph answers it.

## 1. What was done

The expected items of every manifest row that carries an impact-column item set (`DF-1..4`, `CD-1..4`, `RV-1..3`) were derived by hand from the fixture source, `docs/plans/connected-code-graph-answer-spec2.md` §5/§5.1 and the rules of `cpg/ccg/engine/bfs.py` (read for rules, not outputs), written to a working file **before** any engine run (scratch `derivation.md`; its rules R1 to R8 and assumptions A1 to A5 are the basis of §3). The derivation was then compared with the shipped engine on a throwaway instance (`ccg-falkordb-tw-16391`, system `fixture-system1`, scratch config/credentials/keys/audit dirs; `falkordb-dev`, `:6390` and `ccg-falkordb-tw-16390` untouched). No engine code was edited and no manifest was bent to the engine.

Scope check (read from the manifests): the only rows with `kind` `impact-forward` or `impact-reverse` are `DF-1..4`, `CD-1..4`, `RV-1..3` (fx-svc.yaml; `DF-3`/`RV-3` also compact in fx-web.yaml). `AB-1/2`, `LNK-42/43`, `B1F-*`, `MR-*`, `AC6-*` assert links, edges or staleness, not impact item sets, so no other row was owed a re-derivation.

## 2. Derivation rules that decide the answers (summary)

- Anchor: a symbol or `file:line` ask resolves to a `Symbol` (`ccg/engine/anchors.py`: `SymbolAnchor`, `ContractAnchor`, `ConfigAnchor`; no point anchor). A direct caller of a `Symbol` anchor is `confirmed` (entry row unlabelled).
- Bound: `DEFAULT_DEPTH_BOUND = 2`, no request field (spec2 §4.1); an inc-1 row from distance 2 is cut (`bfs.walk`: `base[1] + row.inc > bound`).
- V-both: from a `function`-state hop at distance `d`, callees at `d+k`, labelled; the anchor is never an item; a callee's callers are never walked.
- Depends-on (RV): no callee run (V18 (g)); call-based: every call site of a visited function is walked, `CALLS_NAME` candidates included (every non-`<dynamic>` site has `CALLS_NAME` to every same-name symbol, `ccg/assemble/rows.py`).
- Notice: `langs` = languages of the flag-false units hosting a visited `Symbol`/`CallSite`/point.

## 3. Old versus new expected items (the P1 form, now in the manifests and fixtures plan §5.5)

| Row | Old expected (P1 form) | New expected | Rule that changed it |
|---|---|---|---|
| DF-1 | `apply_discount` d1 `confirmed`/`resolved-call` | same; + notices `[python]` | none (V-both's callee of `apply_discount` is the anchor, never an item) |
| DF-2 | `create_order` d2 `possible`/`flow-unavailable:python` | same; + notices | none |
| DF-3 | `OrderTotal` d3 `possible`/`flow-unavailable:python` | **not listed** | depth bound 2: `TO ←` into LNK-36 is inc 0 (Link at d2), `VIA ←` to the consumer is inc 1, cut. Not a V-both effect: the old derivation ignored D3. Applies to the flag-true form too (d3) |
| DF-4 | "unchanged" (no `p1Form`) | `accept_quote` d2 `possible`/`name-match`, now explicit + notices | stated, value unchanged |
| CD-1 | not asserted | not asserted, not listed (`audit` is inside the anchor) | stated |
| CD-2 | caller `finalize` d1 `possible`/`flow-unavailable:python` | caller `finalize` d1 **`confirmed`/`resolved-call`**; callee run item `notify` d2 **`possible`/`flow-unavailable:python`**; control half still "not asserted" | (a) the ask is a `Symbol`, not a `Cond`: a direct caller of a symbol anchor is `confirmed` (the old value described a point anchor, which no resolver produces); (b) V-both callee run of `finalize` |
| CD-3 | not asserted | not asserted, not listed | `publish_created` is a callee of `notify` at d3: cut by the bound |
| CD-4 | "unchanged": `publish_alert` d2 `possible`/`name-match` | **not listed** | `publish_alert` is a callee of `notify` at d3: cut. The old "unchanged" had no chain from the anchor at flag-false |
| RV-1 | `connect` d2 `possible`/`flow-unavailable:python`; others as manifest | same + **extra** `OrderCache.get` d2 `possible`/`flow-unavailable:python` | call-based walk: `load_rate`'s `_RATES.get(...)` is name-matched (A3). Not V-both |
| RV-2 | `load_rate` d2 `possible`/`flow-unavailable:python`; others as manifest | same + **extra** `audit` d1 `confirmed`/`resolved-call` | call-based walk lists `audit(o)` (unused result), which the data-dependency model omits. Not V-both |
| RV-3 | `base_price` d3 `possible`/`flow-unavailable:python`; others as manifest | **none listed**; notices `[javascript]` | (a) `base_price` d3 beyond the bound; (b) derivation error corrected: a nested function is its own `Symbol`, `fetch` belongs to `OrderTotal.place`, so `OrderTotal` owns only `useState` and reaches no `Link` |

Callee-run items exist only in `CD-2` (`notify`); no callee-run item appears in `CF-DF`. `notices` per ask: `python` for `CF-DF`, `CF-CD`, RV-1, RV-2; `javascript` for RV-3. Counts: the census of `cpg/tests/test_fixtures.py` (passing) asserts 134 rows; 93 are P1 (the "92/133" of the dispatch text is the 2026-11-10 count, superseded by the 2026-11-11 note of the fixtures plan: no row was added or removed).

A derived alternative, **not adopted** (decision below): with RV-3's subject `OrderTotal.place`, P1 would list `create_order` d1 `confirmed`, `apply_discount` d2 `confirmed`, `Repo.save`/`SqlRepo.save`/`MemRepo.save` d2 `possible`/`name-match`, `base_price` cut, notices `[javascript, python]`.

## 4. Engine versus derivation

Run: `refresh` of the materialized fixture system, then `Client.ask` for the five asks (algorithm `bfs-v1` `1.2`, `depthBound.bound` 2 on every answer). Observed output: `ask1.out` in the scratch dir.

| Ask | Engine items (observed) | Equals derivation? |
|---|---|---|
| CF-DF | `apply_discount` d1 confirmed; `accept_quote` d2 possible/name-match; `create_order` d2 possible/name-match; notices `[python]` | item set yes; `create_order`'s reason differs |
| CF-CD | `finalize` d1 possible/name-match; `notify` d2 possible/name-match; notices `[python]` | item set yes; both statuses differ |
| RV-1 | `find_hook` d1 confirmed; `is_member`, `load_rate` d1 possible/name-match; `connect` d2 possible/name-match; `OrderCache.get` d2 possible/flow-unavailable:python | item set yes; four statuses differ, the extra matches |
| RV-2 | `find_hook`, `audit`, `tier_of` d1 confirmed; `OrderCache.get` d1 possible/name-match; `load_rate` d2 possible/flow-unavailable:python | item set yes; `find_hook` and `OrderCache.get` differ, the extra `audit` matches |
| RV-3 | no item; notices `[javascript]` | yes |

Every difference has one cause, on the **extraction layer**, not the walk. From the engine's own graph (`MATCH (c:CallSite)-[:CALLS]->(s:Symbol)`): the frontend left **no resolved `CALLS` edge** for the cross-file calls the manifests plant as resolved (`create_order`→`apply_discount`, `base_price`→`load_rate`/`is_member`, `finalize`→`eligible`, `notify`→`publish_created`, `is_member`→`connect`, `tier_of`→`load_rate`; all are `from fx_svc.<pkg> import ...` while the unit root is `.`; the import path as the cause is my inference, not tested). It **did** resolve calls the manifests plant as name-match (`hooks.find_hook` in `base_price` and `eligible`, `bus.publish_alert` in `notify`) and left `_CACHE.get` unresolved where RV-2 plants it confirmed. Re-deriving the five asks from the observed edges with the same walk rules (derivation B, in the working file) reproduces the engine's items, statuses, reasons and distances exactly, so **no walk (engine) defect was found**; the differences are classified: derivation error (RV-3/DF-3 nested function, corrected) or frontend resolution (the rest, to be reported red by row id in dispatch B under layer "extraction": DF-2, CD-2, RV-1 (`find_hook`, `is_member`, `load_rate`, `connect`), RV-2 (`find_hook`, `OrderCache.get`)).

## 5. Defects and findings (for the re-gate and dispatch B)

1. **Fixture build blocked by the kafka pack (P1T-04).** `refresh` with the default packs raised `ccg.packrt.match.UnsupportedSubject: onUnknown config-derived is not produced by the P1-16 runtime` (`ccg/packrt/run.py:155`, pack `packs/kafka/pack.yaml:22`; trigger is presumably `producer.send(os.environ["ORDERS_TOPIC"], o)` in `bus/producer.py::publish_dynamic`, inferred, not isolated). The run in §4 used the packs minus `kafka`. Severity: blocks the 92/93-row execution as specified.
2. **Depth bound versus the fixture plan.** `DF-3` and `RV-3`'s `base_price` sit at distance 3; the bound is 2 and has no request field, so neither form (flag-true included, by derivation, not run) can list them. Decision for `teco`/`architect`: re-plant those rows at distance <= 2, or raise the bound (a new algorithm version).
3. **Subject locator.** LNK-36/DF-3/RV-3 name `OrderTotal`; the `fetch` consumer is `OrderTotal.place` (graph: both Symbols exist). In my reduced-pack build no `Link` for that consumer existed and no `http` `key-match` link was produced at all (only `partial-key`/`unknown-key`; cause not investigated), so the link-dependent rows are unverified by the engine.
4. **Cross-file resolution.** §4: the frontend leaves the planted `fx_svc.*` cross-file calls unresolved and over-resolves two name-match plants. If this is a fixture defect (package path) the fix is in the fixture tree; if extraction, in `P1-05`/the normalizer. Not classified further here.
5. **P1T-13 `B1F-*` ("a change to W lists R").** `bfs.py` documents `READS`/`WRITES` as not walked at P1, so no row of the shipped walk can list `R` from `W`; not run, flagged for the plan owner.
6. **Source drift (not edited, not mine).** `docs/plans/connected-code-graph-p1.md` §5 (the "fixture rows' P1 form" paragraph, ~lines 272-282) still states the old form (DF-3 `possible` d3, CD-2 `finalize` `possible`, RV-3 `base_price` d3) and the CD-4 sentence; owner `architect`.
7. `scripts` (javascript) unit: `not-mapped`/`extract:failed` in my build (31 fresh, 1 not-mapped); not investigated.

## 6. Verification and what is unverified

- `cd cpg && .venv/bin/python -m pytest tests/test_fixtures.py tests/test_fixture_keynorm_mutants.py -q`: **85 passed, 2 deselected** (the two `live` tests). The consistency test now also compares `extras`, `calleeRun`, `listed`/`none`, and per-ask `notices` against the plan; planted-failing inputs added for each. `cross.yaml` regenerated by `materialize.py --emit-cross` (one ref reordered).
- Unverified: the flag-true re-run (no flag-true unit exists); every link-dependent row (DF-3, RV-3 beyond the empty list); the alternative RV-3 derivation; whether `fx_svc` is the cause of the unresolved calls; the full 93-row execution (dispatch B); `render_text`/`render_markdown` notice wording (not exercised); the P1-42 dev-anchor walks that reach flag-false functions.
- The expected items assume the frontend resolves what the manifests plant (derivation A); against the current frontend the rows of §4 will be red at dispatch B for the extraction-layer cause, not the walk.

## 7. Artifacts

Edited: `cpg/fixtures/system1/expected/fx-svc.yaml`, `fx-web.yaml`, `cross.yaml`, `cpg/tests/test_fixtures.py`, `docs/test-plans/connected-code-graph-fixtures.md` (v1.8, §5.5), `docs/test-plans/connected-code-graph-p1.md` (the "re-derivation owed" paragraph replaced, AC-25 row, P1T-05/17/18). Throwaway instance `ccg-falkordb-tw-16391` removed after the run. Working files (not durable): `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/82bb1fdb-fb53-4f8c-80b5-7bf434adbf7b/scratchpad/derivation.md` and `ask1.out`.
