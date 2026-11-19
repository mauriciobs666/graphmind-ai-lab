# Connected code graph — P1-41 RCA: fixture build and query defects

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-41)

CPG: considered, not relevant — the subject is `ccg`'s own pipeline (extractor, pack runtime, assembler) run on the fixture system; `cpg_<component>` graphs describe other components' code, and the loaded `ccg` graph is the thing under test.

## 0. Scope and method

Symptoms: `docs/test-reports/connected-code-graph-p1-41-rederivation-report.md` §5 items 1, 3, 4, 7. Read: `cpg/ccg/packrt/{run,mount,extract}.py`, `cpg/packs/{kafka,fastapi}/pack.yaml`, `cpg/extractors/joern/normalize/{core,assemble_bundle}.py`, `cpg/extractors/joern/run.py`, `cpg/ccg/{refresh,build,units}.py`, the fixtures plan, `docs/plans/connected-code-graph-p1-gates.md` P1-41, the ledger.

Reproduction: own throwaway FalkorDB `ccg-falkordb-tw-16392` (removed after), own `CCG_CONFIG_DIR` / `CCG_CREDENTIALS_DIR` / `CCG_AUDIT_DIR`, systems `rcafx1` (the fixture manifest renamed) and `rcat1` (three 4-file scratch repos). The materialized fixture system was built twice: default packs (raises) and without `kafka` (31 fresh, 1 not-mapped, 110 s). Scratch drivers live in `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/82bb1fdb-fb53-4f8c-80b5-7bf434adbf7b/scratchpad/rca/`. Side effects: my first build ran without `CCG_AUDIT_DIR` and appended its sandbox records to the shared, gitignored, append-only `cpg/.audit/sandbox.jsonl` (harmless); `~/.config/ccg/keys/rcafx1.hmac` and `rcat1.hmac` were created and then deleted by me; no source, fixture, manifest or plan was edited.

Every root cause below was re-derived by a second, differently-built experiment (scratch repos varying one factor, or a monkeypatched run), not only by re-reading the same build.

## 1. Summary

| # | Symptom | Root cause (confidence) | Class | Fix owner / kind |
|---|---|---|---|---|
| a | default-pack refresh raises `UnsupportedSubject` | `packrt/run.py:154-155` aborts the whole refresh on any `config-derived` slot; trigger is **not** `publish_dynamic` but the cross-file constants `TOPIC_ORDERS` / `PREFIX`, unresolved because of defect c2 (confirmed) | **P1 defect, before dispatch B** (runtime) + c2 | `tdd-engineer`, code |
| b1 | `scripts` unit `extract:failed` | `EmptyCpgError`: unit's only file is `build.test.ts`, which the JS frontend ignores by default; D15 counts it as a source file (confirmed) | minor P1 defect / unplanned fixture unit, non-blocking | `coder` (code) or `qa-engineer` (declare in AB accounting) |
| b2 | no confirmed `http` `key-match`, no LNK-36 | fixtures plant module-level `APIRouter(prefix=…)` + `include_router(mod.router, …)`; the pack runtime supports only `include_router(factory())` and never composes a router's own prefix (spec §12.2 O-3, §2.3), so every `fx-svc` handler is `possible/partial-key` with a key lacking mount and prefix (confirmed) | **plan/fixture contradiction: ruling, then re-plant or re-gate** | `architect` ruling; `qa-engineer` fixture or manifest |
| c1 | planted resolved cross-file Python calls have no `CALLS` | fixture layout: repo root *is* the `fx_svc` package (`pyproject` `package-dir`), Joern resolves `from fx_svc.core.rules import …` by path from the unit root, finds nothing (confirmed by 3-layout experiment). Not SCIP: P1-15 is not adopted | fixture/layout; **fix before dispatch B** | `qa-engineer` fixture (or `coder` normalizer, optional) |
| c2 | imported constants never resolve (all layouts) | `Resolver.element` tests "constant of this file" before the import alias, and the importing file's own import-bound name is in that set, so the `ref` names the importing file (confirmed by monkeypatch) | **P1 defect, before dispatch B** | `tdd-engineer`, code (P1-05 normalizer) |

The qa-engineer's two guesses: the kafka trigger (`publish_dynamic`) is **refuted**; the import-path cause for (c) is **confirmed**, but incomplete (c2 is a second, layout-independent cause).

## 2. Problem (a): `UnsupportedSubject: onUnknown config-derived`

**Causal chain.**
1. `refresh.py:81` `DEFAULT_PACKS` includes `kafka`; `run_packs` runs it over every bundle. Pack `kafka` rule `produce` matches `send|send_and_wait|produce` under `scope.imports kafka|aiokafka|confluent_kafka`; slot `topic` has no `onUnknown`, so its default for a `provides` of kind `topic` is `config-derived` (`run.py:141-142`, spec §1.4, O-16).
2. `run.py:_one` (lines 150-155): when a slot value is `UNKNOWN` and its mode is `config-derived` it **raises** `UnsupportedSubject`. The rest of the file's per-site degradation is a diagnostic or a keyless `possible` fact; this branch alone aborts the refresh, a per-site condition becoming a system failure. The P1-16 plan (`-p1-linking.md` P1-16 behaviours 1-9) does not list `config-derived`, and no test runs the `kafka` pack through `run_packs` (`tests/test_refresh.py:804` pins only the pack *names*; `test_packrt_run.py:379` pins the refusal for two other unsupported shapes).
3. Trigger. I wrapped `_one` over the bundles of the fixture build and ran only the kafka pack: `RAISE publish_shipped L33 ('', '\x00\x00')` (`f"{PREFIX}.shipped"`) and `RAISE publish_created L29 ('', '\x00')` (`TOPIC_ORDERS`); `publish_dynamic` (`os.environ["ORDERS_TOPIC"]`) produces **neither OK nor RAISE** (its alternative list is empty, so the product is empty and no fact is emitted, `run.py:131` comment). Both triggers are cross-file constants imported with `from fx_svc.bus.constants import PREFIX, TOPIC_ORDERS`; their value set is `{ref: 'bus/producer.py::TOPIC_ORDERS'}`, which names a constant that does not exist (see c2).

**Why only the fixtures trip it.** No Python file of the proving system imports `kafka`/`aiokafka`/`confluent_kafka` (`grep` over the tree, fixtures excluded: none), so the rule's `scope` never matches and the `config-derived` branch is never reached.

**Rows and plan status.** The planting rows are `LNK-05` (`TOPIC_ORDERS` across files) and `LNK-09` (`f"{PREFIX}.shipped"`), both `c`/`key-match`, **gate P2**; `AC6-TOP-1` (`publish_dynamic`, `possible`/`config-derived`, **gate P2**) is the row the `config-derived` code exists for, and it is not the trigger. All are P2: the plan expects the P1 gate to report them "planted, not asserted at P1" (`-p1-gates.md` P1-41, closed-world sentence). It does not expect the **build** to abort on them. So the fixture is right, the pack contract is right (spec §1.4: `config-derived` degrades to a keyless `possible` fact, reason `config-derived`, or `runtime-built`), and the runtime is the gap: a P1-16 omission that the fixtures plan exposes. The architect's note in `-p1-gates.md` ("kafka-pack findings … stay with their steps (P1-05/P1-16)") agrees.

**Smallest reproduction.** Any python unit with `from kafka import KafkaProducer; producer.send(SOME_IMPORTED_CONSTANT, o)` and the default packs. Scratch: `rca/t/*/bus/producer.py`.

**Fix.**
- (a1) `packrt/run.py:154-155` (`tdd-engineer`, **code**, before dispatch B): replace the raise with the keyless emission the next lines already produce for `runtime-built` (`key=None, status="possible"`), reason `config-derived` when the argument's point has a `fieldAccess` read (spec §1.4 text), else `runtime-built`. The assembler already drops a keyless provider with a counter (`assemble/rows.py:611` `provider-key-none`).
- (a2) Guard, authored **before** a1 so it can be red: a test that runs every pack in `RF.DEFAULT_PACKS` through `run_packs` over a synthetic bundle in which every pack-matched call has an all-`UNKNOWN` argument, and asserts no `UnsupportedSubject`. Axes: pack (from `DEFAULT_PACKS`) × `onUnknown` mode (read from each `pack.yaml`'s slots and the defaults of `run.py:_default_mode`) × edge (`provides`/`consumes`). A hand-listed set of shapes would miss shape N+1.
- Classification: **P1 defect to fix before P1-41 dispatch B** (default packs are the P1-20 build the gate names). After c2 is fixed the fixture would no longer raise for LNK-05/09 but a1 is still owed: `AC6-TOP-3`-style inputs and any repo with `settings.topic` reach the same branch.

## 3. Problem (b): `scripts` unit and the missing `http` links

### b1. `scripts` unit `not-mapped` / `extract:failed`

1. `discover_units` (`units.py:280`) makes a fallback `javascript` unit `fx-web:scripts` for the one file `scripts/build.test.ts`: the SK-8 row plants it deliberately "outside the unit's rootPath" (`app/`), so no manifest unit holds it.
2. `extractors/joern/run.py:374` stages it (1 non-empty file), the parse yields a CPG with no `METHOD`, the D15 content check answers `CONTENT=EMPTY`, and `run.py:406` raises `EmptyCpgError`, which `build.py:fail_code` maps to `extract:failed` (traceback captured with a patched `fail_code`).
3. Why the parse is empty: the staged dir contains only `build.test.ts`. Experiment (second method): `jssrc2cpg.sh` in the toolchain image over a dir holding the same bytes as `build.test.ts` gives a 4,580-byte CPG; as `build.ts` it gives 46,504 bytes. The frontend's AstGen ignores `*.test.ts` by default (its own log says "files that are ignored by default"), while D15's `stage_files` counts any non-empty source file.

SK-8 expects only "not in the unit list"; no row plants or expects this unit, so it is an **unplanned** not-mapped unit that will appear in every staleness block/notice (AC-9b) and could collide with a closed-world or AB-* count at dispatch B. It does not affect DF-3/RV-3 or any link. **Class:** minor P1 defect in the content check's file accounting (D15 should count only files the frontend parses, or a test-only unit should be skipped, not failed). **Fix:** `coder`, code (`extractors/joern/run.py` stage count) with a test using a test-only unit; alternatively `qa-engineer` declares it in the AB accounting. Needs a one-line ruling from `architect`/`teco` on which.

### b2. no confirmed `http` `key-match`, no LNK-36

Graph facts from the no-kafka build (`rcafx1`): 12 `PROVIDES http` edges, **all** `fx-svc` `api/*` handlers `possible/partial-key` with keys such as `GET /health`, `GET /search`, `GET /{}`, `* /POST`; only `job/`, `legacy/`, `tools/` handlers (plain `FastAPI()` apps, no router) are `confirmed`. The consumer side is fine: ContractPoints `POST /api/v1/orders`, `GET /api/search`, `GET /api/health` exist with `CONSUMES` edges. All 18 `http` Links are `partial-key` or `unknown-key`; none is `key-match`.

Chain:
1. `fx-svc/api/*.py` declare `router = APIRouter(prefix="/items")` at module level and `api/app.py` mounts with `app.include_router(health.router, prefix=API_PREFIX)`.
2. The `fastapi` pack's mount declaration is `child: {arg: a0, resolves: factory-call}` (`packs/fastapi/pack.yaml:12`): it resolves only `include_router(make_router(), …)`. `run_packs` over the fixture bundles with `fastapi` alone: `mounts facts: 0`. Every handler is `how: unmounted-router` (provider evidence), so `{mount}` is empty and the key lacks `/api`.
3. `constructorPrefix` (`pack.yaml:15`, `mount.py:_has_own_prefix`, `status()`): a router with its own prefix is `possible/partial-key` by design and its prefix is never composed into the key (`mount_alternatives` joins only the include chain). So even the factory form could not give `POST /api/v1/orders` `confirmed` while the prefix sits on `APIRouter(...)`.
4. Both limits are documented design: rule-pack spec §12.2 **O-3** ("module variable … unsupported in v1 (`possible`)"), §2.3 ("A router with its own prefix"), `docs/plans/connected-code-graph-p1.md` line 153 ("`possible` in v1 … revisit at P3"), P1-16 behaviour (9) ("asserted `possible`").
5. The fixtures plan and the P1-41 gate nevertheless plant exactly these forms as P1 `confirmed`/`key-match`: LNK-04 ("router `prefix="/v1/orders"` mounted at `/api` (mount composition)"), LNK-10, LNK-11, LNK-36, LNK-37 (each needs a mounted `fx-svc` provider), and `-p1-gates.md` P1-41 lists "mount composition" in the P1 scope. DF-3/RV-3 (both through LNK-36) and the AC-1 chain rows depend on them.
6. Separate, secondary: `@router.post("")` (empty route, the LNK-36 provider) produced key `* /POST` (decorator record `a0 = {const: ""}`; key template `"{verb} {mount}{route}"` renders `"POST "`; I did not trace the re-parse in `keynorm`, so *inferred*). Even with mounting solved, an empty route needs its own check.

**Second method.** The experiment of c1/c2 (scratch repos) does not exercise mounts; instead I confirmed independently: (i) `include_router` call sites have `resolution: external`, a0 is an attribute, never a call (bundle records); (ii) zero `MOUNTED_BY` facts out of `run_packs`. Fix proof is not run (it is a design choice, below).

**Class:** a **plan/fixture contradiction**, not a runtime defect: P1 declared the shape unsupported and the fixtures plant it as P1-confirmed. Needs an `architect` ruling before dispatch B, among:
- B-1 re-plant `fx-svc`: routers built by a factory (`def build_router(): r = APIRouter(); …`), `include_router(build_router(), prefix=API_PREFIX + "/v1/orders")` with no constructor prefix, and a non-empty route for the LNK-36 handler. Cheap, keeps LNK rows at P1; moves the O-3 shape out of P1 (add one `possible` row for it).
- B-2 keep the fixtures, re-gate LNK-04/10/11/36/37 (and the dependants DF-3, RV-3, the AC-1 chain) to P3 with their P1 form `possible/partial-key`. Cheap in code, but then P1 has no confirmed `http` key-match fixture, so AC-1/AC-23 get no fixture evidence at P1.
- B-3 implement module-level router support (variable→decorator edge; O-3 says a `-spec5`-class record or P1 mechanism): not a dispatch-B-sized change.
Recommendation: B-1 for dispatch B, B-3 stays at P3. Owner of the fixture edit: `qa-engineer`; manifests and test plan follow.

## 4. Problem (c): cross-file calls planted resolved have no `CALLS`

**Layout experiment (differential, one factor).** Three scratch repos with identical code (`core/flow.py::finalize` calling `eligible` imported from `core/rules.py`), differing only in layout/import form: `flat` (files at the repo root, `from fx_svc.core.rules import eligible`: the fixture's situation), `nested` (files under `fx_svc/`, same import), `rel` (files at root, `from core.rules import eligible`). Result in the loaded graph: `nested` → `CALLS finalize → fx_svc/core/rules.py::eligible`; `rel` → `CALLS … core/rules.py::eligible`; **`flat` → none**.

Chain:
1. `fx-svc/pyproject.toml` maps `package-dir = {"fx_svc" = "."}`, so the repo root *is* package `fx_svc` and every cross-file import is `from fx_svc.<pkg>.<mod> import …`.
2. The extractor (Joern `pysrc2cpg`, run over the unit's staged tree rooted at `.`) resolves an import by path from the unit root: `fx_svc/core/rules.py` does not exist. The call stays `resolution: external`, `externalTarget: "fx_svc/core/rules.py:<module>.eligible"`, `status: possible`, `reason: name-match` (`assemble_bundle.py:310-312`), `targets: []`; the assembler therefore has no `CALLS` edge, only `CALLS_NAME`. Observed: of 132 `fx-svc` call sites 15 resolve (the same-file ones plus the two below).
3. `hooks.find_hook` (`core/flow.py:13`) and `bus.publish_alert` (`:32`) resolve because they are calls on *untyped parameters*, not imports: Joern's `NaiveCallLinker` links a dynamic call by method name, and `assemble_bundle.py:308` turns any non-empty in-edge set into `confirmed`/`resolved-call`. (Consistent with the pass log; I did not isolate the linker.) That is also why the manifests' `name-match` plants for those two are met as `confirmed` by the frontend (report §4): a status-policy question for P1-05, independent of this RCA.
4. **SCIP is not in play.** `build.py:5-6` ("the P1-15 SCIP merge is not adopted, nothing here calls the indexer"); the P1-15 code is accepted but its measurement outcome is "not run: dependencies not provisioned" (ledger 2026-11-11), and `~/.local/share/ccg/provision` does not exist. P1 therefore runs Joern alone. Whether `scip-python` would resolve `fx_svc.*` is unmeasured (pyright would also need `fx_svc` on its path); no claim made.

**Second, layout-independent cause (c2).** The same three repos with a constant imported across files (`bus/constants.py::TOPIC_ORDERS`, used as `producer.send(TOPIC_ORDERS, o)`): in **all three layouts** the value set is `{ref: '<importing file>::TOPIC_ORDERS'}`, a qname with no constant record; `constant` records exist only for `constants.py`. Cause, `extractors/joern/normalize/core.py:293-300`: `Resolver.element` first tests `f"{path}::{n}" in self.constants.get(path)` (names Joern reports as module-level assignments of *this* file, which include the import-bound name), returns that `ref`, and never reaches the alias branch (`:301-309`) that would give `{ref: '<resolvedPath>::<name>'}`. Monkeypatch (second method): making `element` try the resolved alias first gives `OK … 'orders.created'` for `rel` and `nested` and still `RAISE` for `flat` (so c1 and c2 are both required; the RAISE is attributed to `flat` by elimination, as the only other `bus/producer.py` unit). No test pins an imported constant through the normalizer (`importRef` appears in `test_normalize_core.py`, but only for the unresolved branch; not traced further).

**Rows that depend.**
- c1 (resolved cross-file calls): `DF-2` (`create_order`→`apply_discount`), `CD-2` (`finalize`→`eligible`; caller `confirmed`), `RV-1` (`base_price`→`load_rate`, `is_member`; `is_member`→`connect`), `RV-2` (`eligible`→`load_rate`; `OrderCache.get`), `RV-3`/`CF-DF3` (`create_order`→`apply_discount`→`base_price`), `TW-3` (`fixed_dispatch`: `AuditPlugin().run`, `resolved-call`, gate P1; verified absent from the resolved set), plus the report's `notify`→`publish_created`. Not dependent: `DF-1`, `CD-1`, `CD-3` (outside bound), `LNK-30/39/40` (name-match plants). Watch item, unverified: once c1 is fixed, `accept_quote`'s `discount.apply_discount(…)` may gain a unique-name `CALLS` edge (point 3) and move from `possible/name-match` (DF-4) to `confirmed`; re-derive after the fix.
- c2: `LNK-06`, `LNK-37`, `LNK-04` (P1, "constant across files"), `LNK-05`, `LNK-09` (P2), and problem (a).

**Class and fix.**
- c2: **P1 defect, fix before dispatch B**; `tdd-engineer`, code, `normalize/core.py` `Resolver.element` (prefer the import alias when the name is an import-bound local, i.e. when the file does not itself define the constant), reproduction test first: normalizer fixture with a two-file unit and a constant imported across files, asserting the `ref` names `constants.py`, plus the `kafka`/`fastmcp` pack end-to-end (the experiment above). Coverage probe, not a list: parametrize over language (python, javascript) × use position (call arg, decorator arg, concat/template part, attribute `mod.CONST`) × import form (`from x import N`, `import x; x.N`, alias).
- c1: **fixture/layout, fix before dispatch B**; the plan commissions "resolved" for these rows with no mechanism that crosses the `package-dir` remap, and P1 has no indexer. Options: (i) `qa-engineer`: make the layout resolvable, either rewrite the ~20 `fx_svc.*` imports in `fx-svc` to unit-relative form or place the package under a real `fx_svc/` directory (the second changes every manifest qname: much costlier) and keep `python -m fx_svc.mcp.server` (LNK-21) as a string; (ii) `coder`: teach the normalizer/resolver `pyproject` `package-dir` (a real-world layout; backlog, not dispatch B); (iii) adopt `scip-python` (needs provisioning and the P1-15 measurement; not before dispatch B). Recommend (i) imports, plus (ii) as a backlog item. Re-planting is cheaper than leaving ~8 rows red for an extraction-layer reason the report already classes "extraction".
- Re-gate? c1 rows are not "expected at P1, re-gate": the plan's P1 expectation (resolved) is attainable with a conventional layout; they should stay P1 and pass after the fixture edit.

## 5. Order of work (dispatch B entry)

1. a2 test (red) then a1; c2 test (red) then fix, in either order (both `tdd-engineer`, different files).
2. `architect` ruling on b2 (B-1 recommended); then `qa-engineer` edits fixtures, manifests (LNK-36 locator already `OrderTotal.place`), `cross.yaml`, test plan; also c1 import rewrite in the same pass (same files).
3. b1 ruling; fix or declare.
4. Rebuild the fixture system with **default packs** (no pack removed); re-derive DF-2/CD-2/RV-1/2/3/TW-3 against the new graph, because the report's derivation B used the broken edges.

## 6. What is solid

- The report's engine-versus-derivation finding stands: the walk is not at fault; every difference sits in extraction (c1, c2, b2).
- The depth-bound and B1F rulings are untouched by this RCA.
- Ruled out: (i) `publish_dynamic` as the kafka trigger (no fact, no raise); (ii) a missing SCIP adoption as the cause of (c) (not called by the build; the layout experiment reproduces the gap and its absence with Joern alone); (iii) a walk or loader defect for the missing links (the `Link` nodes simply have no `key-match`); (iv) the `scripts` failure being a toolchain/sandbox fault (the same bytes parse when not named `*.test.ts`).

## 7. Unverified

- The `keynorm` mechanics of `* /POST` (inferred).
- That a1's emitted keyless provider is accepted by the P1 linker for every status (the assembler drops it with a counter; not run).
- `NaiveCallLinker` as the origin of the `hooks.find_hook`/`publish_alert` edges (consistent with the pass log, not isolated).
- That `scip-python` would or would not resolve `fx_svc.*`.
- The post-fix expected items of DF-4 and the link-dependent rows (no fix was applied).
