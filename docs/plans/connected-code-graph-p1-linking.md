# Connected code graph — P1 plan, part 2: linking, assembly, store, refresh (W2)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1)

Part of `docs/plans/connected-code-graph-p1.md` (decisions D1–D11, obligation walk §4, step index §7). Contracts:
packs `docs/plans/connected-code-graph-rule-pack-spec.md`, graph `-graph.md` + `-p0-graph.md` (model v1, DDL,
load protocol §4, spike decisions §6), bundle `-bundle-spec3.md` + `-bundle-spec4.md`. The slice's `assemble.py`
hand-coded the FastAPI/fetch recognizers; the pack runtime replaces that, with the slice as the conformance oracle.
Briefing line for every step: dev items and fixtures only; no writes to any shared database (live tests on
`ccg-falkordb`, port 6390, or `infra/ccg-falkordb.sh --throwaway`).

## P1-16 · pack runtime — `tdd-engineer`

- **Files:** `cpg/ccg/packrt/__init__.py`, `match.py` (rule `match`: callee qualified name or regex, decorator,
  receiver type hint, config path, file glob or role), `extract.py` (argument, field path, keyword or config value
  from the resolved value sets), `lift.py` (role lifting to a fixpoint with ordered `LIFTED_VIA` evidence, B2),
  `mount.py` (mount composition and conditional mounts, B3), `cpg/tests/test_packrt_*.py` (one file per module),
  `cpg/tests/slice/test_packrt_conformance.py`. Over five files: two dispatches (`match`+`extract` first).
- **Interface:** `run_packs(bundles: list[list[dict]], packs: list[dict]) -> PackFacts` where
  `PackFacts = (providers, consumers, mounts, lifted, launch_facts, marks)`, every item carrying `status`, `reason`
  (a registry code, `ccg.rulepack.reason_row`) and its hop evidence. Packs are loaded only through
  `ccg.rulepack.load_pack` (which already requires `negatives.yaml` beside signature packs). **No pack-specific
  Python**: the runtime reads `cpg/packs/<name>/pack.yaml`; a `fastapi`-named branch is a defect.
- **Behaviours (red→green):** (1) a literal-key rule emits `confirmed/key-match`; (2) a key built from a
  `{ref}` resolves through value sets, unresolved parts give `possible/partial-key`; (3) role lifting: a
  parameter path flowing by `DATA` to a recognized client call's key argument makes the function a client on that
  path, to a fixpoint, with the chain kept in order (AC-1's `buildUrl.p0` and `apiFetch.p0.path` with verb
  `p0.method`); (4) shell functions lift the same way; (5) mount composition: decorator path + the prefix of
  whatever `include_router(prefix=…)` mounts it, a conditional mount keeps both variants with the condition;
  (6) **conformance:** the `fastapi`, `fetch` and `fastmcp` packs over the stored slice bundles give the slice's
  61 providers and 11 consumers with the same hops; the only differences are the two handlers of rule-pack spec
  §2.3 (`inner.get`, `_probe_rest_app`'s `router.post`) `possible/partial-key`; the two fixtures of §12.1 (a factory
  with `APIRouter(prefix=…)` mounted with another prefix; a function holding `FastAPI()` plus a local `APIRouter()`)
  give `possible`; (7) the `fastmcp` `tool` rule yields `PROVIDES mcp_tool` `confirmed/key-match` for the 25
  `mcp.py` decorators and lists the look-alikes O-8 names (never silently including them); the `tool-call` rule
  yields `CONSUMES mcp_tool`; (8) the 24 leaves the P0 leaf harness could only exercise at P1 are exercised
  (`cpg/tests/test_rule_pack_leaf_mutation.py` reports them); (9) router-in-a-variable (O-3) and the MCP inner
  path override (O-5) stay `possible`/prefix-only, asserted.
- **Also:** the three packs the runtime runs (`fastapi`, `fetch`, `fastmcp`) carry `status: paper`, the only value
  `cpg/ccg/schemas/rule-pack-v1.schema.json` allows (format v1.5 is frozen), and are pinned to the spec's YAML blocks
  by `cpg/tests/test_rule_pack_schema.py`. **`status: paper` stays**: it is a v1 format token, not a claim that no
  runtime exists; no schema, `ccg/rulepack.py` or pin edit is made, and the P1-46 pack uses the same value. The step's
  report notes that the token reads misleadingly once a runtime exists and leaves any change to a format successor. The 24 leaves of `cpg/tests/test_rule_pack_leaf_mutation.py` that only P1 can exercise
  leave its "unexercised" set, and `test_each_leaf_behaves_as_classified` passes with the set empty (the set is
  emptied by P1-16, P1-17 and P1-46 together).
- **Done / Gate:** `analyst` (execute the conformance test, do not read it; the report records the artifact sha256
  and `0 skipped`, index D9); mutation check on the lifting fixpoint and on status propagation.

## P1-46 · shell `mcp-call` consumer pack — `tdd-engineer`

No existing pack recognizes a shell consumer of an MCP tool: the `fastmcp` `tool-call` rule matches `call_tool`
under `scope.imports: mcp`, which a script never satisfies, yet fixture rows LNK-19 (`mcp-call reindex`, `confirmed`
`key-match`) and LNK-20 (`eval "mcp-call $TOOL"`, `possible`/`shell-dynamic`, linked to **every** `mcp_tool`
provider including `fx-job`'s second `reindex`, NC-5) gate at P1.
- **Files:** `cpg/packs/shell-mcp-call/pack.yaml`, `cpg/packs/shell-mcp-call/negatives.yaml`,
  `cpg/tests/test_pack_shell_mcp_call.py`, the leaf-harness registration in
  `cpg/tests/test_rule_pack_leaf_mutation.py`.
- **Behaviours:** a rule matching a call site of the command `mcp-call` in a shell `<script>` extracts the tool
  from argument `a0` and emits `CONSUMES mcp_tool` with the key; a literal gives `confirmed`/`key-match`; a value
  that is a shell variable or sits under `eval` gives `possible` with the registry's `shell-dynamic` code; the
  negatives file (required by `ccg.rulepack.load_pack` for signature-style packs; add it regardless) covers a
  comment, a string containing the word, and `mcp-call` as an argument to another command; the pack passes
  `ccg.rulepack.validate_pack`. If the pack format cannot express a shell-file scope (the `fastmcp` rules scope by
  imports), that gap is **returned to the architect as a finding before any engine change**, not worked around in
  the runtime.
- **Not a pack (D10):** Compose mappings and the `.mcp.json` launch chain are linker code (P1-17).
- **Gate:** `analyst`; mutation check on the literal/dynamic split.

## P1-17 · linker — `tdd-engineer`

- **Files:** `cpg/ccg/linker/__init__.py`, `matchers.py` (`exact`, `path-template`, `glob`/`prefix`, `regex`),
  `bind.py` (URL, env/Compose and launch bindings), `link.py` (`link(facts, bundles) -> Links`),
  `cpg/tests/test_linker_*.py`, `cpg/tests/slice/test_linker_conformance.py`. Two dispatches (matchers+link first).
- **Interface:** `Link = (linkId, kind, keyNorm, provider anchor, consumer anchor, status, reason, paramMap,
  evidence: ordered BOUND_BY and LIFTED_VIA entries)`; ids from `ccg.ids.link_id` (one `Link` per consumer
  **symbol**, `-bundle-spec3.md` BF6); `status` is the weakest hop.
- **Behaviours:** (1) both keys known and matching → `confirmed`/`key-match`; (2) partial or name-only → `possible`
  (`partial-key`, `name-only-provider`); (3) a fully unknown key with no binding → one `possible` link per
  provider of the kind (`unknown-key`), none collapsed; (4) **URL binding:** a consumer URL's port is matched to
  launch facts (explicit flags, env-defaulted values, framework defaults from the pack, Compose mappings), the
  path to the unit's mount paths; on `localhost`, several deployments of one unit (8000 and 8200 serving
  `falkor-chat/server`) are one provider and the link is `binding-confirmed`; two different units on one port are
  `possible`/`port-shared`; with `bindParts` (P1-14) the comparison is on `hostHmac`/`pathHmacs`; (5) env/Compose
  binding: base URL from env → Compose sets it per service → service `build.context` → unit; **the build context
  may lie outside its own repo (`../../fx-job`): the unit index is system-wide** and the case is the first real
  test of `-p0-compose-binding.md`'s inferred rule; if it fails the fallback is a bundle-spec image-name rule,
  which is escalated to `teco`, not patched in a fixture; (6) launch binding: `.mcp.json` server → command → script
  → container → unit (the AC-4 `cypher` chain); (7) `File.role = 'example'` makes every link through it
  `possible`/`example-config`; a prompt mention is `possible`/`prompt-mention`; documentation snippets are not
  callers; (8) a link into a unit never built is `possible`/`target-not-mapped`; (9) O-9: the reason on an
  `mcp_tool` Link when a binding selects the unit follows the `-p1-graph.md` ruling (the default in the spec is
  `key-match`); a framework-default port is evidence from the pack, represented as the ruling says; (10) MCP path
  matching is prefix-only (O-5); (11) a `Link` count at proving-repo scale is reported (the deferred "Link counts
  under runtime-built URLs at scale"); (12) **F-13:** every `http` key goes through `ccg.keynorm` and the RFC 3986 order (percent-decode before dot-segment resolution): a test asserts `GET /a/%2E%2E/b` becomes `GET /b` in `matchers.py`, never the old order.
- **Done / Gate:** conformance with the slice's 11 links and chains; the fixture rows LNK-* of the test plan that
  gate at P1 pass in P1-41. `analyst`; mutation check on status weakening and port-sharing.

## P1-18 · assembler — `graph-dba`

- **Files:** `cpg/ccg/assemble/__init__.py`, `rows.py` (bundles + links → `GraphRows`, ported from
  `spikes/vertical-slice/load.py:build_rows`), `snapshot.py` (`:Snapshot` content), `cpg/ccg/registry/
  flow-availability.yaml` (index D2), `cpg/tests/test_assemble_rows.py`, `cpg/tests/slice/
  test_assemble_equivalence.py`.
- **Behaviours:** (1) equivalence first: on the slice bundles the node and edge sets equal the slice's
  (38,121 nodes, 90,867 edges recorded in `-p0-graph.md` §1.1) plus only the deltas below; (2) D8's rule: `DATA`
  and `CALLS` store `status`/`reason` only when not the default; (3) `MOUNTED_BY {ord}` with `ord` by
  (mount path, line), provider `symbolId` and mount `callSiteId` as ids (the assembler requirement of
  `-p0-graph.md` D5); (4) `SkippedFile` nodes under `Unit -[:SKIPS]->`; (5) `Unit.flowAvailable` from the registry,
  one row per language (python false, javascript false, others false, kotlin absent = unmeasured), each row citing
  the fitness report section; **no other code reads a flag from anywhere else**; (6) the **assembler imports
  `ccg.keynorm.mask_http_segments` and a test fails if step 8a is defined anywhere else**
  (`-eval-protocol2-ml.md` §3); (7) a bundle whose header names a `sandboxAuditIds` entry with no `sandbox` record
  in the audit store is refused at assembly (egress §5.3 loader check; the record store is P1-29's — until it
  exists the check reads `cpg/.audit/` records the sandbox already writes); (8) the pack set's identity (ids,
  versions, content hash) is written where the `-p1-graph.md` ruling (O-10) says; (9) `candidateCount` is **not**
  stored (`-p0-graph.md` D6); (10) the system name is checked against the charset whitelist again here.
- **Done / Gate:** `analyst`; `graph-dba`'s own note is the contract. Two builds of the same bundles give identical
  rows (a determinism test).

## P1-19 · store loader, ACL, swap, stamp — `graph-dba`

- **Files:** `cpg/ccg/store/loader.py` (`load(store, rows, system, report)`; the slice's `LiveStore`, `zero_retry`,
  `batches`, `poll_constraints` ported), `cpg/ccg/store/acl.py` (`create_users(admin, system)`; the key patterns),
  `cpg/ccg/store/swap.py` (`swap(admin, system, build_id)`, the verdict-lock acquire/release with an **empty
  verdict log** and a no-op replay hook `replay_verdicts(...) -> 0` that P4 fills), `cpg/ccg/store/ddl_v1.cypher`
  (adds the `ConfigEntry.entryId` index of EP §5.2.1 HA-1 and any index `-p1-graph.md` adds, index-before-constraint
  order unchanged), `cpg/tests/test_loader_*.py`, `cpg/tests/test_acl_live.py`, `cpg/tests/test_fuzz_idhash_live.py`.
  Two dispatches (loader+swap; ACL+fuzz+measurements).
- **Behaviours (live where marked):** (1) indexes on the empty `ccg_<system>__build_<id>`, then nodes, then edges in
  `UNWIND` batches of at most `BATCH_MAX = 20,000` (never near the measured 100,000 ceiling), one connection,
  retries zero, a per-query `timeout` ≤ `TIMEOUT_MAX` whenever it is nonzero, a created-count check per edge batch;
  constraints after the load, polled to `OPERATIONAL`, a `FAILED` fails the build; `:Snapshot` last; then the lock,
  `RENAME build live` (a guarded `EVAL` only where a kept `prev` is wanted; **zero** old snapshots kept); (2)
  **ACL test on a throwaway instance** (`infra/ccg-falkordb.sh --throwaway`, P1-48; the test mutates ACL users, which `ccg-falkordb.sh status` treats as drift outside the four `ccg_*` users): `ccg_build` holds `~ccg_<s> ~ccg_<s>__*` plus the lock selector, and the test
  plants `shop` beside `shop2` and `shop` beside `shop__x`, asserting `ccg_shop` and `ccg_shop__build_1` are
  granted and `ccg_shop2` is denied, with the old flat `~ccg_shop*` pattern as the negative control (it grants
  `ccg_shop2`); a system named `shop__x` is **refused by the name whitelist** before any key exists, and the test
  shows the ACL pattern alone would grant `ccg_shop__x` (it matches `ccg_shop__*`), which is why the whitelist is
  required and not optional; `ccg_shop:verdict_lock` is reachable only through the key-scoped lock selector; the lock grants
  (`+set +del +eval`, `+get` on the single key; `ccg_build` also `+exists`; all four users `+info`); the
  `ccg_raw_<s>` user has `-@all +graph.ro_query +graph.explain +info ~ccg_<s> ~ccg_<s>__ov_*` and no `GRAPH.LIST`;
  (3) system names are validated against the whitelist before any key is formed; (4) **id-hash fuzz** (long,
  whitespace-edged and odd-character keys; a `nameKey` over 4,096 bytes must not reach a constraint unhashed)
  against a **throwaway instance only**; (5) a mid-load kill leaves the live key intact and the build key
  removable; (6) the loader refuses to run with a `shop`-prefixed key it does not own.
- **Measurements this step reports** (P1-scale replaces P0's): the worst per-edge-type p99 read latency on the
  loaded proving repo, from which `TIMEOUT_MAX` is **re-derived** with the P0 rule (10 × worst p99, rounded up
  in the 1-2-5 series; P0's 10,000 ms is a default, not a design constant); the lock ttl re-derived as 3× the
  worst (replay + `RENAME` + remainder check); the memory per file and the **~18 GB peak re-estimate**
  (`GRAPH.MEMORY USAGE` and `used_memory` delta, as the slice report did); `Link` counts at scale.
- **Corrected ACL line:** `-graph.md`'s `ccg_build` line (`~ccg_<system>*`) is superseded by the two-key pattern of
  proposal §3.5; the corrected line is written in `-p1-graph.md` (the graph note is not edited: it is `Extended by:`
  `-p0-graph.md` and executed against). `create_users` writes the four users' passwords to
  `cpg/.ccg-work/keys/<system>.acl` through `ccg.atrest` (P1-02), never to stdout or a committed file.
  **Gate:** `analyst` (the gate runs the planted ACL test itself, on
  `ccg-falkordb`); mutation check on the lock release and the ACL pattern.

## P1-44 · build dispatch by unit kind — `coder`

- **Files:** `cpg/ccg/build.py` (`build_unit(unit, repo, commit, system) -> BundleResult`), `cpg/tests/test_build_dispatch.py`,
  `cpg/tests/test_build_live.py`.
- **Behaviours:** (1) the unit's kind selects the extractor: python and javascript units → P1-04 run + P1-05
  normalizer (+ P1-15 SCIP merge when adopted); shell files → P1-10; JSON/YAML/TOML/Compose/`.mcp.json`/Kiro/
  Dockerfile → P1-11; Markdown → P1-12; each non-Joern extractor writes **one bundle per unit** through
  `write_bundle`, so a per-file extractor is run over the unit's files and merged into the unit's bundle; (2) an
  unknown kind is a typed error, never a silent skip; (3) a failure in one extractor fails **that unit's bundle**
  and nothing else; (4) bundles and keys are created through `ccg.atrest` only; (5) a test per kind asserts the
  extractor is actually invoked (a stub extractor that records calls), so no extractor is reachable only from its
  own tests.
- **Done / Gate:** `analyst`. **Needs:** 03, 04, 05, 10, 11, 12.

## P1-20 · refresh — `coder`

- **Files:** `cpg/ccg/refresh.py` (`refresh(system, scope=None) -> RefreshReport`), `cpg/ccg/staleness.py`
  (`unit_staleness(unit, head)`; the **one** staleness module, which `engine/` imports in P1-39), `cpg/tests/test_refresh.py`,
  `cpg/tests/test_refresh_live.py`.
- **Behaviours:** (1) per unit: `build_unit` (P1-44) inside its own try; a failure keeps the unit's last good
  bundle marked `stale-kept {built commit, failed at, reason}`; a never-built unit is `not-mapped`, and every link
  into it is `possible`/`target-not-mapped` (AC-9b); (2) units are built **one at a time** (6 GB host); (3) after
  all units: assemble (P1-18) → load and swap (P1-19) → a report (units fresh / stale-kept / not-mapped, wall time
  and peak memory per stage); (4) the staleness check compares a unit's built commit with the **local** head by
  commits or by files changed under the unit root; **there is no remote fetch in P1** (index §6): the manifest
  loader rejects a `remote` key, and the report says "head is the local clone's"; (5) a full rebuild is the default
  (bundle caching by tree id is not built; the lever if the budget fails); (6) `refresh` is the one entry point;
  only the on-demand trigger exists in P1 (schedule, threshold and CI are P3, AC-9a); (7) a shallow clone is a build
  error (D1).
- **Done:** a full refresh of the proving system on the dev host with **wall time and peak RSS recorded** (R-4); an
  unbuildable-unit fixture and a never-built unit behave as in (1). **Gate:** `analyst`.
