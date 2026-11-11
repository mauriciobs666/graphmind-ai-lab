# cpg — Code Property Graph and the connected code graph (C-1001)

`cpg/` holds two things:

- **Durable CPG reload artifacts** for the Joern-built `cpg_<component>` graphs, in
  `.cpg-artifacts/` (gitignored, local only). The `cypher` MCP server that queries them is the
  top-level `cypher-mcp/`, not this directory.
- **The `ccg` Python package**, C-1001 P0's contract code: ids, bundle schema and validators
  (`docs/plans/connected-code-graph-p0.md` §2 D2), plus the P0 infra and spikes around it.

## Setup

Python 3.12. The venv lives at `cpg/.venv`, which the repo-root `.venv` rule ignores.

```bash
cd cpg
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'          # default suite
.venv/bin/pip install -e '.[dev,live]'     # + falkordb-py, for live tests and store spikes
.venv/bin/pip install -e '.[dev,spike]'    # + py-tree-sitter and ten grammars, frontend-fitness spike
.venv/bin/pip install -e '.[dev,live,spike,extract,engine]'   # + P1 extractors and engine
.venv/bin/pip install -e '.[dev,broker]'        # the egress broker needs only the `dev` pins
```

Every dependency is declared once in `pyproject.toml`'s extras, at exact pins, by each phase's one
intake step (P0 plan §5.2; P1-01, `docs/plans/connected-code-graph-p1.md` D8); P1-03 and P1-29 each edit it
once more (D8). A step that finds a dependency missing stops and asks `teco`.

| Extra | Contents | Used by |
|---|---|---|
| `dev` | `pytest`, `jsonschema`, `pyyaml` | the default suite, and bundle, pack and manifest validation |
| `live` | `falkordb` 1.6.x (V17), `redis` | `live` tests; store spikes S13, S17, S20–S22 |
| `spike` | `tree-sitter` + grammars `python`, `typescript`/`tsx`, `javascript`, `java`, `kotlin`, `c_sharp`, `go`, `c`, `cpp` | S25–S27 frontend fitness (`docs/plans/connected-code-graph-p0-ml.md` §5.2, §11 N6) |
| `extract` | `tree-sitter` + `tree-sitter-bash` | P1 shell extractor (`docs/plans/connected-code-graph-p1-extraction.md` P1-10) |
| `engine` | `mcp` 1.28.1 (the SDK `cypher-mcp/` runs) | P1 engine MCP surface |
| `broker` | `jsonschema`, `pyyaml` (the `dev` pins verbatim) | P1 egress broker (`ccg/broker/`): manifests load through `load_manifest`, records validate against `egress-audit-v1.schema.json` |

## Tests

Always run with `cpg/` as the working directory. From the repo root, pytest picks the monorepo
as `rootdir` and ignores this `testpaths`.

```bash
cd cpg
.venv/bin/python -m pytest                                    # default: -m "not live"
CCG_FALKORDB_PORT=6390 .venv/bin/python -m pytest -m live     # needs infra/ccg-falkordb.sh up
```

`live` tests need Docker, the `ccg-falkordb` instance, or Joern. They are deselected by default.

## Layout

| Path | What |
|---|---|
| `pyproject.toml` | package `ccg`; pytest `testpaths=["tests"]`; marker `live` |
| `ccg/ids.py` | canonical tuple → 32-hex id: the one place hashing is defined (bundle spec §3.1–§3.2) |
| `ccg/bundle.py` | bundle JSONL read, validate and write with deterministic order; the reference validator, and the one implementation of the BS2 gates (bundle spec §2, §5, §8 as amended by BS2); `parse_bind_url` and the `bindParts` rules V1–V6 (spec5 §9) |
| `ccg/valuehash.py` | `entry_hmac(key, domain, value)`: the keyed, domain-separated hash of `-bundle-spec5.md` §2 (conformance: §9.1); computes only, never reads a key; the `bindParts` tags, `host_kind`, `bind_parts`, `bind_parts_usable`, `mount_prefix_hmac` |
| `extractors/common/writer_rules.py` | the writer done-condition (P1-07, F-10), the only place an extractor decides W2, W3, W4, the `launchFact` argv through `gate_argv` (SA-1; the one allow-listed exception is `mcp_exec_launch_path`, P1-11b) and `valueHash`/`defaultHash` (an AST tripwire plus the `audit_records` backstop); and `bindParts` (only `bind_parts_fields`, gated by `bind_parts_allowed`; W6/W7 in `bind_value`); tests `tests/test_writer_rules.py`, `tests/test_valuehash.py`, `tests/test_bindparts.py` (spec5 §9 conformance + in-suite mutation pass) |
| `ccg/schemas/bundle-v1.schema.json` | per-record shapes (JSON Schema 2020-12), one `$defs` entry per record type: necessary, not sufficient |
| `ccg/broker/` | the egress broker (P1-29, `docs/plans/connected-code-graph-egress-spec3.md` §2–§5): `policy.py` rule table and mode resolution, `runs.py` tokens, `audit.py` the single writer of `egress.jsonl`, `addr.py` address validation and S6, `structured.py` the upstream exchange, `framing.py`, `server.py` the process, `client.py` `BrokerClient` (the only module another component imports) |
| `ccg/schemas/egress-audit-v1.schema.json` | the broker's `policy`, `decision` and `close` records, closed enums, P1 values only |
| `tests/` | durable-code tests |
| `infra/` | `ccg-falkordb.sh`, the dedicated FalkorDB instance (`infra/README.md`), and `toolchain/` |
| `.ccg-work/` | scratch for corpora, bundles and exports (gitignored) |
| `ccg/atrest.py` | bundle-at-rest and key custody (P1-02): `infra/README.md` § "Bundle-at-rest", which replaces `docs/plans/connected-code-graph-bundle-spec3.md` §13 F-1's default |
| `ccg/sysname.py` | the one module that forms system names, store keys and per-system ACL users (`docs/plans/connected-code-graph-p1-graph.md` §7); `tests/test_sysname.py` |
| `ccg/manifest.py`, `ccg/units.py` | system manifest loader (egress section checks S1-S5, `canonical_endpoint`) and unit discovery (P1-03); `ccg/system.yaml` is this repository's manifest (exclusions and `promptBearingMarkdown` = the census scope of `eval/eval-manifest.yaml`, equality tested; `Manifest.prompt_markdown` exposes the globs), `fixtures/system1/system.yaml` the fixture system's; schemas `ccg/schemas/system-manifest-v1.schema.json` and `egress-v1.schema.json` (`docs/plans/connected-code-graph-egress-spec3.md` §4.2); the unit golden `tests/golden/system-units.json` is regenerated with `.venv/bin/python -m ccg.units --golden` |
| `ccg/gitsafe.py` | the one way to run `git` against a repository under analysis (spec6 security review m3): `git(repo, *args, paths=, timeout=, max_output=)`, `command()` and `environment()` for a streaming caller. No `GIT_*` variable, host/system configuration off, an empty `HOME`/`XDG_CONFIG_HOME`, `-c` overrides of the command-running settings (`core.fsmonitor`, `core.hooksPath`, `core.pager`, `core.sshCommand`, ...), `--no-replace-objects`, `--literal-pathspecs`, no shell, a timeout and a stdout cap; `GIT_ALLOW_PROTOCOL=none`, `GIT_NO_LAZY_FETCH=1`, a read-only subcommand allowlist with `--no-ext-diff --no-textconv` forced on diff readers, and refusal (`UnsafeRepository`) of a `.git` file/symlink, `commondir`, `objects/info/alternates` and `info/grafts`; `safe.directory` is deliberately not set. `units._git_files`, `blame.git` (and so `build.py`) and `extractors/joern/run.py` use it; a new git call goes through it (`tests/test_gitsafe.py`, which also guards that no module runs bare `git`) |
| `extractors/joern/` | Joern raw-fact extraction (P1-04): `run.py` stages a unit (the commit's own tree through `git ls-tree` and `git cat-file`, never `git archive`, minus every descendant same-language unit root), then parse, content check (D15), astgen (javascript) and `extract.sc`, each through `sandbox/offline-run.sh --mode offline`, and returns the raw-facts path and `sandboxAuditIds`; it writes no bundle. One command (units come from the git index until P1-20, so run it with the index equal to the commit): `.venv/bin/python -m extractors.joern.run --root <rootPath> --language python\|javascript --commit <sha> --work <dir below .ccg-work/>`. Tests: `tests/test_joern_run.py`; the equivalence to the slice, `tests/slice/` (`-m slice`, skips when the gitignored oracle is absent; the Docker tier is also `live`) |
| `extractors/joern/normalize/` | Facts-to-bundle normalizer (P1-05), ported from the slice's `normalize.py` and equal to its stored bundles through an ordered list of named deltas (`tests/slice/test_normalize_equivalence.py`; a non-UTF-8 source file fails its unit with `SourceRefused`; the lambda-as-decorator record is not emitted; `ownedRegions` is emitted and fingerprinted when a frontend sets it): `core.py` (`Source`, `Scope`, `Builder`, name/path rules, scope numbering, `Resolver`, the CPython 3.12 check), `python.py` and `javascript.py` (scope tree, spans, imports, parameters per frontend), `blame.py` (`git blame` of the source repository, outside the sandbox; shallow clone refused; empty-file and span-plus-`ownedRegions` fallback), `assemble_bundle.py` (`assemble` raw facts -> records; `build_bundle(unit, raw, repo, commit, *, system, root, build)` writes only through `ccg.atrest.write_bundle`). Bundle bytes do not depend on fact order: `d`/`cb` are re-sorted before the first-wins `data` de-duplication and the G-4 filter. The header names the slice profile's normalizer (`ccg-slice-normalize` 0.2). Tests: `tests/test_normalize_{core,python,javascript,assemble}.py` (fixtures: `tests/normalize_helpers.py`, `tests/normalize_assembly_helpers.py`, `tests/fixtures/normalize_js/`); whole-bundle equivalence for both slice units, `tests/slice/test_normalize_equivalence.py` (`-m slice`, plus a `live` two-run test) |
| `extractors/indexers/` | Indexer adoption (P1-15, code part; the corpus measurement and the `ccg-p1-indexer-v1` record are not built): `scip.py` parses a SCIP index (stdlib protobuf decoder; hostile input: size cap checked before reading into one `readv` buffer, caps on documents, occurrences and symbol infos counted while scanning so empty sub-messages raise `ScipError`, never `MemoryError`; linear per-position de-duplication with a 64-target cap; document paths are dictionary keys only and a path that could escape drops its document; no write-capable call in the module) to `(file, line, col) -> targets`, definitions and imports excluded; `run.py` runs one indexer at a time through `sandbox/offline-run.sh --backend container --mode offline --frontend scip-typescript\|scip-python` (never `joern`), in its own staged directory `<work>/indexer/stage` with the operator-provisioned dependencies copied in (entry and byte bounded; the stager rejects an absolute or escaping symlink, a hardlink, a device/FIFO/socket), `CCG_TOOLCHAIN_IMAGE` set to the `indexer` image per call (`indexer_image_tag()` prints the tag, builds nothing), under an outer wall-clock bound of `SBX_INDEXER_WALL_S` + 60 s + 30 s (TERM, then KILL of the process group, swept after every TERM and on any exception in the parent), runner output drained into capped buffers; runner exit 124 raises `RunnerTimeout`, with `watchdog` judged by the audit record's `durationMs` against the cap; absent dependencies return `Degraded` under the existing code `headers-unavailable` and run nothing; `load_provisioned(system, unit_id)` finds the operator's directory (see "Indexer dependencies" below). Tests: `tests/test_scip_parse.py` (golden `tests/data/scip/tiny.scip`, built by `tests/scip_helpers.py`; expansion probes under a 1.5 GB address-space limit), `tests/test_indexers_run.py` (stand-in runner; the `live` tests replay non-grammar argv through the real runner and index a toy project in the throwaway container), real-tool golden `tests/data/scip/real_ts.*` (scip-typescript 0.6.0 output, Joern bundle and call-site positions of the toy project in `real_ts_src/`) |
| `extractors/joern/normalize/merge_scip.py` | The SCIP merge rule of `docs/plans/connected-code-graph-p1-ml.md` §2.2 (P1-15): `merge_scip` upgrades or confirms a call site from SCIP's target; on a disagreement the site keeps `possible`/`name-match` with no target and both sides surface through `CALLS_NAME`, with `MergeResult.candidates` a merge-time report (not a bundle input); never drops a candidate; maps a class and `<constructor>` to the bundle's constructor symbol (`__init__`, `<init>`), counting `ctor_unmapped`; reports SCIP-only calls instead of minting records; leaves a site on a line with a character above U+FFFF before the callee alone (`astral_skipped`); `positions_from_bundle` builds the site positions from a bundle and `assemble`'s `report["callsitePositions"]`. Tests: `tests/test_scip_merge.py` |
| `extractors/shell/` | Shell extractor (P1-10, `tree-sitter-bash`, extra `extract`): `extract_shell(unit_root, files, *, repo_id, unit_id, history=None, warnings=None)` returns `file`/`symbol`/`param`/`callsite`/`valueSet`/`constant`/`data` records (no header; P1-44's assembly adds it): one `<script>` symbol per file, function symbols, `$1..$N`/`$@` params, a call site per command (`eval`, computed names, computed `source` and `${!n}` are `possible`/`shell-dynamic`), value sets with reaching definitions and `${v:-d}` alternatives. Writes no `configEntry`/`launchFact`; secret-shaped values are withheld; a file is parsed, never run; bounds and the parse-error limit are in `extractors/shell/README.md`. Tests: `tests/test_shell_extractor.py`, fixtures `tests/data/shell/` |
| `extractors/markdown/` | Prompt/Markdown extractor (P1-12): classifies `.md` as `prompt-md` or `doc` by the `promptBearingMarkdown` globs of the manifest path `load_globs(path)` is given (required, no default; production passes the system manifest, the eval manifest is an eval/fixture input; the loader fails on an absent or changed manifest); a `doc` gets a `<document>` symbol only; a `prompt-md` gets mention call sites (`possible`/`prompt-mention`, bare tool names from an optional `known_tools` set), fenced code ignored. Tests: `tests/test_markdown_extractor.py`, fixtures `tests/data/markdown/` |
| `extractors/config/` | Config extractor (P1-11): `configEntry`/`launchFact` records from JSON/YAML/TOML/INI/`.env`, Compose, `.mcp.json`, Kiro agents and Dockerfiles (`extract_config`); one derivation row per spec3 §5.17 format (`formats.py`); `binding_role` (E3) here, every rule (W1-W4, argv gate, hashes, link-local withholding) via `extractors/common/writer_rules`; a repo-root `.mcp.json` server launched as `bash`/`sh -c 'exec "$CLAUDE_PROJECT_DIR/<path>"'` emits `command = <path>` (P1-11b), every other `bash -c` string stays redacted; parsers are bounded and never construct or execute anything. Tests: `tests/test_config_extractor.py`, fixtures `tests/data/config/` |
| `packs/shell-mcp-call/` | Shell consumer of an MCP tool (P1-46): `mcp-call <tool>` and the unquoted `eval mcp-call <tool>` as `CONSUMES mcp_tool` (`call`, `eval-call` rules; a literal tool is `confirmed/key-match`, an unknown one `possible/runtime-built`); a file-only pack scoped by `callKind: command`, not in the rule-pack spec; `negatives.yaml` maps each rule to named shell snippets run through the real extractor; known limits (a leading flag is taken as the tool, prefix wrappers under-matched) are pinned by tests. Tests: `tests/test_pack_shell_mcp_call.py` |
| `extractors/common/errors.py` | the one `SourceRefused` every extractor raises (re-exported by `normalize.core`); message carries the path only |
| `ccg/packrt/` | the pack runtime (P1-16, `docs/plans/connected-code-graph-p1-linking.md`): `run_packs(bundles, packs) -> PackFacts` evaluates the declarative `packs/*/pack.yaml` over unit bundles; no pack-specific code (guarded by `tests/test_packrt_generic.py`); `urlBindings` are the linker's (P1-17). Tests: `tests/test_packrt_*.py`; equivalence to the slice oracle over the stored `s14a` bundles: `tests/slice/test_packrt_conformance.py` (`-m slice`; skipped when the gitignored oracle is absent, which is unrun, not green); `CCG_PACKRT_PACKS_FROM_SPEC=1` runs the runtime tests against the spec's pack blocks (`tests/test_rule_pack_leaf_mutation.py`) |
| `ccg/linker/` | the linker (P1-17, `docs/plans/connected-code-graph-p1-linking.md`): `matchers.py` (`exact`, `path-template`, `glob`/`prefix`/`regex` and pattern `keyForm`s over `keyNorm` strings that are never re-normalized, `http_key` for a raw URL in the RFC 3986 order, `mount_prefix_matches` for O-5's prefix-only mount comparison; a regex-key guard classifies a pattern with Python's own parser (`{,N}`/`{,}` are quantifiers) and refuses by default: no quantified group holding a quantifier or alternation, at most 2 unbounded-or-wide (span over 8) and 4 quantifiers of any kind, at most 4 alternations, a cost-model ceiling (product of the factors, x4 under `(?i)`/`(?s)`, 100,000), a text of at most 256 characters, and an 800-pair budget per link run (worst admitted pair measured 12.2 ms); refused patterns, refused over-long texts and pairs past the budget are counted apart), `link.py` (`link(facts, bundles, *, packs=(), bindings=None) -> Links`: one `Link` per (provider, consumer symbol) keyed by the provider's `keyNorm`, per-site `Via` status and reason by the O-9 rule, fully unknown keys fanned out to every provider of the kind, `shell-dynamic`/`example-config`/`prompt-mention` rules, tool mentions as consumers carrying their `server`, `Unmapped` refs checked value-free, `Binding.weaken` limited to `possible` chain reasons, `count_links`) and `bind.py` (`Binder`, the `Bindings` provider: URL, env/Compose and launch binding over `bindParts` under the system key, lexical only, link-local/metadata hosts never bindable (`valuehash.forbidden_target`, shared with the writer), a service is `(origin, repoId, path, service)`, `bind-key-mismatch <unitId>` lines, the loopback port residual (any Compose port fact) and `binding_duplicate_ids` returned beside the links; a record is accepted only if its id is the one its fields derive and its `repoId` is its bundle's, and a mention binds only through an `.mcp.json` of its own repository; launch facts indexed once, a malformed record skipped alone, each unit's link carrying only its own chain). Tests: `tests/test_linker_matchers.py`, `tests/test_linker_link.py`, `tests/test_linker_bind.py`, `tests/test_linker_bind_launch.py` (helpers `tests/linker_helpers.py`, `tests/linker_bind_helpers.py`), `tests/slice/test_linker_conformance.py` (the slice's 11 links, 8000/8200, the AC-4 `cypher` chain over pinned copies of `.mcp.json` and `cypher-mcp/Dockerfile` in `tests/data/linker_conformance/`; paper-record rows are suffixed `_on_paper_launch_facts`, the 8200 real-output row is a strict `xfail`; `slice`-marked, skips when `cpg/.ccg-work` bundles are absent) |
| `ccg/assemble/` | the assembler (P1-18, `docs/plans/connected-code-graph-p1-linking.md`; contract `docs/plans/connected-code-graph-p1-graph.md`): `build_rows(bundles, links, facts, system=..., packs=...) -> GraphRows` turns stored bundles, the linker's `Links` and the pack runtime's `PackFacts` into node and edge rows (a port of the slice's `load.py:build_rows`) and the `:Snapshot` row (`snapshot.py`: `bundleHash`, `codeHash` over `ASSEMBLY_CODE`, `packSetHash`, `manifestHash`). **Rows only: it writes to no database** (the loader is P1-19). It refuses a bundle the audit check, the version rule or the header schema rejects (`BundleRefused`, value-free code of `FAIL_REASONS`), a broken O-9 `Link`, a malformed `BOUND_BY`, and a provider that is not a `Symbol` (a `Link` provider outside the snapshot included); inputs are `bundle.read_bundle` output (record bodies are not re-checked; a malformed one raises a raw `KeyError`); only `launchUnit` `LAUNCHES` edges are written; `Unit.flowAvailable` comes only from `ccg/registry/flow-availability.yaml`; step 8a is imported from `ccg.keynorm`. Tests: `tests/test_assemble_rows.py`; equivalence to the slice (38,121 nodes / 90,867 edges plus the stated deltas) in `tests/slice/test_assemble_equivalence.py` (`-m slice`, skips when the gitignored bundles are absent: unrun, not green) |
| `ccg/sandbox_audit.py` | `check_sandbox_ids(header, audit_dir) -> list[str]` (P1-18; `docs/plans/connected-code-graph-egress-spec3.md` §5.3-§5.4): exactly one schema-valid (`schemas/sandbox-record-v1.schema.json`), `refused: false`, exit-0, `offline` record per `sandboxAuditIds` entry in the runner's `sandbox.jsonl`; no broker dependency; P1-19 calls the same function. Exit 124 is the watchdog or the tool's own exit and is refused either way (the error compares `durationMs` with the 1800 s cap) |
| `ccg/staleness.py` | the closed `FAIL_REASONS` of `Unit.failReason` (`docs/plans/connected-code-graph-p1-graph.md` §1 G4); P1-20 adds `unit_staleness` to this file |
| `ccg/build.py` | build dispatch by unit kind (P1-44; orchestration layer: with `ccg/refresh.py`, the only `ccg` modules that may import `extractors`, pinned by `tests/test_linker_bind.py` L1; `docs/plans/connected-code-graph-p1-linking.md`): `build_unit(unit, repo, commit, system, ctx)` picks the extractor (python/javascript -> Joern run + normalizer; shell -> P1-10; config -> P1-11; markdown -> P1-12), reads the tree of the built commit, writes one bundle through `ccg.atrest`, and returns a `BundleResult`: a failure fails that unit alone with one value-free code (`build:shallow-clone`, `extract:timeout`, `extract:failed`, `bundle:invalid`); an unknown kind raises `UnknownUnitKind` (P1-20 records it per unit as `extract:failed` and never aborts the refresh) and a `commit` that is not a full object id raises `InvalidCommit`. A non-Joern unit validates the paths it selects before writing anything (`..`, absolute, `.git`, backslash, control and over-long names fail that unit; an odd name outside the unit is never selected) and is bounded by the tree's blob sizes (`MAX_*`; a config blob over 8 MiB is skipped and counted, with no File record). `build_units` builds code units first, computes `known_tools` once (the `fastmcp` pack over the code bundles), then the rest. The host-extractor header is built from the merged records (`bind_header_fields`); `BuildContext.audit_ids` must be supplied for shell/config/markdown bundles. Tests: `tests/test_build_dispatch.py`; `tests/test_build_live.py` (opt-in, real runner) |
| `ccg/render/` | the answer renderers (P1-24, `docs/plans/connected-code-graph-p1-answers.md`): `render_text`/`render_markdown` turn a canonical answer into the grouped text and the Markdown impact report (deterministic templates; every `DATA_FIELDS` value goes through `answer.fence_data`, the F-4 fields through `answer.mask_identifier` first; minted fields such as reason codes, `endpoint`, `lang`, `repoId` and ports are printed as given, so callers run `answer.validate` before `render_*`), and `summary_input` is the aggregate-only text a plain-language summary may be built from (it validates its input and raises `AnswerError` otherwise). Tests: `tests/test_render.py`; goldens `tests/golden/render/` |
| `.cpg-artifacts/` | CPG reload artifacts (gitignored) |

The rest of plan §2 D2's tree (`sandbox/`, `packs/`,
`fixtures/`, `eval/`, `spikes/`) arrives with its own step.

## Ids, in one paragraph

`id(tuple) = sha256(canonical_json(normalize(tuple))).hexdigest()[:32]`. Every string is NFC
then stripped, and integers stay integers. JSON is compact UTF-8 with non-ASCII written literally.
The first element is always the record-type tag. Use the constructors (`symbol_id`,
`callsite_id`, `point_id`, …) rather than building tuples by hand. They supply the tag and the
element order, check each position's type (`str` or `int`), and reject a signature outside
Java, Kotlin, C# and C++, or an empty one. Passing a signature only for function-like symbols,
and always for those, is the caller's job (§4.3). The spec is
`docs/plans/connected-code-graph-bundle-spec3.md` §3, and its golden vectors GV-1–GV-22 are in
`tests/test_ids.py`.

## Bundles, in one paragraph

**The contract** is `docs/plans/connected-code-graph-bundle-spec3.md` (the consolidation of the
earlier spec and "BS2") read together with `docs/plans/connected-code-graph-bundle-spec4.md`
(spec4 §0). **The spec is normative and `ccg/bundle.py` is the reference
validator**: a disagreement between them is a `bundle.py` defect (BS2 §1.12).
`bundle-v1.schema.json` is **necessary, not sufficient**: it never rejects a spec-valid record,
and a consumer that runs only the schema is not a validator (it accepts, for example, a quoted
identifier `a1["method"]`, which would give one fact two ids, or a non-canonical IPv6
`bindValue`).

A bundle is `<unitId>.bundle.jsonl`: the `header`, then the 19 record types in §8.3 order, each
block sorted by `id`, one canonical-JSON line per record. `write_bundle(path, records)` sorts,
validates and writes atomically, so records can be passed in any order; it refuses
`secretSighting`, which v1 writers must not emit (§5.19). `read_bundle`/`loads` reject anything
that is not byte-canonical, then validate. Validation has three layers: the schema (fields,
types, tokens, per-record conditionals); framing and order (§8.1, §8.3, duplicate ids); and the
cross-record rules — every id recomputed through `ccg.ids` (`import`/`launchFact` from their
`ordinal`), in-bundle references, `header.counts`, `data` endpoint kinds, sorted arrays, vendored
paths, `nameNorm` (BS2 §1.2), `controlledBy` with `point ≠ cond` (§1.4), and ASCII-`IDENT` access
paths with one encoding per key (§1.1). The header's `counts` must match, but zero is legal, so a
header-only bundle is valid: the empty-CPG guard (risk R11) is S11's content check.

**The BS2 gates, one implementation each, shared by writer and validator:**
`screen_fails(text)` (§1.7.1, T1–T6), `name_fails(segment)` (§1.7.3, 25 substring and 11
whole-token words over `fold`), `bind_value_allowed(keyPath, value)` (§1.7.2's grammar, screened
per label and segment, plus the name rule and W1), and `gate_argv(argv)` (§1.6.1). A writer passes
`[command, *args]` through `gate_argv`; the validator accepts a `launchFact` argv iff it is a fixed
point of the gate (sentinels stay as written, and an element after a forcing sentinel must be
redacted). `launchFact.default` is accepted only when it passes the value grammar and `envVar` the
name rule; `defaultHash` is required for `env-default`. Writer-only rules (W2–W4, the per-entry
HMAC) are not checked: the validator holds no key and sees only the emitted text.

**Tests read the contract's text.** `tests/test_bundle.py` parses the field tables, enum tokens,
reason registry and §8.3 order from the spec, the replaced tables (`import` §1.5, `launchFact`
§1.6) from BS2, and every AV-1–AV-68 and BV-1–BV-28 row from BS2 mechanically; none is retyped.
§1.6.3's probe runs verbatim: the 17,280-cell grid and the 6,480-cell separator axis (0 values in
clear) and the 675-cell BS-M1 transplant (only BV-11's bare positional in clear).

## Egress broker

One host process per user; a component reaches it only through `ccg.broker.client.BrokerClient`, and a missing or
latched broker fails the call (there is no direct-connection fallback).

```bash
cd cpg
.venv/bin/python -m ccg.broker.server --system ccg/system.yaml      # socket ~/.local/state/ccg/broker/broker.sock
CCG_AUDIT_DIR=/path .venv/bin/python -m ccg.broker.server --system m.yaml --socket /path/b/broker.sock
```

It re-executes itself isolated (`-I`) under a closed environment, so run it as above rather than through a wrapper
that depends on `PYTHONPATH`. Exit 64: usage or configuration (manifest, S1–S6); exit 69: a start precondition (lock
held, live socket, directory or audit-file check). Records go to `cpg/.audit/egress.jsonl` (gitignored) unless
`CCG_AUDIT_DIR` is set. The broker's tests start private instances under a temporary directory and never touch the
real socket directory.

## Indexer dependencies (operator-provisioned)

The SCIP indexers (`extractors/indexers/`, P1-15) run only when the **operator** has already installed the target's
dependencies; no `cpg/` module runs `npm`, `pip` or `uv` (`docs/plans/connected-code-graph-egress-spec3.md` §7.4). Per unit,
put them in `<CCG_PROVISION_DIR>/<system>/<unitId>/` (default `~/.local/share/ccg/provision`):

- `scip-typescript`: a `node_modules/` directory, installed with `npm ci --ignore-scripts`.
- `scip-python`: the venv directory and an `env.json` in `scip-python`'s `--environment` format.
- optional `provision.json`: `{"envJson": "env.json", "lockfile": "package-lock.json"}`, naming the `--environment` file
  (default `env.json`) and the lockfile that is hashed for the pre-registration record.

The tree must hold no symlink that is absolute or leaves the directory, no hardlink and no device or FIFO file. With
nothing there the build runs Joern alone and records the existing `headers-unavailable` degradation. This directory is
**deliberately separate from `~/.config/ccg`** (HMAC keys and ACL credentials): `cpg/` never reads or writes there and
refuses a provision location under it. `cpg/` only reads the provision directory; it is never written by `cpg/`.
