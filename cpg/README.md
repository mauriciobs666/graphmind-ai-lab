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
| `extractors/common/writer_rules.py` | the writer done-condition (P1-07, F-10), the only place an extractor decides W2, W3, W4, the `launchFact` argv through `gate_argv` (SA-1) and `valueHash`/`defaultHash` (an AST tripwire plus the `audit_records` backstop); and `bindParts` (only `bind_parts_fields`, gated by `bind_parts_allowed`; W6/W7 in `bind_value`); tests `tests/test_writer_rules.py`, `tests/test_valuehash.py`, `tests/test_bindparts.py` (spec5 §9 conformance + in-suite mutation pass) |
| `ccg/schemas/bundle-v1.schema.json` | per-record shapes (JSON Schema 2020-12), one `$defs` entry per record type: necessary, not sufficient |
| `ccg/broker/` | the egress broker (P1-29, `docs/plans/connected-code-graph-egress-spec3.md` §2–§5): `policy.py` rule table and mode resolution, `runs.py` tokens, `audit.py` the single writer of `egress.jsonl`, `addr.py` address validation and S6, `structured.py` the upstream exchange, `framing.py`, `server.py` the process, `client.py` `BrokerClient` (the only module another component imports) |
| `ccg/schemas/egress-audit-v1.schema.json` | the broker's `policy`, `decision` and `close` records, closed enums, P1 values only |
| `tests/` | durable-code tests |
| `infra/` | `ccg-falkordb.sh`, the dedicated FalkorDB instance (`infra/README.md`), and `toolchain/` |
| `.ccg-work/` | scratch for corpora, bundles and exports (gitignored) |
| `ccg/atrest.py` | bundle-at-rest and key custody (P1-02): `infra/README.md` § "Bundle-at-rest", which replaces `docs/plans/connected-code-graph-bundle-spec3.md` §13 F-1's default |
| `ccg/sysname.py` | the one module that forms system names, store keys and per-system ACL users (`docs/plans/connected-code-graph-p1-graph.md` §7); `tests/test_sysname.py` |
| `ccg/manifest.py`, `ccg/units.py` | system manifest loader (egress section checks S1-S5, `canonical_endpoint`) and unit discovery (P1-03); `ccg/system.yaml` is this repository's manifest (exclusions = the census scope of `eval/eval-manifest.yaml`), `fixtures/system1/system.yaml` the fixture system's; schemas `ccg/schemas/system-manifest-v1.schema.json` and `egress-v1.schema.json` (`docs/plans/connected-code-graph-egress-spec3.md` §4.2); the unit golden `tests/golden/system-units.json` is regenerated with `.venv/bin/python -m ccg.units --golden` |
| `extractors/joern/` | Joern raw-fact extraction (P1-04): `run.py` stages a unit (the commit's own tree through `git ls-tree` and `git cat-file`, never `git archive`, minus every descendant same-language unit root), then parse, content check (D15), astgen (javascript) and `extract.sc`, each through `sandbox/offline-run.sh --mode offline`, and returns the raw-facts path and `sandboxAuditIds`; it writes no bundle. One command (units come from the git index until P1-20, so run it with the index equal to the commit): `.venv/bin/python -m extractors.joern.run --root <rootPath> --language python\|javascript --commit <sha> --work <dir below .ccg-work/>`. Tests: `tests/test_joern_run.py`; the equivalence to the slice, `tests/slice/` (`-m slice`, skips when the gitignored oracle is absent; the Docker tier is also `live`) |
| `extractors/joern/normalize/` | Facts-to-bundle normalizer (P1-05), ported from the slice's `normalize.py` and equal to its stored bundles through an ordered list of named deltas (`tests/slice/test_normalize_equivalence.py`; a non-UTF-8 source file fails its unit with `SourceRefused`; the lambda-as-decorator record is not emitted; `ownedRegions` is emitted and fingerprinted when a frontend sets it): `core.py` (`Source`, `Scope`, `Builder`, name/path rules, scope numbering, `Resolver`, the CPython 3.12 check), `python.py` and `javascript.py` (scope tree, spans, imports, parameters per frontend), `blame.py` (`git blame` of the source repository, outside the sandbox; shallow clone refused; empty-file and span-plus-`ownedRegions` fallback), `assemble_bundle.py` (`assemble` raw facts -> records; `build_bundle(unit, raw, repo, commit, *, system, root, build)` writes only through `ccg.atrest.write_bundle`). Bundle bytes do not depend on fact order: `d`/`cb` are re-sorted before the first-wins `data` de-duplication and the G-4 filter. The header names the slice profile's normalizer (`ccg-slice-normalize` 0.2). Tests: `tests/test_normalize_{core,python,javascript,assemble}.py` (fixtures: `tests/normalize_helpers.py`, `tests/normalize_assembly_helpers.py`, `tests/fixtures/normalize_js/`); whole-bundle equivalence for both slice units, `tests/slice/test_normalize_equivalence.py` (`-m slice`, plus a `live` two-run test) |
| `extractors/shell/` | Shell extractor (P1-10, `tree-sitter-bash`, extra `extract`): `extract_shell(unit_root, files, *, repo_id, unit_id, history=None, warnings=None)` returns `file`/`symbol`/`param`/`callsite`/`valueSet`/`constant`/`data` records (no header; P1-44's assembly adds it): one `<script>` symbol per file, function symbols, `$1..$N`/`$@` params, a call site per command (`eval`, computed names, computed `source` and `${!n}` are `possible`/`shell-dynamic`), value sets with reaching definitions and `${v:-d}` alternatives. Writes no `configEntry`/`launchFact`; secret-shaped values are withheld; a file is parsed, never run; bounds and the parse-error limit are in `extractors/shell/README.md`. Tests: `tests/test_shell_extractor.py`, fixtures `tests/data/shell/` |
| `extractors/markdown/` | Prompt/Markdown extractor (P1-12): classifies `.md` as `prompt-md` or `doc` by the eval manifest's `promptBearingMarkdown` globs (the loader fails on an absent or changed manifest); a `doc` gets a `<document>` symbol only; a `prompt-md` gets mention call sites (`possible`/`prompt-mention`, bare tool names from an optional `known_tools` set), fenced code ignored. Tests: `tests/test_markdown_extractor.py`, fixtures `tests/data/markdown/` |
| `extractors/config/` | Config extractor (P1-11): `configEntry`/`launchFact` records from JSON/YAML/TOML/INI/`.env`, Compose, `.mcp.json`, Kiro agents and Dockerfiles (`extract_config`); one derivation row per spec3 §5.17 format (`formats.py`); `binding_role` (E3) here, every rule (W1-W4, argv gate, hashes, link-local withholding) via `extractors/common/writer_rules`; parsers are bounded and never construct or execute anything. Tests: `tests/test_config_extractor.py`, fixtures `tests/data/config/` |
| `packs/shell-mcp-call/` | Shell consumer of an MCP tool (P1-46): `mcp-call <tool>` and the unquoted `eval mcp-call <tool>` as `CONSUMES mcp_tool` (`call`, `eval-call` rules; a literal tool is `confirmed/key-match`, an unknown one `possible/runtime-built`); a file-only pack scoped by `callKind: command`, not in the rule-pack spec; `negatives.yaml` maps each rule to named shell snippets run through the real extractor; known limits (a leading flag is taken as the tool, prefix wrappers under-matched) are pinned by tests. Tests: `tests/test_pack_shell_mcp_call.py` |
| `extractors/common/errors.py` | the one `SourceRefused` every extractor raises (re-exported by `normalize.core`); message carries the path only |
| `ccg/packrt/` | the pack runtime (P1-16, `docs/plans/connected-code-graph-p1-linking.md`): `run_packs(bundles, packs) -> PackFacts` evaluates the declarative `packs/*/pack.yaml` over unit bundles; no pack-specific code (guarded by `tests/test_packrt_generic.py`); `urlBindings` are the linker's (P1-17). Tests: `tests/test_packrt_*.py`; equivalence to the slice oracle over the stored `s14a` bundles: `tests/slice/test_packrt_conformance.py` (`-m slice`; skipped when the gitignored oracle is absent, which is unrun, not green); `CCG_PACKRT_PACKS_FROM_SPEC=1` runs the runtime tests against the spec's pack blocks (`tests/test_rule_pack_leaf_mutation.py`) |
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
