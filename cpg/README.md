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
.venv/bin/pip install -e '.[dev,live,spike,extract,engine]'   # + P1 extractors and engine (broker: stdlib)
```

Every dependency is declared once in `pyproject.toml`'s extras, at exact pins, by each phase's one
intake step (P0 plan §5.2; P1-01, `docs/plans/connected-code-graph-p1.md` D8). **No other step edits
that file.** A step that finds a dependency missing stops and asks `teco`.

| Extra | Contents | Used by |
|---|---|---|
| `dev` | `pytest`, `jsonschema`, `pyyaml` | the default suite, and bundle, pack and manifest validation |
| `live` | `falkordb` 1.6.x (V17), `redis` | `live` tests; store spikes S13, S17, S20–S22 |
| `spike` | `tree-sitter` + grammars `python`, `typescript`/`tsx`, `javascript`, `java`, `kotlin`, `c_sharp`, `go`, `c`, `cpp` | S25–S27 frontend fitness (`docs/plans/connected-code-graph-p0-ml.md` §5.2, §11 N6) |
| `extract` | `tree-sitter` + `tree-sitter-bash` | P1 shell extractor (`docs/plans/connected-code-graph-p1-extraction.md` P1-10) |
| `engine` | `mcp` 1.28.1 (the SDK `cypher-mcp/` runs) | P1 engine MCP surface |
| `broker` | none (stdlib only, unless P1-28 decides otherwise) | P1 broker |

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
| `ccg/bundle.py` | bundle JSONL read, validate and write with deterministic order; the reference validator, and the one implementation of the BS2 gates (bundle spec §2, §5, §8 as amended by BS2) |
| `ccg/schemas/bundle-v1.schema.json` | per-record shapes (JSON Schema 2020-12), one `$defs` entry per record type: necessary, not sufficient |
| `tests/` | durable-code tests |
| `infra/` | `ccg-falkordb.sh`, the dedicated FalkorDB instance (`infra/README.md`), and `toolchain/` |
| `.ccg-work/` | scratch for corpora, bundles and exports (gitignored) |
| `ccg/atrest.py` | bundle-at-rest and key custody (P1-02): `infra/README.md` § "Bundle-at-rest", which replaces `docs/plans/connected-code-graph-bundle-spec3.md` §13 F-1's default |
| `ccg/sysname.py` | the one module that forms system names, store keys and per-system ACL users (`docs/plans/connected-code-graph-p1-graph.md` §7); `tests/test_sysname.py` |
| `ccg/manifest.py`, `ccg/units.py` | system manifest loader (egress section checks S1-S5, `canonical_endpoint`) and unit discovery (P1-03); `ccg/system.yaml` is this repository's manifest (exclusions = the census scope of `eval/eval-manifest.yaml`), `fixtures/system1/system.yaml` the fixture system's; schemas `ccg/schemas/system-manifest-v1.schema.json` and `egress-v1.schema.json` (`docs/plans/connected-code-graph-egress-spec3.md` §4.2); the unit golden `tests/golden/system-units.json` is regenerated with `.venv/bin/python -m ccg.units --golden` |
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
