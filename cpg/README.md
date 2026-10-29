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
```

Every P0 dependency is declared once in `pyproject.toml`'s extras, at exact pins. **No other
step edits that file.** A step that finds a dependency missing stops and asks `teco` (plan §5.2).

| Extra | Contents | Used by |
|---|---|---|
| `dev` | `pytest`, `jsonschema`, `pyyaml` | the default suite, and bundle, pack and manifest validation |
| `live` | `falkordb` 1.6.x (V17), `redis` | `live` tests; store spikes S13, S17, S20–S22 |
| `spike` | `tree-sitter` + grammars `python`, `typescript`/`tsx`, `javascript`, `java`, `kotlin`, `c_sharp`, `go`, `c`, `cpp` | S25–S27 frontend fitness (`docs/plans/connected-code-graph-p0-ml.md` §5.2, §11 N6) |

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
| `tests/` | durable-code tests |
| `infra/` | `ccg-falkordb.sh`, the dedicated FalkorDB instance (`infra/README.md`), and `toolchain/` |
| `.ccg-work/` | scratch for corpora, bundles and exports (gitignored) |
| `.cpg-artifacts/` | CPG reload artifacts (gitignored) |

The rest of plan §2 D2's tree (`ccg/bundle.py`, `ccg/schemas/`, `sandbox/`, `packs/`,
`fixtures/`, `eval/`, `spikes/`) arrives with its own step.

## Ids, in one paragraph

`id(tuple) = sha256(canonical_json(normalize(tuple))).hexdigest()[:32]`. Every string is NFC
then stripped, and integers stay integers. JSON is compact UTF-8 with non-ASCII written literally.
The first element is always the record-type tag. Use the constructors (`symbol_id`,
`callsite_id`, `point_id`, …) rather than building tuples by hand. They supply the tag and the
element order, check each position's type (`str` or `int`), and reject a signature outside
Java, Kotlin, C# and C++, or an empty one. Passing a signature only for function-like symbols,
and always for those, is the caller's job (§4.3). The spec is
`docs/plans/connected-code-graph-bundle-spec.md` §3, and its golden vectors GV-1–GV-22 are in
`tests/test_ids.py`.
