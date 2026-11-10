# Connected code graph — P1 plan, part 1: extraction (W0 intake, W1)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1)

Part of `docs/plans/connected-code-graph-p1.md` (the index: decisions D1–D11, the flow gate §5, the obligation
walk §4, the step index §7). Citations "index D<n>" and "index §<n>" point there. Contracts: bundle
`docs/plans/connected-code-graph-bundle-spec3.md` + `-bundle-spec4.md`, rule packs `-rule-pack-spec.md`,
egress and sandbox `-egress-spec3.md`. Layout is proposal §3.13: extractors under `cpg/extractors/<name>/`,
shared code under `cpg/ccg/`. Every step's briefing line: builders use `cpg/eval/dev/` and the fixtures, never held-out
items (index §2); live tests run on a throwaway instance (`infra/ccg-falkordb.sh --throwaway`), or on `ccg-falkordb` (6390) only where a step says so.

Every extractor writes bundles through `ccg.atrest.write_bundle` (P1-02), which validates through `ccg.bundle.dumps` and creates a `0600` file under `bundles/<system>/`; no extractor imports another's
internals. Frontend quirks live in the extractor's normalizer and nowhere downstream (proposal §3.2).

## P1-01 · layout and Python dependency intake — `devops`

- **Files:** `cpg/pyproject.toml` (extras `extract`: `tree-sitter-bash`, pinned and verified installable like P0's;
  `engine`: the `mcp` SDK pin used by `cypher-mcp/`; `broker`: left empty here and filled by P1-29 (index D8);
  `packages.find` widened to `include = ["ccg*", "extractors*", "clients*"]`, index D8; the `slice` marker registered,
  D9), `cpg/extractors/__init__.py` and `cpg/clients/__init__.py` (empty package stubs), `cpg/eval/tools/audit.py`
  (O3), a derived `absent-tool` stream under `cpg/tests/fixtures/headless/`, `cpg/README.md` (extras rows).
- **O3 ruling:** a call to an absent tool (`<tool_use_error>… No such tool available`, nothing executed) is
  recorded `refused: true, reason: absent-tool` and does not void the session; tested with the planted derived stream
  (as the `derived.mcp-call.*` streams were).
- **Behaviours:** (1) `pip install -e '.[dev,live,spike,extract,engine]'` succeeds and `python -c "import extractors,
  clients"` resolves the two stubs; (2) the full `cpg` default suite is green before and after (R-7); (3) the O3
  stream is counted refused and the session is not void.
- **Gate:** `analyst`.

## P1-48 · instance, indexer image and throwaway mode — `devops`

- **Files:** `cpg/infra/toolchain/Dockerfile` and `build.sh` (a **new target `indexer`**, `FROM toolchain`, adding
  Node 24 and the pinned `scip-python`/`scip-typescript` from the `truth-npm` lockfile; the default `toolchain` image
  the sandbox runs stays free of Node and indexers, so the AC-8 "no dependency download" surface does not widen for
  builds that need no indexer; the `truth` target is not reused because it also carries the truth corpora's tools and
  smoke projects), `cpg/infra/ccg-falkordb.sh`, `cpg/infra/tests/test-ccg-falkordb.sh`, `cpg/infra/README.md`. `offline-run.sh` reads only
  `CCG_TOOLCHAIN_IMAGE`, so the indexer runner (P1-15) sets that variable to the indexer image per call; there is no
  second variable. The runner's indexer entries are P1-48b's.
- **Behaviours:** (1) `ccg-falkordb.sh up` reports `TIMEOUT_MAX` 10000 and `MAX_INFO_QUERIES` 0 from
  `GRAPH.CONFIG GET` (`-p0-graph.md` §6 rows 5, 12); (2) **`status` and its test change:** a nonzero `TIMEOUT_MAX` and
  the per-system `ccg_<role>_<s>` ACL users are the P1 steady state, a user outside that set is still drift;
  `status`'s allowed-user check is `default` or `ACL_USER_RE` (`^ccg_(build|verdict|engine|raw)_[a-z0-9]+([-_][a-z0-9]+)*$`)
  plus a separate 1–48-character check on the system part, and a test
  asserts it gives the same verdict as `ccg.sysname.acl_user_re()` over every role × valid name, malformed names, `default` and 48- and 49-character system names (neither regex holds the length; `-p1-graph.md` §7; the test is P1-03's, `cpg/tests/test_sysname.py`, which reads the
  script); the script header and the
  P0-scoped `reset-p0` verb text are updated; the README states the instance has no persistence (`save ""`, no volume),
  so the first query after a restart is a rebuild; (3) **`--throwaway`** starts a disposable instance on a free port,
  labels its container `ccg-falkordb=1`, and **refuses ports 6379 and 6390** and any port already published; (4) the
  `indexer` image builds, carries the label `ccg-toolchain-indexer=1`, and `ccg-versions` names both indexers offline
  (`cpg/infra/toolchain/tests/test_toolchain.sh`, section 5).
- **Gate:** `analyst`.

## P1-48b · the runner's indexer entries — `devops`

`-egress-spec3.md` §7.4–§7.6, §7.8, §10.1 tests (21)–(26), §10.2. **Needs:** P1-28, P1-48. **Blocks:** P1-15.
- **Files:** `cpg/sandbox/offline-run.sh` (`SBX_FRONTENDS` gains `scip-typescript` and `scip-python`, their fixed
  absolute executables, the `index` argv grammar, the `ccg-toolchain-indexer=1` image-label check, `PYTHONSAFEPATH=1`
  in the indexer pass-list, `--output /out/<auditId>.scip`, the §7.5 (4) and `usage` extensions),
  `cpg/sandbox/allowed-flags.txt`, `cpg/sandbox/forbidden-flags.txt`, `cpg/sandbox/tests/test_sandbox.sh`
  (`tests/mutants.sh` and `cpg/sandbox/README.md` as the spec's tests and citations need); `cpg/infra/README.md` and
  `cpg/infra/toolchain/README.md` stop saying the runner has no indexer entry.
- **Behaviours:** the spec's tests (21)–(26) pass, the sandbox suite and its mutants re-run in full (a `--frontend
  joern` script that runs `scip-typescript` proves only that the image works and is not a check of this step).
- **Gate:** `analyst`; `security-expert` advisory on the sandbox boundary.

## P1-02 · bundle-at-rest and HMAC-key custody (F-1) — `devops`

- **Decide and implement:** (a) bundle files `0600`, directory `0700`, under `cpg/.ccg-work/bundles/<system>/`
  (gitignored; check `cpg/.gitignore`); (b) retention: the last good bundle per unit plus the one being built;
  a refresh deletes older ones; (c) **no LLM-backed tool reads a bundle**: the plain-language front and every
  agent read answers and `raw_query` only; enforce by a test that applies the **glob-containment logic of
  `cpg/eval/tools/audit.py` (`pattern_escapes`)** to the headless-agent settings and the MCP tool definitions, with the
  **root named** (the headless kit directory, since `pattern_escapes(root, base_dir, pat)` answers whether a glob can
  name something outside `root`) and planted `Read(cpg/**)` and `Read(cpg/.ccg-work/**)` as positive controls (both
  flagged; any other pattern outside the root fails closed); the
  residual for an unrestricted-`Read` interactive agent is stated, not closed; (d) the per-system HMAC key (for `valueHash`,
  `bindParts`, §1.7.5) in `~/.config/ccg/keys/<system>.hmac` (outside the repository: `atrest` refuses a key area
  inside it), `0600`, generated once, never logged, never in a bundle, an answer or the broker audit, read only
  through `ccg.atrest.hmac_key`, which refuses the two published test keys of `-bundle-spec5.md` §9; rotation = a
  full rebuild (it changes every `valueHash`); (e) the four `ccg_<role>_<system>` credentials that P1-19's
  `create_users` mints live in `~/.config/ccg/keys/<system>.acl` (`0600`, one line per user, written and read
  through `ccg.atrest.write_acl`/`read_acl`), read by the clients through `CCG_CREDENTIALS_DIR` (also refused inside
  the repository) and never placed in a committed file: the repo-root `.mcp.json` entry P1-27 proposes carries no
  credential.
- **Files:** `cpg/infra/README.md` (section "Bundle-at-rest"), `cpg/ccg/atrest.py` (bundle area: `bundle_dir`,
  `write_bundle`, `read_bundle`, `prune_unit`; key area: `key_path`, `hmac_key`, `bind_key_check`, `acl_path`,
  `write_acl`, `read_acl`, `credentials_dir`; `ensure_private(path, area)`, which refuses a path outside its area,
  a symlink component and an exposed file, and creates with `0700`/`0600`),
  `cpg/tests/test_atrest.py`, `cpg/.gitignore`.
- **Behaviours:** a created bundle dir/key has the modes above; `ensure_private` on a world-readable existing
  file raises; a path outside its area (the work area for bundles, `~/.config/ccg/keys/` for keys and credentials)
  is refused; the guard test fails if a tool allow-list names the
  bundle dir. **Done:** the default stated in `-bundle-spec3.md` §13 F-1 ("P0 bundles live only in the gitignored
  work area") is replaced by this decision, cited from `cpg/README.md`. **Gate:** `analyst`; `security-expert`
  advisory. **Blocks:** P1-07, P1-19, P1-20, P1-41.

## P1-03 · system manifest and unit discovery — `tdd-engineer`

- **Files:** `cpg/ccg/manifest.py`, `cpg/ccg/units.py`, `cpg/ccg/schemas/system-manifest-v1.schema.json`,
  `cpg/ccg/schemas/egress-v1.schema.json` (verbatim from `-egress-spec3.md` §4.2),
  `cpg/ccg/sysname.py` (the one module for system-name, key and ACL-user construction, specified in `-p1-graph.md` §7;
  P1-18, P1-19, P1-21 and P1-25 import it) and the format-sweep and regex-equality test
  `cpg/tests/test_sysname.py` of `-p1-graph.md` §7, `cpg/tests/test_manifest.py`, `cpg/tests/test_units.py`.
- **Interface:** `load_manifest(path) -> Manifest` (repos with `repoId`, path, exclusions; a `remote` key is rejected, below;
  `groupings` rejected until P2 with a clear error; policy: `egress` (`local-only` | `hosted`, mandatory per
  repository, `-egress-spec3.md` §4.4), `rcaWindow`, `secretsWindow`, `previewCap`; the `egress` section is the
  system's only allow-list, validated against `egress-v1.schema.json` with checks S1–S5 and `canonical_endpoint`
  of `-egress-spec3.md` §4.3); `discover_units(repo_root, manifest) -> list[Unit]`
  where `Unit = (unitId, repoId, rootPath, language)` from `pyproject.toml`/`package.json`/`tsconfig.json`
  manifests, a directory-per-language fallback, and per-file shell/config/Markdown units handled by the non-Joern
  extractors (proposal §3.2). `unitId` and `repoId` come only from `ccg.ids`. `cpg/ccg/system.yaml` is this repo's
  manifest: repo `graphmind-ai-lab` (the id `-bundle-spec3.md` §3.4 fixes). **Its exclusions are the census
  scope** the held-out keys were written under: `cpg/`, `deprecated/`, `docs/**/connected-code-graph*`,
  `docs/BACKLOG.md` and the content-check files `cpg/eval/eval-manifest.yaml` lists (`exclusions.paths`,
  `exclusions.contentCheck.files`); a test asserts the two lists are equal, so a drift fails loudly. The fixture
  system has its own manifest (`cpg/fixtures/system1`, four repos) and is built as a separate system. A `remote`
  key on a repository is **rejected** with "git fetch is not enabled in P1" (index §6).
- **Behaviours (red→green):** (1) a manifest with an unknown key, an absent `repoId`, or a system name that fails
  `ccg.sysname.check_system_name` (`-p1-graph.md` §7) is rejected; (2) a repo with one `pyproject.toml`
  gives one python unit with the right `rootPath`; (3) `salesperson/` gives a javascript unit; (4) a directory
  with `.py` files and no manifest falls back to a directory unit; (5) nested manifests give two units, not one;
  (6) exclusions remove units, and no unit under `cpg/` is discovered for the proving system; (6b) a manifest naming a `remote` is rejected; (7) unit ids are stable across runs; (8) the real repo yields exactly the units
  the slice used (`falkor-chat/server`, `salesperson`) among its list; (9) (`-egress-spec3.md` §10.1) S1–S5 each
  refuse a planted manifest, `policy.egress` is mandatory on every repository (`cpg/ccg/system.yaml` and the fixture
  manifest carry it), and its enum equals `$defs/egressMode`.
- **Done:** `discover_units` on the monorepo is committed as a golden list. **Gate:** `analyst`.

## P1-04 · Joern extractor port — `graph-dba`

- **Files:** `cpg/extractors/joern/extract.sc` (from `spikes/vertical-slice/extract.sc`), `.../astgen.sc`,
  `cpg/extractors/joern/run.py` (replaces `run_extract.sh`: stage via `git archive`, parse, content check,
  astgen for javascript, extract, all through `cpg/sandbox/offline-run.sh --mode offline`; returns the
  `sandboxAuditIds`), `cpg/extractors/joern/__init__.py`, `cpg/tests/test_joern_run.py`,
  `cpg/tests/slice/test_extract_equivalence.py`.
- **Behaviours:** (1) first dispatch only: regenerate the slice's raw facts at the pinned commit
  `defca60ff16534aa12a74e35856de9be98bfc710` if `cpg/.ccg-work/vertical-slice/` lacks them, record their sha256
  in the test; (2) `run.py` assembles the exact `offline-run.sh` argv for each stage and a unit test asserts no
  forbidden flag (`cpg/sandbox/forbidden-flags.txt`) is present and `--mode offline` is set (live tests are
  marked `live`); (3) the ported `extract.sc` run on the slice corpus yields raw facts equal to the slice's (a
  `slice`-marked test comparing canonical JSON, with the artifact sha256 and `0 skipped` recorded, index D9); (4) the D15 content check (a parse is
  certified by content, `cpg/sandbox/scripts/cpg-content-check.sc`) refuses an empty CPG (risk R11); (5) staging excludes from a
  unit's file set every path under a descendant unit root of the **same language** (`ccg.units` overlaps: a fallback unit over a
  manifest unit when forced, and a fallback over a fallback), so no file is staged in two bundles; a test pins both overlaps.
- **Done:** one command builds raw facts for a unit; behaviour is unchanged from the slice (the improvement
  happens in P1-09). This step writes no bundle: bundle writing, and its `0600` mode test, belong to P1-05 (5b) for
  the Joern extractors and to P1-44 for the others. **Gate:** `analyst`. **Risk:** the 6 GB host; run units one at a time.

## P1-05 · normalizer port (Python and TS/JS) — `tdd-engineer`

Two dispatches (the slice's `normalize.py` is 1,244 lines).
- **5a files:** `cpg/extractors/joern/normalize/__init__.py`, `core.py` (`Source`, `Scope`, `Builder`,
  `name_norm`, `quote_field`, `path_from`, `Resolver`), `python.py` (`py_resolve`, `py_imports`, `py_spans`,
  `build_scopes` for Python), `cpg/tests/test_normalize_core.py`, `cpg/tests/test_normalize_python.py`.
- **5b files:** `.../normalize/javascript.py` (`js_resolve`, `js_imports`, `js_type_spans`, astgen walk),
  `.../normalize/blame.py` (`blame`, `last_changed`, the `git archive`-staged-tree rule of BF12: computed
  **outside** the sandbox from the source repo), `.../normalize/assemble_bundle.py` (the old `assemble()` that
  turns raw facts into bundle records and calls `ccg.atrest.write_bundle`), `cpg/tests/test_normalize_javascript.py`,
  `cpg/tests/slice/test_normalize_equivalence.py`.
- **Behaviours:** (1) per-frontend goldens (small Python and TS files with expected records, one per rule of
  `-bundle-spec3.md` §4–§5: qname grammar, `<anon>#k`, decorators, `fieldAccess`, `Ret` synthesis for
  expression-bodied lambdas, `use == def` kept); (2) the slice's stored bundles are reproduced byte for byte for
  `falkor-chat/server` and `salesperson` under the **`slice` profile** (P1-06's masking, P1-08's override and P1-09's
  projection off; header `builtAt`/`sandboxAuditIds` exempt, §8.5), with the artifact sha256 and `0 skipped` recorded
  (index D9); later steps pin their own named deltas against this baseline; (3) `lineCount`-0 file
  rule and the span-plus-`ownedRegions` fallback of spec4 §2 (the slice code tested the span only; this step
  tests both); (4) the normalizer runs under CPython 3.12 (spec4 §3 host-Python note), asserted at start; (5) a
  shallow clone is a build error (index D1); (5b) **G-4:** a `controlledBy` record with `point == cond` (a CDG
  self-edge) is dropped in `assemble_bundle.py` while a `data` self-edge (`use == def`) is kept, each with a test; (6) every emitted bundle passes `ccg.bundle.validate`; (7) the bundle is written only through
  `ccg.atrest.write_bundle`: the file is `0600` and its directory `0700` under `bundles/<system>/`, under umask 0 as well.
- **Done:** `build_bundle(unit, raw_facts, repo, commit) -> path`. **Gate:** `analyst`. Python has no
  primary-constructor owned region, so 5b's `ownedRegions` test uses a synthetic region lying outside the
  symbol's span; the first real case is the P1-41 fixture (0 of the slice's 4,943 symbols carry one).

## P1-06 · `Cond.code` literal masking (F-9) — `graph-dba`

- **Files:** `cpg/extractors/joern/normalize/mask.py` (`mask_cond_code(code: str) -> str` replaces each string
  literal that fails `ccg.bundle.screen_fails` or `name_fails` with `"…"`), wiring in `assemble_bundle.py`,
  `cpg/tests/test_mask_cond.py`.
- **Behaviours:** a literal passing the screen is kept verbatim; a failing one (32-hex, token prefix,
  UUID-shaped) becomes `"…"`; escapes and nested quotes in Python and TS template literals are handled; a
  condition with no literal is byte-identical; `Cond.code` after masking never contains a string that
  `screen_fails` flags (a property test over generated literals).
- **Done:** hard prerequisite of P1-41 and of any non-proving build. **Gate:** `analyst`.

## P1-07 · writer done-condition (F-10) — `tdd-engineer`

`-bundle-spec3.md` §1.6.1, §1.7.4 (W2–W4), §1.7.5.
- **Files:** `cpg/ccg/bundle.py` (defence in depth: `write_bundle` creates its temp file with `O_EXCL`, mode `0600` and
  `fchmod`, so a stray caller cannot widen a bundle; a test under umask 0), `cpg/extractors/common/writer_rules.py`
  (shared by P1-10, P1-11, P1-12), `cpg/ccg/valuehash.py`
  (`entry_hmac(key: bytes, domain: str, value: str) -> str`, per-entry domain separation of §1.7.5),
  `cpg/tests/test_writer_rules.py`, `cpg/tests/test_valuehash.py`. The launch and config writers themselves are
  P1-11's; this step supplies the rules they call.
- **Behaviours (one test group per obligation, each with the spec's negative case):** (W2) a `bindValue` or
  `launchFact.default` is emitted only for a declared service/binding role, with `default` only when it passes
  the grammar; (W3) alias and interpolation (`$`, `${…}`) values are never bound; (W4) the third writer rule of
  §1.7.4 as written there; (SA-1 writer side) `launchFact` `command` and `args` are produced only through
  `ccg.bundle.gate_argv` — a test fails if any writer module builds an argv field without it; (§1.7.5)
  `valueHash = entry_hmac(key, r["id"], text)`, where `r["id"]` is the `configEntry` record's own 32-hex id
  (`ids.config_entry_id(repoId, path, keyPath)`) and `text` is the value as the format's unescaping yields it;
  `launchFact.defaultHash = entry_hmac(key, <the launchFact's id>, <default text>)`. The same value under two
  `keyPath`s gives two hashes. `entry_hmac` reproduces `-bundle-spec5.md` §9.1, EH-31 for the id-domain form, with
  expected hex read from the spec text. The key is read only through `ccg.atrest.hmac_key` (P1-02), which refuses
  both of spec5 §9's test keys; tests pass a test key to `entry_hmac` directly.
- **Needs:** P1-13 (`-bundle-spec5.md` fixes `entry_hmac`'s bytes and EH-31).
- **Done:** a `writer_rules` import is the only way an extractor decides these rules. **Gate:** `analyst`;
  `security-expert` advisory (HMAC, SA-1).

## P1-08 · `jssrc2cpg` override, `skippedFiles`, coverage probe — `graph-dba`

- **Files:** `cpg/extractors/joern/skipped.py` (`skipped_files(unit, parsed, override) -> list[{path, reason}]`),
  `cpg/extractors/joern/run.py` (override flag, if any), `cpg/tests/test_skipped_files.py`
  (the spec4 §1 coverage probe: a table test over language × reason × overlap × location × override), a note in
  `cpg/extractors/joern/README.md` recording the mechanism.
- **First behaviour (a probe, reported):** does `jssrc2cpg` 4.0.579 accept any switch that re-includes
  `i18n`/`test` paths? Run `jssrc2cpg --help`, its config object and one experiment on a 3-file copy that keeps
  paths (a staged-copy rename is not an escape: it changes `qname`s). Record the result in the README.
- **Behaviours:** `test` pattern (`*.test.<ext>`, `*.spec.<ext>`, a segment `e2e` or `__tests__` below
  `rootPath`) applies whether or not the frontend parsed the file; precedence `test` over `frontend-ignore`;
  `.d.ts`, `.pyi` and vendored paths are listed in neither; every table cell asserts exactly one reason or none and
  no `file` record for it. If the override exists, the five `salesperson/src/i18n/` files get `file` records and
  the ten empty-`targets` call sites resolve; if not, they are `frontend-ignore` (R-3).
- **Done:** `header.skippedFiles` is written exactly when non-empty. **Gate:** `analyst`.

## P1-09 · projection fix and golden sets — `graph-dba`

Index §5 decides what and why. Scope: python and TS/JS only (the C# guard of `-p0-2.md` §15 is reported not closed:
not a P1 language).
- **Files:** `cpg/extractors/joern/extract.sc` (the projection), `cpg/extractors/joern/normalize/points.py`
  (join fixes: a Python `for` CDG node coded `iteratorNonEmptyOrException` maps to the `for` statement's condition
  point (36 misses); the contains-join that resolves the label `not (self._open or ...)` to the operand
  `self._open` is fixed so the whole condition point is found (4 misses)), `cpg/tests/golden/projection/*.json`,
  `cpg/tests/test_projection_golden.py`, `cpg/tests/slice/test_projection_not_regressed.py`.
- **Behaviours (ordered):** (1) R1: `defsOf` also follows a `REACHING_DEF` whose source is an earlier same-name
  identifier that is not an assignment target, through its own reaching definitions (67 python misses); (2) R2:
  the receiver of a `<init>` call is defined by that call (0 python, JS and Java gains); (3) **R3 as a projection
  rule**: a frontend-synthetic call (one whose name is not in the source, such as iterator `next`) is a
  transparent intermediate when the projection walks `defsOf`, so the pair connects without scoring-time help (17
  python misses); this is a design line of its own because the report's R3 was a scoring relaxation designed after
  reading the misses, so the golden states the rule and a negative case; (4) the 27 unverified projection-side causes
  (comprehension/generator scopes 12, a call that is itself a CDG source 10, dict/array/spread literals 4, a
  bound-method alias 1) each get a golden and a fix or a recorded "not fixed, because"; (4b) **synthetic-cond transparency** as `-p1-ml.md` §3.4 states it (recognizers, nearest-real-cond redirect, ungoverned
  edge dropped; scoring stays direct-edge): goldens for the 8 `definition` pairs (no projected edge in those methods ends
  at a synthetic cond), the joint golden on `services.py::_normalize_opaque`, nested and ungoverned cases, and the
  negatives (the `test_repository.py:4703→4706` pair stays unconnected; a point under a real intermediate cond keeps
  its direct edge); **mutant:** with transparency off all 8 positives fail; (5) **no connected pair is lost**:
  on the slice's stored raw facts the shipped projection's `data` records are a subset of the fixed projection's and
  its `controlledBy` records are a subset **modulo the rewritten synthetic-cond edges**, each listed with its
  replacement set (one edge, several, or none), and no golden pair the shipped projection connects is unconnected by
  the fixed one (`slice/test_projection_not_regressed.py` carries the same carve-out); (6) the spurious rate does not rise on the 30 dev-contact methods (`cpg/spikes/frontend-
  fitness/truth`, `s28_metrics.py` reused, not copied); (7) `header.flowProjected` stays true and `fieldDepthCap` 3.
- **Done:** a report line with the dev recall reached (with its cluster interval) and the list of fixable python
  misses still open. P1-34's entry value is 465/481 against a dev ceiling of 467 (`-p1-ml.md` §3.1), so at most 2 of
  the 159 fixable misses may stay open, and **the expected outcome is a flag that stays false** (P1-34 "not run"); this step still ships because R1 loses no pair and every fix
  helps the call-level-versus-flow question. **Gate:** `analyst`; `data-scientist` reads the report line.
  **Risk:** Java-heap blowups in the "follow every source" variant (§a2.2.1) — do not adopt it.

## P1-10 · shell extractor — `coder`

- **Files:** `cpg/extractors/shell/extract.py`, `.../__init__.py`, `cpg/tests/test_shell_extractor.py`,
  `cpg/tests/data/shell/` (small `.sh` fixtures), `cpg/extractors/shell/README.md`.
- **Interface:** `extract_shell(unit_root, files) -> list[records]` using `tree-sitter-bash` (extra `extract`),
  one `<script>` file-scope symbol per script (`-bundle-spec3.md` §4.1 rule 4), `function` symbols, `Param`s
  for `$1…`, `"$@"`, call sites for commands, `DATA` from assignment to use inside functions, `shell-dynamic`
  `possible` for `eval`, indirect expansion and computed `source`; shell value sets feed role lifting (the
  `gquery` function with 50 call sites in `bootstrap_schema.sh` lifts Cypher literals, AC-3 is P3 but the lifting
  must work now).
- **Behaviours:** goldens for each construct above; `$1` positional flows to a command argument; a function's
  parameter lifts a literal at its call site; `source "$x"` is `possible`/`shell-dynamic`; quoting edge cases
  (`"$@"`, `${var:-default}` default is **not** a `bindValue`: W3); the extractor passes `ccg.bundle.validate`;
  the real `falkor-chat/scripts/*.sh` and `cypher-mcp/*.sh` parse with zero error nodes; an unparseable shell file
  is reported as a unit warning and the script still gets a `<script>` symbol with no calls (never silently
  absent; the `skippedFiles` enum is `test|frontend-ignore` and does not apply to shell).
- **Done / Gate:** goldens plus the corpus smoke; `analyst`.

## P1-11 · config, Compose, `.mcp.json`, Kiro and Dockerfile launch extractors (F-6) — `coder`

- **Files:** `cpg/extractors/config/extract.py` (dispatch by file role/format), `.../formats.py` (per-format
  `keyPath` derivation table; one row per `format`, as `-bundle-spec3.md` §5.17 requires), `.../launch.py`
  (`launchFact`s from Dockerfile `CMD`/`ENTRYPOINT`/`EXPOSE`, `uvicorn --port`, env defaults `${VAR:-8200}`,
  Compose `ports`/`build.context`), `cpg/tests/test_config_extractor.py`, `cpg/tests/data/config/`.
- **Behaviours (F-6, SA-3 allow-by-role):** `bindValue` only for a binding role: Compose
  `services.*.ports[i]`/`expose[i]`, `.mcp.json` `mcpServers.*.url`, the Kiro agent's MCP URL, or a `keyPath`
  whose last segment contains a role token (`port`, `host`, `hostname`, `addr`, `address`, `url`, `uri`, `endpoint`,
  `base`, `server`, `listen`, `bind`) after splitting on `_ - .` and camelCase; **array elements under
  `args`/`command`/`entrypoint` are never `bindValue`**, they are `launchFact`s through `gate_argv` (P1-07);
  Compose list-form `environment` entries split on `=` and the name is read from inside the value; example
  configs get `File.role = 'example'`; `.mcp.json` yields the launch chain input (server → command → script);
  every format has a derivation row and a test; the `Cond`-free path is `ConfigEntry` only; secrets never
  stored (`valueHash` HMAC through P1-07). Compose `build.context` resolves relative to the compose file and may
  leave its repo (`-p0-compose-binding.md`): the extractor records the relative path, resolution is the linker's.
- **Done / Gate:** one derivation table row per format of `-bundle-spec3.md` §5.17; `analyst`; `security-expert`
  advisory.

## P1-12 · prompt/Markdown extractor — `coder`

- **Files:** `cpg/extractors/markdown/extract.py`, `cpg/extractors/markdown/globs.py` (loads the class glob
  list from `cpg/eval/eval-manifest.yaml`, never duplicates it), `cpg/tests/test_markdown_extractor.py`,
  `cpg/tests/data/markdown/`.
- **Behaviours:** Markdown is classified `prompt-md` or `doc` by the manifest's globs, so census scope equals
  evaluated scope (EP Q3); a `doc` produces a `<document>` symbol and no mentions (documentation snippets are not
  callers); a `prompt-md` file's mentions of a tool, script or command become call sites of kind `mention`,
  `possible`/`prompt-mention`; fenced code in a prompt is not a call; the loader fails if the manifest is
  absent or its glob list changed shape (a schema check).
- **Done / Gate:** goldens plus a corpus smoke on `claude/**` prompts; `analyst`.

## P1-13 · `-bundle-spec5.md` (F-7, `bindParts`) — `architect`

An additive successor (`Extends:` `-bundle-spec3.md` and `-bundle-spec4.md`; both gain `Extended by:`):
the optional `configEntry.bindParts {scheme, port, hostKind ∈ loopback|ip|service|external, host?, hostHmac,
pathHmacs[]}` per the SA-4(b) design in `docs/reviews/connected-code-graph-bindvalue.md`; HMAC through
`entry_hmac` with its own domain tag; `host` only for `loopback|ip|service`; a minor schema version bump rule
(§8.6); the writer rule that a value with `bindParts` also keeps `bindValue` only if it passes §1.7, and that
`bindValue` is never rendered (the review's P2-6); the linker's equality comparison on `hostHmac`/`pathHmacs`;
the conformance vectors. States what changes in `bundle.py`, the schema, `ids.py` (nothing: not in any
identity tuple) and the graph note. **Gate:** `analyst`; `security-expert`.

## P1-14 · encode `bindParts` — `tdd-engineer`

- **Files:** `cpg/ccg/schemas/bundle-v1.schema.json` (optional field, no required change), `cpg/ccg/bundle.py`
  (validator checks per spec5), `cpg/ccg/valuehash.py` (the three tags, `bind_parts`,
  `bind_parts_usable`, `mount_prefix_hmac`; spec5 §8; the key and `bind_key_check` have one home, `ccg.atrest.hmac_key`
  and `ccg.atrest.bind_key_check`, P1-02), `cpg/extractors/common/writer_rules.py` (`bind_parts_allowed`),
  `cpg/extractors/config/formats.py` and `launch.py` (emit), `cpg/tests/test_bundle.py`
  (rows parsed from spec5's text as P0 did: "tests read the contract's text"), `cpg/tests/test_bindparts.py` (including the one-definition test: an `ast` scan of `cpg/ccg/` and
  `cpg/extractors/` finds `def hmac_key` and `def bind_key_check` only in `ccg/atrest.py`, and `valuehash.py`
  defines no key-reading function), `cpg/tests/test_answer.py`, and the P1-11 tests whose expectations change: W6 withholds `bindValue` for `ip` and
  `external` URL hosts, and W7 plus spec5 §1 give every P1-11 URL entry `bindParts`, so those bundles become `"1.1"`
  with `bindKeyCheck` (spec5 §6.1, §6.2).
- **Key epoch (spec5 §5.4):** this step supplies `bind_parts_usable` and its test (a bundle planted under spec5's
  second test key is loaded but its `bindParts` are not usable; the same bundle under the first key is the positive
  control); BR-1–BR-3's `usable` column. P1-17 (`bind.py`) owns the binding outcomes and the `bind-key-mismatch`
  line; P1-17 already needs 14.
- **Behaviours:** a bundle without the field validates unchanged (positive control: the stored slice bundles);
  every vector of spec5 passes; a capability URL (`https://ntfy.sh/x7k2-q9w4-m1p8-z3v6`) yields `bindParts` with
  no `bindValue`; a multi-segment base (`http://localhost:8000/api/v1`) binds by `pathHmacs`; failing vectors
  first (the pin and the field are not authored in one step). **Gate:** `analyst`.

## P1-15 · indexer adoption (`scip-typescript`, `scip-python`) — `coder` + `graph-dba`

Blocked on the `-p1-ml.md` note's question 1 (method, offline protocol, adopt criterion). The step carries
its output to the bundle as a **resolution source**, never as a new schema element.
- **Files:** `cpg/extractors/indexers/scip.py` (parse a SCIP index to `(file, line, col) → target qname`; the `.scip` file is hostile input, since the process that wrote it may have executed staged code: a size cap, and no path read from it is used for a write),
  `cpg/extractors/indexers/run.py` (runs the indexer through `offline-run.sh --mode offline --frontend
  scip-typescript|scip-python`, setting `CCG_TOOLCHAIN_IMAGE` to the P1-48 `indexer` image for that call; `toolArgv` is
  the §7.4 grammar: `index --cwd /src` for `scip-typescript`, and `index --cwd /src --environment /src/<env.json>
  --project-name <n>` for `scip-python`; the `node_modules`, venv and `env.json` the **operator** provisioned outside the system (no `cpg/` module runs `npm ci`, `pip` or `uv`; `-egress-spec3.md` §7.4) are placed in this
  run's **own staged directory**, never the one a Joern parse reads, and arrive read-only through `/src`; the
  stager rejects symlinks that point outside the directory, hardlinks and device or FIFO files, and provisioning
  installs dependencies only, per `-egress-spec3.md` §7.4; `run.py` runs one indexer at a time under a wall-clock
  timeout that sends TERM to the runner (R-E10); the runner and its `indexer` entries are P1-48b's, not this step's), `cpg/extractors/joern/normalize/merge_scip.py` (merge
  rule from the note: a SCIP-resolved target upgrades a name-match site to `confirmed`/`resolved-call`; a
  disagreement with Joern keeps both and marks the site `possible`/`resolution-unverified`; candidates are
  never dropped), `cpg/tests/test_scip_merge.py`, `cpg/tests/test_scip_parse.py`.
- **Behaviours:** parser golden on a stored tiny SCIP index; merge cases (agree, upgrade, disagree, SCIP-only);
  the run's argv is the §7.4 grammar and the runner's own refusals enforce it (a test replays a non-grammar argv through
  the runner and sees the refusal); no run goes through `--frontend joern`; a build runs an indexer only if the
  operator already provisioned the dependencies, and otherwise runs Joern alone and records `reason: headers-unavailable`-class degradation
  under the existing registry code (no new code) — decided in the note.
- **Done:** per language, on the P0 call-site truth (`cpg/spikes/frontend-fitness/truth`, `s27_metrics.py` reused), the
  measures of `-p1-ml.md` §2.1: M1 (offline/online agreement k/n with Wilson, every differing site adjudicated), M2
  (merged recall and precision with n_P, Joern-alone figures, confirmed share, fan-out p90) and M0 (bare, informational),
  with wall time and peak memory; the merge rule of §2.2 and the adopt criteria of §2.3; the `ccg-p1-indexer-v1`
  record (§8) committed before the offline run. **The step ends in one of four outcomes:** adopted (the criterion
  holds on the offline figure); "not adopted, because" (it does not); "not run: runner" (P1-48b's indexer entries are
  absent, or the first run cannot place the staged dependencies, `-egress-spec3.md` R-E9); or "not run: dependencies
  not provisioned" (the operator has not provisioned them; `teco` is told and asks the operator, and no `cpg/` module
  provisions). In the last three the Joern call graph with `possible` name-match items is the P1 baseline (R-2).
  **Gate:** `analyst`; `data-scientist`.
