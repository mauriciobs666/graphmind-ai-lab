# Review: C-1001 P1-44 — build dispatch by unit kind (implementation)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Reviewed the uncommitted P1-44 delivery (`cpg/ccg/build.py`, `cpg/tests/test_build_dispatch.py`, `cpg/tests/test_build_live.py`,
`RunnerTimeout` in `cpg/extractors/joern/run.py` + `cpg/tests/test_joern_run.py`, one `cpg/README.md` row) against
`docs/plans/connected-code-graph-p1-linking.md` § P1-44 (Behaviours (1)-(8)), `-p1-graph.md` §1 G4, `cpg/sandbox/README.md`,
bundle spec3 §2.2, egress spec3 §5.4, root `REVIEW.md`. The live test was **not run** (Docker/Joern). `cpg/ccg/assemble/` was not read
(another unit's); `ccg/sandbox_audit.py` was read because it is the P1-18 check P1-44's headers must satisfy.

**Verdict: needs changes** (one blocker).

`CPG: considered, not relevant — the CPG holds the repo's old code; every claim here was settled by running the new, uncommitted files and probes against them.`

Evidence run: `cpg/.venv/bin/python -m pytest tests/test_build_dispatch.py tests/test_joern_run.py` (cwd `cpg/`): 120 passed, 3 live deselected;
neighbours `test_units test_extractor_errors test_shell_extractor test_config_extractor test_markdown_extractor test_atrest test_bundle
test_normalize_assemble test_skipped_files`: 2212 passed. The rest of the suite was not completed inside the 160 s cap (killed, not green).
Mutations ran on a scratch copy of `cpg/` (`cpg0/`), 31 mutants of `build.py` against `tests/test_build_dispatch.py`.

## Findings

### B1 (blocker) — shell stage writes outside its temp dir for a repository-controlled path (verified, exploited in scratch)

`build.py:_shell_records` does `dest = Path(td, path); dest.parent.mkdir(parents=True); dest.write_bytes(<blob>)` for every path of the
commit's tree, **before** `extract_shell`'s `_check_rel` ever sees it. Git trees may hold a `..` entry (made with
`git hash-object -t tree --literally`; `git ls-tree -r` prints `a/../../evil.sh`). Probe: a commit with that tree, `build_unit(shell unit "a")`
with `TMPDIR=<scratch>/tmpbase/x/y` wrote `<scratch>/tmpbase/x/y/evil.sh` (one level above `ccg-shell-*`), contents attacker-chosen
(`echo PWNED`), and the file **survives** the unit's `extract:failed`. More `..` entries reach any absolute path the build user can write
(`~/.bashrc`, `~/.config/...`). The source repository is untrusted (spec3 §8.5; the extractors' own D12 stance). The config and
markdown extractors reject such names (`_clean_path`, `_check_path`); the shell path is the only one where the build writes first.
No test plants a hostile name for the stage.

Fix: validate every path of the tree once in `_tree` (reject, as `SourceRefused`-equivalent value-free failure of the unit, any path with a
`..`/`.`/empty segment, a leading `/`, a backslash or NUL — reuse `extractors.shell.extract._check_rel` rather than a second rule), and in
`_shell_records` assert `dest.resolve().is_relative_to(Path(td).resolve())` before writing. Test: the probe's tree through `build_unit`;
it must fail alone, value-free, with nothing written outside `td` (positive control: the same test red against the current code).

### M1 (major) — "reads the built commit, never the working tree" is pinned for shell only

Mutants that read `(repo / path).read_bytes()` in `_markdown_records` and in `_config_records` both **survived** all 50 tests; only the shell
mutant was killed (`test_a_shell_unit_runs_the_shell_extractor_over_the_files_of_the_commit` edits the work tree after the commit). Add the same
"edit the working tree after committing, assert the bundle has the committed bytes" row for a config and a markdown unit (`contentHash` of the
`file` record is a ready assertion).

### M2 (major) — a symlink / non-regular blob is excluded by code but the exclusion is untested

`_REGULAR = ("100644", "100755")` keeps symlinks (`120000`) out. Mutant adding `"120000"` **survived**. Probe (real symlinks `link.json`, `pw.json -> /etc/passwd`,
`s.sh -> ../../x` in the tree): all three were excluded from the file lists, and a unit whose root is a symlink built an empty (header-only)
bundle. The behaviour is right; nothing would notice if it regressed, and the regression is a symlink's target string indexed as file content.
Add the symlink and gitlink rows. (Submodules are skipped by the `typ == "blob"` filter; LFS pointers are small regular blobs and are
extracted as text — fine, not a finding.)

### M3 (major) — one non-UTF-8 Compose file fails every config unit of its repository (verified)

`_declared_services` runs `extract_config` over **all** Compose files of the repository, whose `_decode` raises `SourceRefused` for a non-UTF-8 file. The
raise propagates out of every config unit's build. Probe: repo with `compose.yaml` holding `\xff\xfe`, plus `app.json` and `.mcp.json`; all
three units returned `extract:failed` — including `.mcp.json`, which the linker needs. Plan (3): "a failure fails **that unit's** bundle and nothing else".
The implementer's choice is defensible as fail-closed, but its blast radius contradicts the rule, and discovery makes config units per file
by default. Suggest: compute the declared set with the Compose files that decode, and let the undecodable one fail only its own unit (its own
`extract_config` call still raises); add the probe as a test.

### M4 (major) — `extract:timeout` is reached only by the Python-side cap, not by the sandbox's own 124

`run.py:352 need_ok` turns any non-zero stage exit into the generic `RunnerError`, including the runner's watchdog exit **124** (`cpg/sandbox/README.md`:
the cap is `SBX_INDEXER_WALL_S` = 1800 s, and `run.py:STAGE_TIMEOUT_S` is also 1800). Which fires first is a race: the same hang gives `extract:timeout` when `_run`
wins and `extract:failed` when the runner's 124 wins. Plan text says `RunnerTimeout` is raised "where `_run` times out", so this is conformant,
but the code's intent (a timed-out unit says so) is met only by chance. The README line P1-15 must set its timeout above the cap + 60 s is also not followed
(`STAGE_TIMEOUT_S == cap`). Suggest `need_ok` raise `RunnerTimeout` for rc 124 (the README says 124 is "watchdog **or** the tool's own 124", so the
`durationMs` caveat applies; mapping it to the timeout code is still the better bucket for a cap-length run) and a test with the stub runner exiting 124.
If the owner prefers to keep the plan's narrow wording, record the race in the plan.

### M5 (major) — oversize Markdown is blamed before it is skipped, so a slow or failing blame fails the unit

`_markdown_records` evaluates `history=hist.whole_file(path, data)` as an **argument** of `extract_markdown`, i.e. before the size check that raises
`MarkdownError`. Probe: `blame.blame` was called for `c/big.md` (2 MiB + 1 B); making it raise `HistoryError` for that file turned a unit that
should skip the file and count `skippedDocs` into `extract:failed`. A hostile large file with a pathological history (`git blame` over up to the 300 s `GIT_TIMEOUT`) does the same
wastefully. Check `len(data) > MAX_FILE_BYTES` first (or pass a lazy history), then blame. Test: oversize file + blame that raises.

### M6 (major) — no size or count bound on what is read from the tree (inferred from code; blame/`cat-file` probes only)

`_blob` is `git cat-file blob` into memory with no `cat-file -s` pre-check; `_config_records` holds every file of the unit in one dict, and
`extract_config` decodes each (`_decode`) **before** its `MAX_BYTES` check; `_shell_records` writes each blob to `$TMPDIR` before `_read`'s cap. A repository
with a multi-GB blob can exhaust the 6 GB host's memory or its temp disk on a unit that the extractors would merely have skipped or refused. Read
sizes first (`git cat-file --batch-check` over the unit's oids) and treat a blob over the extractor's own cap as that extractor does (config:
`skippedFiles`/File role; markdown: `skippedDocs`; shell: refuse). Not demonstrated at multi-GB scale (the probe host is 3 GB `ulimit -v`); say so if the owner deprioritises.

### m1 (minor) — header values for host bundles

Checked against the schema (`bundle-v1.schema.json` `tools`: all seven strings; no pattern) and spec3 §2.2.

- `tools.joern: "none"`, `flowProjected: False` — valid and correct ("whether the per-method projection ran").
- `frontendVersion: "0"` and `normalizerVersion: "0.1"` are constants, not versions (§2.2: "put their name and version in `frontend`/`frontendVersion`"). Use
  `importlib.metadata.version("tree-sitter-bash")` for shell and a module `VERSION` for each extractor.
- `extractScriptHash` hashes **one file**: config `extract.py` only (not `formats.py`/`launch.py`), markdown `extract.py` only (not `globs.py`),
  shell `extract.py` but not `common/writer_rules.py`. A change in an unhashed module does not change the header, so the provenance can claim
  "same extractor" for different logic. Hash the sorted set of the modules the extractor imports from `extractors/`.
- `vendoredDirs` is `assemble_bundle.DEFAULT_VENDORED`, but `_unit_files` skips `units.SKIP_DIRS`, which also holds `.git .venv venv .tox __pycache__
  .pytest_cache .mypy_cache .ruff_cache`: an excluded file is not "explainable" from the header (§2.2). Same inconsistency in the Joern path; a manifest per-repo replacement is not honoured here either. Note, do not block.
- `builtTree` for host bundles is never asserted: mutant `tree_id = sha` **survived**. Add `header.builtTree == git rev-parse <sha>:<root>` (and `^{tree}` for `.`).

### m2 (minor) — `_language_of` and `units` private helpers

`build.py` imports `units_mod._CONFIG_EXT`, `_shebang_is_shell`, `_skipped_dir` and re-implements `units._file_language` over blob bytes. Two copies of the
classification drift silently; no test compares them. Make `units._file_language` take a "first line" callable (or expose a public classifier) and add
a test that `_language_of` and `discover_units` agree on one tree. For extensionless files it is also one `git cat-file` per file per unit (measured: 1,500
extensionless files under 20 shell units = 1,540 subprocesses, 1.7 s; linear, not alarming — batch via `--batch` if the monorepo is large).

### m3 (minor) — stale `_declared` cache across commits

`ctx._trees` is keyed `(repo, sha)`; `ctx._declared` is keyed `(repo, repoId)` only. Probe: one `BuildContext`, `build_unit` at commit 1 then commit 2 where Compose
dropped the service: the second bundle kept `bindValue == "api"` from the first. A refresh builds one commit per repo, so latent; add `sha` to the key or document single-commit use.

### m4 (minor) — an uncheckable commit argument

`build_unit(..., commit)` is not validated: `ls-tree`/`cat-file` receive it raw (`--help` fails as `extract:failed`; `HEAD`, `main` and a 12-char prefix are accepted by git and then fail
as `bundle:invalid` after the work is done, a misleading code for a caller bug; a branch name could move between `ls-tree` and `cat-file`/blame calls). Call
`blame._check(commit)` (or `is_commit_id`) first and raise a typed error as for an unknown kind. Not exploitable: `blame.git` passes argv lists with `--literal-pathspecs`, and `ls-tree`
output and `cat-file` object ids are never path-interpreted. Verified probes with `-rf.sh`, `--help.md` and a newline-named JSON: the first two built; the newline name made the unit
`bundle:invalid` (the bundle validator refuses it), value-free.

### m5 (minor) — a few unpinned claims (mutants that survived)

`_counts` filtering (`counts-leak` survived: nothing feeds the function a non-int). It is the line that keeps a report value-free, so give it one test with a planted
string/dict in the report of a code unit. Also `test_build_live.py` never runs `ccg.sandbox_audit.check_sandbox_ids` over its own bundle and temp `CCG_AUDIT_DIR`, so the one test that has a real audit id does not prove the header's ids
exist; a failing-runner row (real `RunnerError`/timeout through the real runner) is absent too.

### n1 (nit)

`require_full_history` also raises `HistoryError` for a git timeout; the plan maps that to `build:shallow-clone` (done), so a timed-out `rev-parse` is labelled shallow. Plan-conformant; mention it in the module doc.
`build_units` keeps `read_bundle`/`loads`/`known_tools_of`/`repos[...]` outside any `try` (an error there aborts the whole call); plan P1-20 owns the loop, so call this out in P1-20 rather than here.

## Answers to the brief's questions

- **Mutation coverage (31 mutants):** killed — code-first order, once-per-repo `extract_config` (no-cache, own-unit-only), `extra_declared_services` dropped, `known_tools` per-unit and never,
  header always `1.0` / from the first file only, every mapping row (timeout, bundle-invalid, shallow, history check removed), markdown skip/not-counted/history/known_tools dropped, config File record and role,
  dot directories, reason text, unknown-kind skipped, exception narrowed, bundle written outside `atrest`, shell work-tree read, frontend/lang fields. **Survived: M1 (md + config work-tree reads), M2 (symlink included), m1 (`builtTree`), m5 (`_counts`).**
- **Value-free mapping:** verified with probes: planted secret in an exception message, in non-UTF-8 config and shell file **names and bytes**, captured stdout/stderr/`logging` and the result objects: the secret appeared nowhere except
  `BundleResult.unit.rootPath` (the manifest's own value, not extractor output); no `ccg-shell-*` temp dir left after failures. `fail_code` maps by class, so a subclass of `ExtractError` and any unlisted exception land on `extract:failed`.
- **Raising `UnknownUnitKind` / `MissingAuditIds`:** `UnknownUnitKind` is **not** a plan violation: plan (2) demands a typed error, never a silent skip, and (3)'s "any other exception" governs failures **after** dispatch, inside the try. It does leave P1-20 with
  no code of `FAIL_REASONS` for it, so P1-20's own `try` must decide (propose: it is a caller bug and aborts the refresh, or maps to `extract:failed` — the plan should say which). `MissingAuditIds` **is** a deviation: the plan has no such raise, it fires for every
  host unit of a context lacking ids, and under P1-20's per-unit `try` it becomes a failure with no assigned reason. It exists only because of the audit-id gap below and goes when that is resolved.
- **`sandboxAuditIds` gap — independently confirmed:** `bundle-v1.schema.json` (`sandboxAuditIds`, `minItems: 1`, UUIDv4 pattern); spec3 §2.2 ("one per `offline-run.sh` invocation that produced this bundle's inputs ... each must have a `sandbox` record"); egress spec3 §5.4 and P1-18 (7)
  (`check_sandbox_ids`, presence-only). The shell/config/markdown extractors never call the runner (no `offline-run.sh`/`sandbox` reference in `extractors/{shell,config,markdown}`). Probe: a fabricated UUID against an empty `sandbox.jsonl` is refused
  (`0 records ... exactly one required`). So the only way `build_unit` can emit a host bundle today is with ids that name no record, and every test fixture uses such a made-up `AUDIT`: the dispatch tests prove the bundle validates, not that P1-18 would accept it.
  I note the concurrently written ruling (empty list for host languages) and `check_sandbox_ids`'s `HOST_LANGS` branch only as context; the follow-up for P1-44 is to emit `[]`, delete `MissingAuditIds`/`BuildContext.audit_ids` for host kinds, and add
  a test that `check_sandbox_ids` returns `[]` over each host bundle's header and non-empty over a **code** bundle with `[]` (positive control).
- **Config units that extract nothing:** a File-only bundle is correct and tested (`examples/broken.json` -> File, role `example`); one consequence worth a test: a unit of an *ignored* config name (`F.classify` is None) also gets a File record claiming `kind: "config"`.
- **`bundle_path` always `None`:** correct for P1-44. Bundle paths are repo-relative (spec3 §2.2/§3.2, `assemble_bundle`), the same string the globs match; plan (6)(b) is conditional and its condition never holds. Add a one-line comment saying so.
- **Default `globs`:** `load_globs()` reads `cpg/eval/eval-manifest.yaml`, the **eval** manifest, as the production prompt-bearing glob list. Not a defect here (the extractor's own default), but a refresh silently depends on an eval file; raise it with the architect (open question).

## What's solid

- Failure mapping: one `fail_code`, class-based, `RunnerTimeout(RunnerError)` addition is minimal and `except RunnerError` callers still work (test asserts both).
- Order and once-per-build discipline (`known_tools`, code first, per-repo Compose set) each have a test that went red under mutation.
- Header (8): the version follows the merged record set; the key is read through `atrest` only (a spy test and the `write_bytes` count guard).
- Blobs are read by object id from one listing (no list-then-read race for a full-sha commit); option-like and newline names never reach git as options.
- Value-free results (probed), temp stage removed on every path, `MarkdownError` skip counted.

## Open questions

1. `UnknownUnitKind` in a refresh: abort, or map? (P1-20 owner / architect.)
2. Should the build ever honour a manifest's per-repo `vendoredDirs`? (Joern path also does not.)
3. Is the eval manifest an acceptable production source of prompt-bearing globs?

### Proposed REVIEW.md lines

Target `REVIEW.md` § *Always check* (source: finding B1; recurring class "write-before-validate"): "A step that materialises untrusted names (tree paths, archive members) onto a filesystem is probed with a name holding `..` and an absolute prefix; the write lands inside the intended directory or fails before writing."

## Pass 2

Re-gate of the fix wave (`build.py` 475 lines, `ccg/units.py` public `classify_file`/`is_skipped_dir`, `run.py` `RunnerTimeout`/`INDEXER_WALL_CAP_S`/`Completed.elapsed`/`wall_cap`, `build.py`'s use of `ccg.gitsafe` via `blame.git`; `gitsafe.py` itself not reviewed). Run: `tests/test_build_dispatch.py test_joern_run.py test_units.py test_gitsafe.py` = 347 passed (3 live deselected, live test not run). Scratch copy `cpg1/`, 34 argument-level mutants (output files under the session scratchpad).

**Verdict: needs changes** — one new major (N1); everything else is fixed or minor.

### Dispositions

- **B1 fixed.** My own hostile trees (crafted with `hash-object --literally`), each at unit root `.` for shell/config/markdown: `x/../../evil`, absolute `/abs_evil`, `.git/x`, `.GIT/x`, `a/.Git/x`, backslash, `\x01`, newline, 300-byte segment, 30x200-byte path, `a/./b`, `a//b`: all 36 failed `extract:failed`, nothing written anywhere in the scratch tree (positive control: the same tree wrote one level up in Pass 1). Case-colliding names and a leading `-` build (tests). Ordering/symlink: a symlink entry `x` -> outside placed before `x/evil.sh`, name check monkeypatched off: nothing outside written. Reason: symlinks are filtered by mode before staging, so no link ever exists in the temp dir, and `_assert_inside` (mutant `if False` killed) is a second guard. Both guards run before `mkdir`/write.
- **M1 fixed.** Config and markdown work-tree-read mutants now killed.
- **M2 fixed.** Symlink-mode mutant killed; symlink/gitlink rows and symlink-rooted unit tested.
- **M3 fixed, judgement agreed.** Undecodable Compose is dropped from the declared set and fails only its own unit (mutant keeping it killed). Services declared only there are unknown to W2, so a reference binds nothing (`bindValue` None): conservative, deterministic, never a false bind; the cost (sibling units bind less) is stated in the docstring and tested.
- **M4 fixed, rule agreed.** `exit_error` = `RunnerTimeout` iff `rc == 124 and elapsed >= wall_cap`, else `RunnerError`. Sound against `sandbox/README.md`: Python's wall clock starts before the runner's own timer, so a watchdog firing is never measured shorter than the cap; a tool's own early 124 and a sub-cap boundary race stay plain failures (fail-closed). Mutants `elapsed ignored`, `never`, `rc != 0` killed. Note: `STAGE_TIMEOUT_S` still equals the cap, so `_run`'s own timer normally fires first (also `RunnerTimeout`); the 124 branch is the backstop. Survivor `>=` -> `>` (equality of a float with the cap): nit.
- **M5 fixed.** Blame-first mutant killed (oversize markdown never blamed).
- **M6 fixed, choices judged.** Sizes come from `ls-tree -l` before any fetch; boundary mutants killed for shell, markdown, config blob, unit file count. Shell refuses the unit (the extractor's rule), markdown skips+counts. **Config > 8 MiB: skip + count, no File record — acceptable**: a File needs `contentHash` and `lineCount`, which need the bytes, and `header.skippedFiles` only admits `test`/`frontend-ignore` (schema), so there is no honest record to write; the file is absent from the graph and visible only in `BundleResult.report["skippedFiles"]`. Say so in the plan (plan (7) says oversize gets a File; it holds only between 512 KiB and 8 MiB).
- **m1 fixed.** `frontendVersion` = installed `tree-sitter-bash` (shell) / `"none"`; `normalizerVersion` = first 12 hex of the hash (`[:11]` mutant killed); `extractScriptHash` over the sorted module set, with an AST test that every imported `extractors.*` module is listed (dropping `launch.py` killed); `vendoredDirs` = `units.SKIP_DIRS` (mutant killed; `*.egg-info` suffix rule not expressible, documented); `builtTree` mutant killed.
- **m2 fixed.** `classify_file(rel, first_line)` public, `_file_language` delegates; agreement test on one tree. `build.py` still imports `shell_mod._check_rel` (private): nit.
- **m3 fixed.** Cache keyed (repo, sha, sha256 of the Compose path set); both key mutants killed.
- **m4 fixed.** `InvalidCommit` for anything but full lowercase 40/64 hex before any work (mutant killed).
- **m5 fixed.** `_counts` mutant killed. Live test not run; its `check_sandbox_ids` gap was not re-examined.
- **n1** documented in the module docstring. **MissingAuditIds** still raises (awaits the spec6 ruling; unchanged, out of this pass).
- **Open questions (facts only):** `load_globs()` still defaults to `cpg/eval/eval-manifest.yaml` (`globs.py:20`), `ccg/system.yaml` has no `promptBearingMarkdown`; `build.py` ignores any per-repo `vendoredDirs` (`manifest.py` has none), header uses `units.SKIP_DIRS`. Architect to rule.

### New findings

**N1 (major) — one hostile or merely odd name anywhere in the repository fails every host unit of that repository.** `_tree` calls `_check_tree_path` on **every blob of the commit**, not on the unit's files. Probe (verified): a commit holding `ok.json`, `ok.sh` and one unrelated blob named `unrelated\x01.txt` (or `.GIT/config`, a 300-byte name, `back\\slash.txt`): `build_unit` of `ok.json` (config) and of `ok.sh` (shell) both returned `extract:failed`. The same blast radius as Pass 1's M3, now reachable by any file in an untrusted repo (a legitimate file with a backslash in its name suffices), and it contradicts "a failure fails that unit alone". The existing multi-repository test only isolates *repositories*. Fix: list unvalidated in `_tree`, validate in `_unit_files` the paths it selects (the only ones ever written or read) plus the unit root; a non-selected hostile path is ignored. Tests: the probe above (two units, one repo, unrelated hostile blob -> both build) and the existing B1 matrix kept green (hostile path *inside* the unit still fails it).

### Minor

- Boundary mutants that survived: Compose blob cap `<=` -> `<`; `MAX_UNIT_BYTES` `>` -> `>=`; segment `>255` -> `>=`; path `>4096` -> `>=`; control chars `ord<32` -> `<31` and the `127` term. The caps are tested at the file-count and the three blob boundaries only; add the same exact-at/one-over row for these five (cheap with the existing `cap_attr` parametrisation) and one name each with `\x1f` and `\x7f`.

## Pass 3

Committed tree at `cd8daaf2` (gitsafe, P1-18, P1-03a at `192b3253`) plus the uncommitted P1-44 files. Run (cwd `cpg/`, `ulimit -v 3000000`): `test_build_dispatch test_joern_run test_units` = 323 passed (3 live deselected, not run); `test_linker_bind test_assemble_rows test_gitsafe` = 369 passed. Mutants on a fresh scratch copy (`cpg2/`).

**Verdict: P1-44 may be committed (MissingAuditIds pending P1-44a).**

- **N1 closed.** `_tree` validates nothing; `_unit_files` validates the unit root and the selected paths only. Mutants killed: validation moved back into `_tree` (`test_an_odd_name_outside_every_unit_fails_no_unit_of_the_repository`), root unvalidated, selected paths unvalidated (hostile-inside-unit still fails that unit).
- **Boundary mutants closed.** Compose blob cap `<=`->`<`, `MAX_UNIT_BYTES` `>`->`>=`, segment `>255`, path `>4096`, `ord<32`->`<31`, drop `127`: all six killed by exact-at/one-over rows.
- **Nit (private `_check_rel`) closed:** `shell_mod.check_rel` is public (`build.py:163`).
- **P1-03a M1 closed.** `ccg/globs.py` is in the markdown hash list; dropping it is killed (`test_the_extractor_hash_covers_every_module_of_the_extractor[markdown]`). The AST test scans each listed module, including `ccg/globs.py`, so a new `from ccg.<mod>` import fails unless hashed.
- **Known gap `UNHASHED_CCG_HELPERS = {ids, bundle, atrest, valuehash, units}`: acceptable, minor.** `from ccg import <name>` is exempt, and `ids`/`valuehash` do shape record content, so a change there leaves `extractScriptHash` unchanged. The header hash is provenance; a record change is still caught by the snapshot's `bundleHash` (P1-18), and a *new* unlisted helper fails the test. Say in the docstring that these five are covered by `bundleHash`, not by the header.
- **P1-03a M2 closed.** `BuildContext.manifest` defaults to `ccg/system.yaml`; `load_globs(ctx.manifest)`; mutant `load_globs()` (default eval manifest) killed by the fixture-manifest test; fail-closed rows (manifest without `promptBearingMarkdown` -> that markdown unit `extract:failed`, others build) are in the suite.
- **L1 allowlist.** `ORCHESTRATION = {build.py, refresh.py}` (`tests/test_linker_bind.py:957`); `_extractor_importers` compares relative paths. Mutant planting `import extractors...` in `ccg/ids.py` killed; the non-vacuity and "allowlist is exactly orchestration" tests hold. Survivor: widening the constant itself (`"ids.py"` added) passes. Minor: pin `ORCHESTRATION == frozenset({"build.py", "refresh.py"})` in one assertion.
- **Audit-id handling.** The only dependants are `build.py:430` (the pre-try `MissingAuditIds` for `HOST_KINDS`) and the header's `sandboxAuditIds` line (381); code units take the Joern run's ids and are untouched (`_build_code`, no `audit_ids` read). Nothing else in P1-44 depends on it; P1-44a replaces both lines.
- **Coherence.** `build.py` imports only committed modules (`gitsafe` via `blame.git`, `units.classify_file`/`is_skipped_dir`, `ccg/globs.py`); `units.py`/`blame.py`/`run.py` edits are in the commit; no uncommitted file outside the P1-44 set (plus the unrelated flow spikes and `extractors/shell/extract.py` `check_rel`, which P1-44 needs and must be committed with it).
