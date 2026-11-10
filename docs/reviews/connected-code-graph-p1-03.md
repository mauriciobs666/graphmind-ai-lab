# Review: C-1001 P1-03 (system manifest, unit discovery, sysname), delivered uncommitted

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

Reviewed against the P1-03 row (`docs/plans/connected-code-graph-p1.md`), its section in
`docs/plans/connected-code-graph-p1-extraction.md`, `docs/plans/connected-code-graph-egress-spec3.md` §4.2, §4.3, §4.4, §10.1,
`docs/plans/connected-code-graph-p1-graph.md` §7 and `docs/plans/connected-code-graph-bundle-spec3.md` §3.3, §3.4. Scope: the
P1-03 files under `cpg/` only (not P1-48b's `sandbox/*` or the infra/toolchain READMEs). Root `REVIEW.md` applied; no component
`REVIEW.md` exists for `cpg/`.

**Verdict (pass 1): needs changes** (one blocker, F1a; one design fork, F1b). Current verdict: `## Pass 2`.

CPG: considered, not relevant — the unit under review is new, uncommitted Python no loaded graph has seen; checked by execution.

Executed: the three new test files (284 passed, 0.9 s); the full `cpg/` suite (4593 passed, 1 skipped slow test, 164 s);
`python -m ccg.units --golden` on a copy reproduces the delivered golden byte for byte; 18 mutants on a copy (a `git ls-files`
export of the tree plus the new files, scratchpad `base/`; the real tree was never written, no FalkorDB touched). Mutants that
went red: the shell script's ACL regex (separator loosened; length bound 48 to 49), fallback root, `tsconfig.json` and
`requirements.txt` manifests, the census-scope list, S4 multicast, `SKIP_DIRS`, window and previewCap plumbing, unit-id input,
repoId lowercasing, directory-prefix exclusion, egress-schema `required`, the inline mode enum, sort key, `remote` rejection.
One mutant of mine (S2 message text) was a no-op, not a survivor.

## Findings

### F1a. blocker. Fallback units depend on file-name sort order (verified by execution)

`units.py` `_repo_units` adds fallback roots into the same `roots` set it consults for coverage (`not any(... _under(d, f) for d, l in roots)`),
iterating files sorted. Two trees of identical shape:

| tree | `discover_units(per_file=False)` |
|---|---|
| `a.py`, `b/x.py` | `[('.', python)]` |
| `b.py`, `a/x.py` | `[('.', python), ('a', python)]` |

Whichever sorts first decides whether a root-level file swallows the subdirectory. The real repo has no root-level code file today,
so the golden is right today; adding a root `zz.py` would silently change it. No test covers a root-level file beside a directory.
**Fix:** compute coverage against manifest roots only (first loop), never against fallback roots added in the second loop.
**Probe (a coverage probe, not a case list):** a property test that, for generated trees over (root-level file sorted before/after
the directories) × (nested manifest below a fallback directory) × (two languages) × (excluded file), asserts every code file of
language L is owned by exactly one unit of L (nearest enclosing root), and that shuffling the file names leaves the unit set's
shape unchanged.

### F1b. major, a design fork for the architect/stakeholder. Fallback units overlap manifest units

The golden contains `falkor-chat` (python, fallback) **and** `falkor-chat/server` (python, manifest): the fallback root is the
top-level directory, which encloses the nested manifest unit. Same for `salesperson` python beside `salesperson` javascript (different
language, fine) and any top-level dir with a nested manifest (`a` and `a/sub` in my probe). `Unit` is a fixed 4-tuple, so a consumer
cannot tell that `falkor-chat` must not stage `falkor-chat/server/**`; P1-04's `git archive` staging at `rootPath` would extract the
server files twice, in two bundles. The plan says only "a directory-per-language fallback" and "nested manifests give two units". Options:
(A) fallback root = the directory of each unowned file, collapsed to the highest ancestor holding no manifest unit below it (no
overlap); (B) keep the top-level rule and make P1-04's stager exclude nested unit roots (then say so in P1-04's row and test it);
(C) keep overlap and accept duplicates. I recommend A or B; see the question below.

### F2. minor. The exclusion matcher diverges from the census matcher in two latent ways (verified)

`system.yaml`'s lists equal the eval manifest's (test passes), but the *semantics* differ:
1. Census `path_excluded` (`eval/tools/export_kit.py:132`) excludes any path with a `connected-code-graph*` segment under `docs/`,
   including files below such a directory; `_glob_re` `fullmatch` does not. `docs/a/connected-code-graph-d/x.md` and
   `docs/connected-code-graph/y.md`: census True, manifest False. A differential over all 2477 tracked files gives no difference
   today, so it is latent.
2. A bare directory in `exclusions.paths` without a trailing `/` (`"dep"`) matches only a file of that name and silently excludes
   nothing (probe: `dep/x.py` stayed in).

The plan's invariant is "system scope equals census scope", and the guard compares strings, not behaviour. **Fix:** make a glob match any
leading segment prefix; reject (or treat as a directory) a bare `exclusions.paths` entry that names a directory. **Probe:** a test that runs
`_excluder` and `export_kit.path_excluded` over every tracked path plus planted paths for each axis (directory-named match, nested
`**`, bare directory, content-check file). Source of truth for the expectation is `export_kit`, not the new code.

### F3. minor. `git ls-files` lists the index, not a commit (verified by reading; no run needed)

`_git_files` reads the index/worktree. P1-04 stages with `git archive` of a commit; a staged-but-uncommitted directory yields a unit
whose archived tree is empty, and a tracked file deleted in the work tree stays listed. The plan's `discover_units(repo_root,
manifest)` has no commit parameter, so this is not a deviation, but P1-20 (refresh) should state which tree it discovers from
(e.g. `git ls-tree -r <commit>`). Ruling on item 5: `git ls-files` over a directory walk is right for the golden (untracked scratch
never moves it; tested), and the walk fallback is right for a fixture dir not yet tracked.

### F4. minor. Window values carry no unit

`rcaWindow` and `secretsWindow` are integers `>= 1` with no stated unit (commits? days?) and default `None`. Nothing fixed by the plan or specs
is contradicted: the plan names the fields only; the spec examples show only `repoId` and `policy.egress`; `previewCap` default 2 is
grounded (`docs/plans/connected-code-graph-graph.md` Q-R8, FR-23); `system`, `schemaVersion` and `path` are required by plan
behaviour (1) and the interface text. P3/P5 will have to choose the unit. **Fix:** a `description` in the schema stating it, or a note
in P1-03's row.

### F5. minor. The `pyproject.toml` edit is unnecessary, and D8's text is now false

The added `package-data` entry `"system.yaml"` is not a dependency (the case the header allows asking `teco` about) and not needed:
`ccg/schemas/*.json` was already covered, and `ccg/system.yaml` is loaded by explicit path with `path: .` resolved against the caller's
root, so shipping it in a wheel has no consumer. It does not conflict with P1-29: that step replaces the separate line `broker = []`
(`-egress-spec3.md` §10.2), and D8 reserves only the `broker` extra. But the header sentence "No other step edits this file" is now false
and D8 says the same. **Fix:** revert the line, or leave it and have `teco` amend D8 and the header comment. I would revert.

### F6. nit. Ruling on `[::ffff:127.0.0.1]` (item 6), and one inferred bound

Refused with `S3` (verified: `ipaddress.IPv6Address('::ffff:127.0.0.1').compressed` is `::ffff:7f00:1` on 3.12.3), which follows the spec's
"IPv6 literal equals its RFC 5952 compressed form", and S6 would refuse it independently (not in `127.0.0.0/8` or `::1`). The test at
`tests/test_manifest.py:286` states the reason. **Inferred, not run** (only 3.12 is installed): the stdlib's string form of mapped addresses
changed in Python 3.13, which would flip this verdict; `requires-python` is `>=3.12` and the plan pins 3.12, so note it where S3 is specified.

### F7. nit. Ruling on `tsconfig.json` and tests-path manifests (item 3)

Python and JavaScript only is right for P1 (bundle spec3 §3.3 tokens exist for the rest; P2). `requirements.txt` as a python manifest is
required, not invented: `export_kit.py:333` `_MANIFESTS` has it and `cypher-mcp` (a census unit) has no `pyproject.toml`; without it the
census list is not a subset of the units. Two latent differences from the census (`export_kit.list_units`): it omits `tsconfig.json`,
and it skips test paths for manifests; this code does neither. Today the manifest-derived roots are exactly the census's six units (checked
from the golden and `eval/eval-manifest.yaml`), so no effect. Not a finding to fix; record the choice.

## Rulings on the ten points

1. **Per-file units and the golden.** Default `per_file=True` is right: the plan's interface puts shell/config/Markdown units in
   `discover_units`. The golden pinning code units only (12) is a defensible reading: with per-file units it would be 1091 rows, 801 of
   them Markdown, and would go red on every doc commit in an agent-edited repo. The plan says only "`discover_units` on the monorepo is
   committed as a golden list", so this is an ambiguity, not a drift. Per-file classification is pinned by synthetic tests plus
   spot-checks of the real repo (`AGENTS.md`, `falkor-chat/scripts/start_agent_team.sh`, a `cpg/` negative). Record the reading in the
   plan's Done line.
2. **Invented manifest shape.** Nothing fixed is contradicted: see F4.
3. **Languages and `requirements.txt`.** See F7.
4. **Fallback unit-root rule.** Top-level directory per language is a reasonable reading of "a directory-per-language fallback", but see
   F1a (order-dependent: a bug) and F1b (overlap: a fork). Also note the real golden's `claude`, `docs`, `falkor-chat`, `salesperson`
   (python) and `skills` fallback units are outside the census's six units: any frame "over the same scope" in P1-36/P1-37 should name
   whether fallback units count.
5. **`git ls-files` vs walk.** See F3.
6. **S3 `[::ffff:127.0.0.1]`.** Correct, see F6.
7. **`normalize_repo_id` in `manifest.py`.** Acceptable. `ccg.ids` has no repoId function, so "repoId comes only from `ccg.ids`" is satisfied
   for the id hash and not violated; the steps match bundle spec3 §3.4 in order (NFC, strip, lowercase, run to `-`, trim, empty is an
   error, duplicates are an error); mutants for lowercasing and the duplicate check went red. Later consumers importing it pull
   `jsonschema`/`yaml` through the module, which is fine for the planned users.
8. **`pyproject.toml`.** See F5.
9. **Same verdict, `acl_user_re` vs `acl_user_ok`.** Holds and is a real check: the test extracts the script's own function text, runs it
   in bash over 5000+ users (every role × every name to length 5 over `{a,b,_,-}`, malformed users, `default`, 47–50 characters), and both
   mutants (loosened regex; length bound moved) went red. Bound: the harness feeds users line by line, so names with a newline or NUL are
   exercised only on the Python side; locale-dependence of `[a-z]` in bash was not testable (only `C`/`C.UTF-8` installed). Neither
   matters for names Redis ACL accepts.
10. **Reachability.** Everything the plan commissions loads: the proving manifest, the fixture manifest (four repos including the Java
    `fx-lib`, which yields no code unit in P1, by design) and the spec §4.2 example shape; P1-29 gets `Manifest.egress` as the loaded dict
    plus `repos[].policy.egress` for the `allowListDigest` input, and `canonical_endpoint`. Refused by design: `remote`, `groupings`,
    `registries`, and so on. I found nothing a later step needs that the strict schemas would refuse. Not reachable from this step, by
    ownership: S6 (broker).

## What's solid

- The test suite constrains the data plumbing: expected values are written out, the egress schema is compared byte for byte with the spec
  text, the mode-enum equality and `policy.egress`-mandatory checks fail under mutation, and the real-repo checks (census subset, no unit
  under `cpg/`, golden, id recomputation) fail under mutation. Strict YAML (duplicate keys refused) and `\A…\Z` anchors are a good fit for S5.
- `sysname.py` matches `-p1-graph.md` §7 (the isolation test is exhaustive to length 4 with a positive control; the formatting sweep has a
  detection test per shape).
- The golden is reproducible: the documented regeneration command gave the identical file on a copy.

## Open questions

- **For the caller (high-stakes fork): F1b.** Overlapping fallback units: option A (non-overlapping roots), B (overlap kept, stager excludes
  nested unit roots, P1-04 row amended) or C (accept duplicates). It decides P1-04's staging and the golden's `falkor-chat` row, so it
  should be answered before P1-04 dispatches; I recommend A unless the 12-unit census-style root (`falkor-chat`) is wanted as one unit.
- F3: which tree (index or a commit) does P1-20 discover from?

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: F1a, F1b, F2 (this review).
- "A function that assigns files to owners (unit discovery, a classifier, an exclusion matcher) is checked by one planted tree per axis:
  the same shape under permuted file names (sort order), an ancestor beside a descendant root, and, for a matcher with a counterpart
  elsewhere (a census list, `export_kit`), a differential over every tracked path. A guard that compares the two lists as strings is a bound."
No prune proposed.

## Pass 2

Executed against the revised tree: 998 tests of the three P1-03 files plus `test_rule_pack_schema.py` pass on a copy (scratchpad
`base2/`, a `git ls-files` export; real tree and FalkorDB untouched); my own probes ran on the real tree read-only.

- **F1a: fixed.** The coverage test now uses `manifest_roots` only. My probe: 2985 random trees (root files, nested manifests, 6
  directories) each built twice under renamed files (sort order flipped): 0 differing unit sets; `a.py`+`b/x.py` and `b.py`+`a/x.py`
  now give the same shape. Mutant restoring the old coverage set (`roots`) goes red in both order tests. Bound: the implementer's
  property test covers 2^len(_POOL) subsets x 3 name shapes; I did not re-derive its pool.
- **F1b: partly resolved; one plan edit still owed.**
  - Real repo: verified zero same-language ancestor/descendant pairs in the golden (12 units; `falkor-chat` is now javascript,
    `falkor-chat/scripts` python, `falkor-chat/server` python, matching the implementer's account). `salesperson` x2 differ in language.
  - Is the forced overlap unavoidable? Yes, **given directory-rooted units**: `fc/x.py` beside manifest `fc/server` can root only at `fc`,
    and `.` for a root-level file beside any nested manifest. The alternatives are file-rooted code units or an exclusion list on the unit,
    neither in the plan's 4-tuple. Confirmed by running `fc/x.py`+`fc/server/...` -> `[fc, fc/server]`.
  - Harmless to P1-04? Not shown. Nothing in P1-04's row (`-extraction.md` § P1-04: "stage via `git archive`") excludes nested unit
    roots of the same language, so `fc` would stage and extract `fc/server/**` too. **Owed:** one sentence in P1-04's row (staging excludes
    descendant unit roots of the same language) with a test; owner `architect`/`teco`. Today's real repo is unaffected.
  - Residual, not forced: a fallback unit can enclose another fallback unit (`a.py`+`b/x.py` -> `[., b]`; `a/y.py`, `a/b/y.py` beside
    manifest `a/c` -> `a`, `a/b`). The delivered test pins `[(".", "python"), ("b", "python")]`, and its "overlap only when forced" assertion
    checks fallback-over-manifest only. Minor: the same stager rule covers it; or merge a nested fallback root into its fallback ancestor.
- **F2: fixed.** Differential test is not vacuous: it runs over all tracked paths plus 18 planted paths covering the dir-named-glob,
  nested `**`, prefix-sibling, bare-directory and content-check axes, with positive controls. Mutants: glob checked against the whole
  path only (red, both new tests), plain entry matched exactly (red x4), trailing `/` not stripped (red x4). My independent differential,
  exhaustive over 4-segment paths from 13 segment names plus 60000 random paths, differed from `export_kit.path_excluded` only on
  non-paths (the bare directory name `cpg`, and paths below the file `docs/BACKLOG.md`).
- **F3, F4, F6, F7: unchanged** (not addressed; none blocks).
- **F5: refuted by the implementer; my recommendation was wrong.** Removing the entry on the copy fails
  `tests/test_rule_pack_schema.py::test_every_data_file_under_ccg_is_covered_by_a_package_data_glob` (line 1374: every non-`.py` file under
  `ccg/` must be covered by a `package-data` glob), and the plan fixes the path `cpg/ccg/system.yaml`, so the edit is forced. Still owed: D8 in
  `docs/plans/connected-code-graph-p1.md` still says only P1-01 and P1-29 edit the file (`teco` to amend). The rewritten header comment in
  `pyproject.toml` reads awkwardly ("No other step edits its dependencies, and P1-03 edits it once more…"); wording nit.
- **Cross-fix check.** F2's prefix matching with F1's roots: an excluded nested manifest or an individually excluded `pyproject.toml` leaves
  its files to a fallback unit and the golden test and `test_a_directory_exclusion_removes_units_below_it` still pass; the golden
  regenerates identically. No conflict between the fixes found.

**Updated verdict: approve with suggestions.** No blocker remains. Conditions: P1-04's row gains the nested-unit-root exclusion
(F1b), and `teco` amends D8 (F5). Pass-1 proposed `REVIEW.md` line stands (source F1a, F1b, F2).
