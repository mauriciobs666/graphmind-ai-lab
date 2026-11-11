# Review: C-1001 P1-03a — production prompt-bearing-Markdown globs from the system manifest (implementation)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Reviewed the uncommitted P1-03a against `docs/plans/connected-code-graph-p1-linking.md` (P1-03a, P1-44 (6)) and `docs/plans/connected-code-graph-p1-extraction.md` P1-03, root `REVIEW.md`: `cpg/ccg/globs.py` (new), `cpg/ccg/manifest.py`, `cpg/ccg/schemas/system-manifest-v1.schema.json`, `cpg/ccg/system.yaml`, `cpg/fixtures/system1/system.yaml`, `cpg/extractors/markdown/globs.py`, the three test files, the two README rows, and the `ASSEMBLY_CODE` line. `ccg/build.py` and `tests/test_build_dispatch.py` (P1-44, another unit) were read only where they bear on P1-03a.

**Verdict: approve with suggestions.** The core is sound, but M1 and M2 must close before P1-44 closes. Production never reads the eval manifest, the validator is value-free and hostile-input-safe, and include/except semantics are identical to the old matcher. The two majors sit in P1-44's files and are consequences of this move.

CPG: considered, not relevant — a single-module change to a Python validator/loader, whose callers I swept with grep, and `cpg_cpg` was not consulted.

Verified by running (cwd `cpg/`, `ulimit -v 3000000`, `timeout -s KILL`):
- `tests/test_manifest.py` plus `tests/test_markdown_extractor.py`: 308 passed.
- `tests/test_fixtures.py`: 78 passed, 2 deselected by the suite's own markers.
- `tests/test_linker_bind.py` plus `tests/test_build_dispatch.py`: 337 passed. This includes the L1 layering pin.
- Scratch probes and mutants under `p103a_gate_*` names in the session scratchpad; the 7 GB scratch copy was removed. No tree edits, no `falkordb-dev`, `~/.config/ccg` or held-out eval data touched. The unrelated scratchpad mutation script is ignored as instructed.

## Findings

### M1 (major, verified by reading) — the markdown `extractScriptHash` is blind to the code that now decides `prompt-md` vs `doc`

- `ccg/build.py:343` lists `("extractors/markdown/extract.py", "extractors/markdown/globs.py", "extractors/common/writer_rules.py")`. The test pin at `tests/test_build_dispatch.py:1038` repeats that list. `_extractor_hash` (build.py:353-355) hashes exactly those files into the header's `extractScriptHash`, and into `normalizerVersion` via `_versions`.
- This unit moved the matcher, the `Globs.classify` logic and the validator out of `extractors/markdown/globs.py` into `ccg/globs.py`. `extractors/markdown/globs.py` is now a 56-line loader that re-exports them. A change to `_match`, `_seg_match`, `classify` or the caps therefore leaves the markdown bundle's extractor hash unchanged. A stale bundle then claims "same extractor" for different classification logic, which is the failure the docstring at build.py:350-351 exists to prevent.
- The `ASSEMBLY_CODE` addition (`ccg/assemble/snapshot.py:45`) covers the assembler's hash only, not the bundle header's.
- **Fix** (P1-44's file, a one-line change): add `"ccg/globs.py"` to `_EXTRACTOR_MODULES["markdown"]` and to `MODULES["markdown"]`. Better, make the pin derive from a closure of the extractor's `ccg`/`extractors` imports (an AST walk from `extractors/markdown/extract.py`). A hand list drifted here, and `extractors/markdown/extract.py:40` and `writer_rules.py:49` also import `ccg.*`. That is wider than this unit; at minimum add the one line.

### M2 (major, verified by grep, behaviour traced not run) — the planned fail-closed test does not exist, and the default-manifest branch of `build.py` is exercised nowhere

- P1-03a's Files list names `tests/test_build_dispatch.py` for "the fail-closed refresh test", and P1-44 (6) requires "a refresh with no explicit globs fails closed". `grep -n "ManifestError\|fail.*closed\|manifest=\|system.yaml" tests/test_build_dispatch.py` finds nothing relevant. `ctx_for` (`test_build_dispatch.py:78`) always injects `globs=EVAL_GLOBS`. `tests/test_build_live.py:41` builds a `BuildContext` without globs but is opt-in.
- So `ctx.globs or load_globs(ctx.manifest)` (build.py:322) is never run by a default-suite test. Both the default `ctx.manifest = CPG/"ccg"/"system.yaml"` (build.py:122) and the `ManifestError` → `extract:failed` mapping are unpinned. The mapping rests on `fail_code` (build.py:141-147, anything unlisted → `EXTRACT_FAILED`) and the `except Exception` at build.py:442, with `_markdown_records` called inside that try. I traced this by reading and did not run it.
- Confirmed by reading and one direct call: `load_globs("ccg/system.yaml")` returns 9 include and 3 except globs, so the default path works today.
- **Fix:** add to `test_build_dispatch.py` (a) a markdown unit built with a `BuildContext(manifest=<system manifest without the block>)` and no `globs`, asserting `failReason == "extract:failed"`, no bundle written, and no manifest or glob text in the result; (b) one with `manifest=` the real `ccg/system.yaml` and no `globs`, asserting the unit builds with the same classification as the proving system's block. Mutants that must turn it red: the default pointed at `cpg/eval/eval-manifest.yaml`, and `load_globs` falling back to a built-in list.

### m3 (minor, verified) — the strict loader leaks a raw `TypeError` for an unhashable mapping key

- Input `? [a]\n: 1\npromptBearingMarkdown: {include: [a.md]}` (also inside the block) makes both `extractors.markdown.globs.load_globs` and `ccg.manifest.load_manifest` raise `builtins.TypeError: unhashable type: 'list'`.
- In `load_globs` this is new: the old `yaml.safe_load` raised a `ConstructorError`, a `YAMLError` mapped to `ManifestError`. The docstring promises "`ManifestError` if …", and `_construct_mapping`'s `key in seen` is the cause.
- Fail-closed still holds in production, because build.py:442 catches everything. A direct caller doing `except ManifestError` is not protected.
- **Fix:** in `_construct_mapping`, `try: key in seen` / `except TypeError: raise ManifestError("unhashable key")`, or widen the `except` in `load_globs` to `(OSError, yaml.YAMLError, TypeError)`. Add the test with the input above. `manifest._construct_mapping` has the same pre-existing defect.

### m4 (minor, verified by mutation) — matcher anchoring and path caps are unpinned

- The plan says semantics are unchanged, and I confirm they are: see "What's solid". But the test set does not constrain the matcher's end anchor or its path caps. On a scratch copy these mutants all SURVIVED the full `test_manifest.py` plus `test_markdown_extractor.py` run, with baseline failures subtracted:
  - `ccg/globs.py` `r = j == len(segs)` → `r = j >= len(segs) - 1`. `Globs(("a/*.md",),()).classify("a/b.md/c")` is `doc` today, and the mutant makes it `prompt-md`.
  - `_MAX_SEGMENTS` 64→65, `_MAX_SEGMENT_LEN` 255→256, and removing the path-cap `return False` entirely.
- Killed, for contrast: cap 64→65, pattern length 200→201, except ignored, include ignored, `**` one-segment-only, `_seg_match` end anchor, `?` loosened, memo removed (via `test_glob_matching_is_not_exponential`), each validator rule that is not redundant (partial `**`, alphabet, `..`, unknown key, empty include), echoing the glob in a message, the block-equality test (dropping one `except`, altering one `include`), and the strict loader.
- **Fix, as a coverage probe not a case list:** a differential test of `Globs.classify` against an independent reference (a `re`-translated glob, segment-wise) over the product of pattern shapes and path shapes. The axes: number of segments (0 to cap+1), `**` position (leading, middle, trailing, doubled), `*`/`?` position within a segment, and a trailing extra segment. The reference must not be derived from the code under test. Add one assertion per cap at cap and cap+1.

### m5 (minor, verified) — two caller routes parse the same block with different strictness

- `Manifest.prompt_markdown` (schema + `check_block`, loaded by `load_manifest`) has no production consumer yet; the sweep finds only tests. `build.py` goes through `load_globs` (`check_block` only, no schema, no egress checks). Both use the one validator, so the grammar cannot drift. The route is not a defect today.
- **Suggestion:** decide in P1-20/P1-44 which route production uses. `refresh` already loads the manifest, so passing `manifest.prompt_markdown` as `ctx.globs` removes a second parse of an operator file and makes the `BuildContext.manifest` field and its default unnecessary.

### n6 (nit)

- `schemas/system-manifest-v1.schema.json` description still says the alphabet and caps are "the ones of extractors/markdown/globs.py"; they now live in `ccg/globs.py`.
- `manifest.py` docstring has a stray lowercase sentence, "``exclusions`` and a mandatory ``policy.egress``. an optional top-level …".
- `test_manifest.py` `_block(doc, **kw)` takes an unused `**kw`.
- `p.startswith("/")` and `"//" in p` in `_check_patterns` are subsumed by the empty-segment rule. Removing them left the suite green, an equivalent mutant, so they are redundant and harmless.
- `test_the_widest_accepted_block_matches_a_hostile_path_in_bounded_time` finishes in about 1 ms, because its path misses on the first segment. It guards only against gross regressions, which `test_glob_matching_is_not_exponential` already does.

## Requested checks, one line each

- **Production never defaults to the eval manifest.** Confirmed. The only production caller of `load_globs` is `ccg/build.py:322`, and `BuildContext.manifest` defaults to `CPG/"ccg"/"system.yaml"` (build.py:122). `MANIFEST` is gone, and `test_the_manifest_path_is_a_required_argument_with_no_default` asserts `TypeError` and `not hasattr(mdg, "MANIFEST")`. An unfiltered grep of `eval-manifest|eval_manifest` over `ccg/` and `extractors/` finds only docstring/comment text in `extractors/markdown/globs.py`.
- **Fail-closed on an absent block.** The behaviour is correct by trace (see M2); the test in `test_build_dispatch.py` does not exist.
- **Validator hostility.** Run through `check_block` and `load_globs`:
  - Refused value-free: absolute, `..`, `.`, NUL, `//`, trailing slash, 201 chars, 65 patterns, `***`/partial `**`, non-str, null, `yes`, `0755`, a trailing newline, tab, unicode lookalikes and NFC `é` (ASCII-only alphabet), and unknown keys including an `on:` key.
  - Duplicate keys: refused at the top level and in the block.
  - YAML features: merge keys (`<<`) and `!!python/*` tags are refused as `ConstructorError` → `ManifestError`. Aliases to lists are accepted and harmless; an 11-level alias bomb loads in 0.0 s, since PyYAML shares the objects.
  - Case is significant by construction. Only the unhashable key (m3) escapes as the wrong exception.
  - The planted `S3CR3T[a]/x` and `/S3CR3T` globs are not echoed by `load_globs`, `check_block` or `load_manifest`: a refused glob is reported only as `include[i]`/`except[i]`, and the schema route as `refused by the schema (<validator>)`.
- **Matcher complexity.** Polynomial. The widest accepted block (64 include plus 64 except patterns, 200 chars, `**/<stars>` segment repeats, 64 segments × 255 chars) classified in 0.14-0.26 s, against the delegate's `< 5 s` bound (about 20× headroom). The memoised `go` is O(patterns × segments) states, each calling the O(p·t) `_seg_match`.
- **Semantics identical to the old matcher.** Old `git show HEAD:cpg/extractors/markdown/globs.py` was loaded from a scratch file and compared on the eval manifest. Include/except tuples are equal, and `classify` agrees on all 1314 paths: 974 tracked `.md` files, each plus synthetic `x.md` and `kaizen/x.md` variants under every ancestor directory, and edge paths (empty, absolute, `a//b.md`, wrong-case, nested `README`). 139 tracked files are `prompt-md`. The matcher bodies are byte-identical. The intended differences are only a value-free message, optional `except`, the strict loader, and system-manifest acceptance.
- **Proving system block equality.** `test_the_proving_systems_prompt_globs_equal_the_eval_manifests_block` reads the expected side from `eval-manifest.yaml` independently, and anchors counts 9 and 3 plus `claude/**/kaizen/**`. The mutants dropping one `except` and altering one `include` in `ccg/system.yaml` both turn it red.
- **Layering.** `ccg/globs.py` imports only `dataclasses`. `ccg/manifest.py` imports `ccg.globs`. `extractors/markdown/globs.py` imports `ccg.globs`. The L1 pin in `tests/test_linker_bind.py` passes.
- **Mutation coverage.** See m4 for the survivors; the cap, the equality, except handling and the validator rules are killed.

## What's solid

- The move is a faithful lift: the matcher body is unchanged, and the old behaviour is reproduced on a corpus that includes every tracked Markdown file.
- The "no default" contract is enforced by signature plus a test, and the removal of the `MANIFEST` constant is asserted.
- The value-free message discipline holds on every route, including the schema route's special case, with a parametrized planted-secret test that fails when the echo mutant is applied.
- The fixture system's block is minimal and its test checks both classes (three prompt files `prompt-md`, `docs/README.md` and `README.md` `doc`). `test_fixtures.py`'s planted-failure test was switched to the fixture system's own manifest and still turns red.
- The `ASSEMBLY_CODE` addition is correct: `ccg/globs.py` is in the tuple at `snapshot.py:45`, and the assembler's `codeHash` therefore changes with the validator.

## Open questions

- M1: is the owner of P1-44 prepared to take the `ccg/globs.py` entry in `_EXTRACTOR_MODULES` as part of that unit's gate round, or should the coordinator route it back here? I recommend P1-44, as the file and its pin test are its own.
