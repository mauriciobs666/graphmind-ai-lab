# Review: P1-11b `.mcp.json` exec-path launch fact (uncommitted diff)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-11b)

## Scope & verdict

Reviewed the uncommitted P1-11b change: `cpg/extractors/common/writer_rules.py` (`mcp_exec_launch_path`), `cpg/extractors/config/launch.py` (`mcp_drafts(..., exec_launch=)`), `cpg/extractors/config/extract.py`, the `p1_11b` tests in `cpg/tests/test_writer_rules.py` and `test_config_extractor.py`, and the two row-scoped `cpg/README.md` cells (writer_rules row, config row). Baseline: the P1-11b entry in `docs/plans/connected-code-graph-p1-extraction.md`, P1-17 (6) in `-p1-linking.md`, the P1-07 writer rules, root `REVIEW.md`.

**Verdict: approve with suggestions.** No blocker. The exception cannot emit anything `gate_argv` would not already accept as a `command` (see F-0 analysis); the weaknesses are one real test gap, one dead check, one 128/512 cap mismatch, and a plan/implementation divergence.

CPG: considered, not relevant — a self-contained string-rule change in a few functions, read directly; no cross-component impact question.

## Checks run (all verified, `cwd cpg/`, `ulimit -v 3000000`, `timeout -s KILL`)

- `test_writer_rules.py` + `test_config_extractor.py`: 856 passed.
- Neighbours in chunks: `test_linker*.py` 295 passed; `test_scip*`, `test_indexers_run`, `test_valuehash`, `test_bindparts` 712 passed; `test_atrest` 119 passed; `test_answer`, `test_eval_tools*`, `test_units` 1306 passed; `tests/slice` 88 passed, 3 deselected, **10 errors**.
- Mutation on a scratch copy (repo-shaped copy under the scratchpad, `.mcp.json` included; baseline 155 `p1_11b` passed): results in F-1/F-2.
- Probes by direct call and through `extract_config` (see (a) below).

## Findings

### F-1 (minor, verified) the whole-path `screen_fails(path)` has no discriminating test; the per-segment check is dead

Mutants on `writer_rules.py:309`: dropping `screen_fails(path)` alone **survives** (155 passed); dropping the per-segment check alone **survives**. They are not alike:
- **Per-segment check is an equivalent mutant.** `screen_fails(path)` is at least as strict as any segment: T1-T3/T6 operate on alphanumeric/`._:+,-` runs, which `/` and `@` split, so a failing run lies inside one segment and fails the whole; T5 (`len >= 20`, prefix after `=/:@,`) fails whole whenever it fails a longer-or-equal segment. Fuzz, 300,000 random multi-segment paths seeded with the T5 prefixes: 0 paths where a segment fails and the whole passes; 115,163 where the whole fails and no segment does. The line is dead code and its comment/docstring ("whole and per segment") promises a check that adds nothing. Remove it, or keep it and say it is belt-and-braces.
- **Whole-path check is a real, untested distinction.** Without it, `scripts/dir/ghp_abc.sh` and `github_pat_sAyIe-/y1` (a T5 prefix inside a short segment of a path of at least 20 chars) pass the allow-list. They are still redacted at the end, because `launch_argv_fields` runs `gate_argv` on argv[0] (whole-string `_plain_ok` screen), but the fact degrades to `command "<redacted>"`, `args []` instead of the intended `bash -c <redacted>`, a different shape. Fix: add one parametrized case to `test_p1_11b_the_screen_decides_a_secret_shaped_path` with `scripts/dir/ghp_abc.sh` and `github_pat_sAyIe-/y1` expecting `mcp_exec_launch_path(...) is None`; it reddens on the mutant. (Run on the mutant: both inputs; the current plant uses a long run that either check catches.)

### F-2 (minor, verified) allow-list cap 512 vs the gate's 128 gives a new degraded fact shape

`_EXEC_PATH {1,512}` admits 129-512 character paths; `_plain_ok` caps `command` at 128, so `extract_config` on `exec "$CLAUDE_PROJECT_DIR/abc/abc/...x.sh"` (164 chars) writes `command "<redacted>"`, `args []` (probe run), a bare redacted command with no `bash -c` trace, which `bind.py` cannot tell from "no program". No leak. Cap the path at 128 (then the string falls back to the old redaction), and move the boundary test from 512 to 128. The `{1,5120}` mutant is killed by the 512 boundary test, so the test would simply be re-pointed.

### F-3 (minor) knock-on is not one test: the whole conformance module errors

`tests/slice/test_linker_conformance.py` has **10 errors**, not one: line 85 (a shared fixture) asserts `['-c', '<redacted>']` and every test using it errors at setup (the named `test_the_real_extractors_redacted_command_leaves_the_cypher_chain_unbound` is only the one that pins it on purpose). All 10 trace to the same assertion (`AssertionError: the real extractor redacts the $CLAUDE_PROJECT_DIR word`), i.e. the old redaction, so it is one cause and is owned by the P1-17 fix batch, but that batch must re-point the fixture, not a single test. No other suite in the repo that touches `extract_config`/`.mcp.json` regresses (list above). Also `tests/test_linker_bind_launch.py:81` has a docstring claiming `extract_config` writes `bash -c <redacted>` for this shape; true of its hand-built fact, stale as a statement about the extractor. Not run-affecting.

### F-4 (minor, plan divergence) `${CLAUDE_PROJECT_DIR}/` and unbalanced quoting dropped

The plan's rule admits `\${CLAUDE_PROJECT_DIR}/` and an optional quote at each end independently; the implementation admits only `$CLAUDE_PROJECT_DIR/` and balanced quotes. Both narrowings only reduce reach (fail closed to the old redaction), and unbalanced quotes are a shell syntax error, so refusing them is right. The braced form is not used by the repo's `.mcp.json`, and `ccg/linker/bind.py:79` accepts both forms, so a braced launcher would extract redacted but could have bound. Acceptable as a first loosening, provided the plan row is amended to name `$CLAUDE_PROJECT_DIR/` only (owner `architect`) so the plan and code agree, or the braced form is added with its own refused-case test. Needs a decision, not a fix.

## Answers to the adversarial questions

**(a) Can anything outside the shape reach the output?** No. Verified by direct probe and `extract_config`: `fullmatch` over a closed ASCII class `[A-Za-z0-9._/@+-]` plus two literal prefixes; `\n`, trailing space, double space after `exec`, `;`, `$x`, `$(..)`, backtick, fullwidth `ａ` and U+2028 all return `None`; `${CLAUDE_PROJECT_DIR}` returns `None`; `a/../b`, `//`, `./`, leading `-`, `~x`, `a:b`, `a=b`, `a/[b]` return `None`. `..c` and `.git/config` pass, correctly (not traversal components). Decisive point: the emitted argv is `[path]` through `launch_argv_fields` -> `gate_argv`, and `_plain_ok` requires `<=128` chars, `_VALUE_CHARS` and `not screen_fails` on argv[0], so the output is bounded by what the gate already allows for any `command`. A short secret-shaped segment (`a/ghp_abc.sh`, `a/AKIAabcdefghij.sh`, `a/user@host.com`) passes — exactly as it would as a plain `command` today; T5 only acts at 20+ chars of whole text. Exposure the exception adds: the repo's own relative script path in place of `bash -c <redacted>`.

**(b) Survivors.** Length pre-filter 10^6: **equivalent** (verified: `{1,512}` plus the literal prefix cannot match above ~560 chars and `fullmatch` is linear over a closed class; only a hostile-input cost bound remains, not an output change). Group order: **equivalent** (exactly one group is non-`None`, path non-empty). Per-segment: **equivalent** (F-1). Whole-path alone: **real gap** (F-1, one case closes it). Other mutants killed: leading `-`, `..` segment, shell set (`zsh`), `-lc`, arg count, 5120 cap.

**(c) Narrowing decisions.** Quoted and unquoted forms: fine. Kiro agents excluded and `.mcp.json` only: right (a Kiro agent's `$CLAUDE_PROJECT_DIR` has no defined meaning; the exclusion pins as `test_p1_11b_the_exception_is_mcp_json_only` plus the Kiro test). `$CLAUDE_PROJECT_DIR/` only: F-4.

**(d) Provenance in `audit_records`.** Not needed. `_audit_launch_fact` checks `gate_argv(argv) == argv`; a bare path `command` row is a fixed point whatever produced it, and the confidentiality property is "nothing the gate refuses leaves", which the audit still proves. What the audit cannot show is "this row came from an `exec` string", but that is a derivation fact, not a safety fact: a provenance field would be a record-schema change (spec and validator) to guard against a writer bug whose worst output is already gate-bounded. The planted-offender and unreduced-shape audit tests (`...backstop_still_flags...`) are the right extent. Revisit only if a later exception ever emits something `gate_argv` would not accept.

## What's solid

- Single guarded entry point in `writer_rules.py` (the only place allowed to decide), `extract.py` gates it on `fmt == "mcp-json"`, input typed `object` and never raises (probes with tuples, wrong types).
- Linear-time shape, no nested quantifier, with a hostile-input test.
- Negative coverage is broad and kills real mutants (shells, `-c`, arg count, `..`, leading `-`, 512 boundary, repo's own `.mcp.json` yields `cypher-mcp/docker-run.sh` through the real extractor).
- README cells are row-scoped and accurate to the code (modulo F-4 wording if the braced form is added).

## Open questions

- F-4: should the plan row be amended to `$CLAUDE_PROJECT_DIR/` only, or the braced form added? (`architect`/`teco`)

## Proposed REVIEW.md lines

None.
