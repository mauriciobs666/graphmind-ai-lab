# Review: C-1001 P1-01 — layout and Python dependency intake (implementation)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

**Scope.** The P1-01 working-tree changes, uncommitted, delivered by `devops`. Reviewed against step `## P1-01` of
`docs/plans/connected-code-graph-p1-extraction.md`, index decisions D8/D9 of `docs/plans/connected-code-graph-p1.md`,
and root `REVIEW.md` (there is no `cpg/REVIEW.md`). Files: `cpg/pyproject.toml`, `cpg/extractors/__init__.py`,
`cpg/clients/__init__.py`, `cpg/eval/tools/audit.py`, `cpg/tests/test_eval_tools.py`,
`cpg/tests/fixtures/headless/derived.absent-tool.{stream,transcript}.jsonl`, `cpg/README.md`. The `cpg/infra/` changes
belong to a concurrent unit and were not reviewed.

**Verdict: approve with suggestions.** There are no blockers and no majors. The O3 rule is narrow. On the real S54
logs it changes exactly the three absent-tool calls and nothing else. Every guard in the rule is killed by a test.
Three minor findings are stale text that the change contradicts (the `pyproject.toml` header, the `cpg/README.md`
Setup sentence, and the `audit.py` docstring). Fix them before commit.

CPG: considered, not relevant — the unit is a dependency manifest, two empty stubs and one ~20-line function in
`cpg/eval/tools/audit.py`. All of it was read in full and run against real logs, and no `cpg_cpg` graph covers it.

### Checks run

- `cd cpg && .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_eval_tools.py` gave **434 passed**, run once,
  on the real tree.
- **Mutation** on a scratchpad copy of `eval/tools/audit.py` (the tree was not edited), running the
  `absent or refused or denied` tests. Seven mutants, and each one turned at least one test red:
  - dropping the called-name equality;
  - dropping `name not in allowed`;
  - dropping `d is None`;
  - `fullmatch` changed to `search`;
  - `"absent-tool"` changed to `"denied"`;
  - removing the `DENIAL_RE` branch (an existing refusal path);
  - dropping the `is_error` requirement.
  
  In the copy, `test_derived_fixtures_match_builders` and `test_list_cypher_tools_on_the_real_server` also fail at
  baseline. Both depend on the copy's location, so they say nothing about the mutants.
- **Real-shape replay**, read-only, with output going only to the scratchpad. The HEAD and working-tree `audit.py` were
  both run over every session in the real eval dir `~/.local/share/ccg-eval/cf7b2042cce7/` (20 sessions).
  - The only difference is in session `bf8e71bc` (S54 breach2). Old: `void`, `mcp-call: 3`. New: `clean`, with the
    three calls refused and `reason: absent-tool` (`mcp__cypher__query`, `…search_documents`, `…ingest_document`).
  - The other 19 sessions produce identical verdicts and refused/void tallies.
  - The `--details` lines are identical apart from the new `reason` key. So the existing refusal paths are unchanged
    on real data.
  - `fb75b8e5` (the genuine guard breach, `path-outside-root: 2` executed) is still `void`.
- **Behaviour 1:** checked from the installed state. I did not re-run `pip install`, because it would mutate the venv.
  - `python -I -c "import extractors, clients"` run from `/tmp` resolves to `cpg/extractors/__init__.py` and
    `cpg/clients/__init__.py`.
  - The `ccg` metadata carries `tree-sitter==0.25.2`/`tree-sitter-bash==0.25.1` (extra `extract`) and `mcp==1.28.1`
    (extra `engine`).
  - `pip check` reports no broken requirements.
  - `tree-sitter-bash` parses a pipeline without error under the pinned `tree-sitter`.
- **Pins:** `cypher-mcp/requirements.txt:10` bounds `mcp>=1.28,<1.29`, and `cypher-mcp/.venv` resolves to `1.28.1`.
  The pin is the cypher-mcp venv's resolved version, as the comment says. The `tree-sitter` pin equals `spike`'s.
- **Behaviour 2** (the full default suite is green before and after, R-7): **not run by me**. The brief limited me to
  one module. Only the 434 tests above are evidence.

## Findings

### F1 · minor · `cpg/pyproject.toml` header contradicts the new block (verified, read)

`pyproject.toml:6-10` still says every dependency is declared up front "so no later step edits this file" and dates
the pins 2026-10-29. Line 41 now says the file is "edited once, by P1-01" and line 43 repeats "stops and asks
`teco`". `description` (line 4) still reads "P0 contract code". This is what the brief flagged. A reader of the header
learns a rule the file itself breaks 30 lines later.

**Fix:** rewrite lines 6-10 in place rather than adding another paragraph. For example: "No base dependencies: each
phase's dependencies are declared in the extras below by its intake step (P0 plan §5.2; P1 index D8, step P1-01) at
exact pins verified installable on the date given per block. A later step that finds one missing stops and asks
`teco`." Then cut the duplicated sentences from the P1 block's comment (lines 41-43) down to the date. Update
`description` to cover P1, or make it phase-neutral.

### F2 · minor · `cpg/README.md` Setup still says "No other step edits that file", and has no install line for the new extras (verified, read)

`cpg/README.md:23-24`: "Every P0 dependency is declared once … **No other step edits that file.**" It is now false,
because P1-01 edited the file. My grep for the phrase missed it at first because the sentence wraps across a line
break. Also, the install block (lines 18-20) has no command for behaviour 1's
`pip install -e '.[dev,live,spike,extract,engine]'`. As a result, the README documents the three new extras without
saying how to install them.

**Fix:** reword the sentence the same way as F1 ("each phase's intake step: P0 §5.2, P1-01/D8"), and add one install
line for `.[dev,extract,engine]`. The rest of the README P1 rows (Layout, Tests markers) are P1-close work per index
§"documentation" and are out of scope here.

### F3 · minor · `audit.py` docstring: the new rule was appended, and the older text it contradicts was left (verified, read)

`audit.py:24-27` still says a call is refused "only on positive evidence: a … decision … not accept/allow, or
`is_error` with a result text that STARTS with a denial message … Anything else … counts as successful (voids)".
The new paragraph at lines 37-40 adds a third kind of evidence. Line 21's `--details` field list also omits `reason`.
That paragraph is the contract a GC re-auditor reads first, and it now contradicts itself.

**Fix:** fold the absent-tool clause into the 24-27 list, add `reason` to line 21's field list, and delete
lines 37-40. Also drop "not yet observed (S54 confirms)", because S54 has now observed the shape.

### F4 · nit · a list-shaped `content` is untested (inferred)

`_refusal` requires `isinstance(content, str)`. A `tool_result` whose content is a list of text blocks fails closed
(it voids), which is the safe direction, but no test pins it. The real S54 shape is a string (confirmed in the
breach2 stream). One parametrize row in `test_audit_absent_tool_text_is_matched_exactly` with
`[{"type":"text","text": absent_tool_text(...)}]` would pin it. Its mutant is replacing the `isinstance` guard with
`str(content)`.

## O3 over-breadth assessment (the brief's specific question)

The rule refuses only when **all** of the following hold:

1. The result is `is_error`.
2. Neither side has a permission decision.
3. The whole content is exactly the harness envelope.
4. The envelope names the called tool.
5. That tool is outside `--tools`.

Each condition is killed by a mutant.

**Can a genuine executed call be miscounted?** For a tool that is in the session but outside `--tools` (the only kind
that could execute and then emit look-alike text), `check_init` (`audit.py:334`) already voids the session, because
the init tool list must equal `allowed`. A missing init voids too (`init-missing`). The agent cannot author a
`tool_result`.

The only gap I found is inferred, not observed. A tool registered after init, absent from the init list, that returns
exactly the envelope string as an error would be counted refused. That is below the bar for a finding. I record it so
a future dynamic-MCP setup re-checks it.

**Existing paths:** the denied branch is ordered first and behaves as before. That is shown by mutant 6 and by the
real-log replay. The fallback `t_results.get(i)` returning `None` gives "not refused", as the old `False` default did.

## What's solid

- The extraction refactor keeps the raw evidence, `(decision, is_error, content)`, and classifies it per call
  together with the call's name. That is what allows the called-name check. Before, the boolean was computed before
  the name was known.
- The test set has a positive control (the derived stream) plus negative controls for every guard. That includes the
  transcript-only path, and the CLI exit code, which is the gate's actual interface.
- The fixtures are generated by `all_derived()` and pinned by `test_derived_fixtures_match_builders`, as the
  `mcp-call` streams are.
- The stubs are minimal, and `packages.find` matches D8 verbatim. The `slice` marker text restates D9's rule.

## Open questions

- None that block. Whether the S54 test report's O3 item (`docs/test-reports/connected-code-graph-p0-ground-truth-report.md:43`)
  gets a disposition pointer is `qa-engineer`'s or `teco`'s call. The P1 index row at line 189 already records the
  ruling's home.
