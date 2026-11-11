# Review: P1-46 shell `mcp-call` consumer pack

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Reviewed (uncommitted delivery, code and tests as ground truth) against `docs/plans/connected-code-graph-p1-linking.md` P1-46 and the P1-16 "Also" text, `docs/plans/connected-code-graph-rule-pack-spec.md` §1.2, the committed pack runtime `cpg/ccg/packrt/` and the shell extractor `cpg/extractors/shell/`, and fixture rows LNK-19/LNK-20.

**Verdict: approve with suggestions.** No blocker. The pack does what the P1-46 text and the P1-16 runtime allow, with two known gaps (LNK-20 and the reason code) left to the architect. One major: a silent wrong confirmed key (F1).

CPG: considered, not relevant — the unit is a YAML pack plus tests over a pure-Python runtime; behavior was established by executing the extractor and runtime, not by graph queries.

Executed: `tests/test_pack_shell_mcp_call.py` + `test_rule_pack_leaf_mutation.py` + `test_rule_pack_schema.py` = 742 passed, 1 skipped (the slow `CCG_LEAF_MUTATION` one, not run); `tests/test_packrt_*.py` = 152 passed. Probe scripts and 9 own mutants ran in-process (scratchpad; copies of the pack under `cpg/.ccg-work/gate-p1-46/shell-mcp-call/`). Not run: the slow leaf test, the delegate's 187/138 totals (a subset of mine). Read only: the `SERVED` guard internals, `negatives.yaml` rationale.

## Findings

### F1 (major, verified) — the tool is `a0` blindly: `mcp-call --json reindex` is a confirmed consumer of tool `--json`
`pack.yaml` rule `call`: `extract: {tool: {arg: a0}}`. Executed through the real extractor: `mcp-call --json reindex` gives `('--json', 'confirmed', 'key-match')`. The consumer is a confident wrong key (not `possible`), so the linker (P1-17) will link it to no provider and silently lose the real `reindex` edge. This is the P1-46 text ("the tool from argument `a0`") and the fixture only has the flag-free form, so it is conformant — but the unsoundness is unpinned and unmentioned. Suggestion: add a negative/known-limit test pinning the present outcome (as done for the quoted eval), and return the question to the architect: either a pack-format way to skip `-`-prefixed arguments (a `value` regex on a0 for the confirmed rule, `possible` otherwise) or an explicit note in P1-46. Not a fix for this unit alone, since the format has no negated arg match.

### F2 (minor, verified) — delegate's two accounts are TRUE
- LNK-20: `fixtures/system1/fx-svc/scripts/dyn.sh` through `extract_shell` gives ONE `valueSet` on the `eval` call: `a0 = template[const "mcp-call ", unknown runtime-built]`; `run_packs` yields `[]` (executed; pinned by `test_a_quoted_eval_template_is_not_derivable_by_the_current_pack_format`). `eval 'mcp-call reindex'` (and `eval "mcp-call reindex"`) arrives as `a0 = const "mcp-call reindex"`, so rule `eval-call` (`a0 == mcp-call`, exact) does not match either; executed, no consumer. Note this means any quoted eval, even fully literal, is a silent miss, not just LNK-20; the test file only discusses the template case. Ruling is the architect's.
- `shell-dynamic` on CONSUMES: `ccg/registry/reasons.yaml:24` lists `on: ["shell DATA", callsite]`; executed `pack_errors` with `reason: shell-dynamic` on either rule: `'shell-dynamic' is not valid on CONSUMES mcp_tool`. The delegate's use of `runtime-built` (valid on CONSUMES, `reasons.yaml:25`) is the only valid choice today; the P1-46 text asking for `shell-dynamic` is not satisfiable, and the fixture row LNK-20 expects `shell-dynamic` on the Link (a linker-level reason, so a pack consumer carrying `runtime-built` may still be right; architect).

### F3 (minor, verified) — `behaviour_failures` cannot see a structural widening; the negatives file is not dead weight
Own mutant: `callee: {glob: "*mcp-call"}` on rule `call` is caught only by `negative call/similar-name` (`full: 1`, `no-negatives: 0`), because the positive cases exercise only `mcp-calls`. The delegate's survivor claim (positives alone kill all 24 scalar leaves) holds, but negatives are the independent kill for matcher-shape changes the scalar-leaf harness never makes. Keep both. Suggestion (nit): add `xmcp-call` as a positive case to `behaviour_failures` so the glob/regex widening is also killed without negatives.

### F4 (minor, verified) — a removed `status: possible` on `eval-call` is accepted as `confirmed` with reason `runtime-built`
Mutant `eval -> confirmed` (key `status` dropped) passes `pack_errors` and emits `confirmed/runtime-built`; only the behaviour cases catch it. The validator checks an explicit status against the reason (`rulepack.py:481`) but not a missing one. Pre-existing format behavior, not this unit's fault; noted for the architect.

### F5 (nit, verified) — over/under-match probes (crafted snippets, executed)
Match as intended: pipe, `$(…)`, `if mcp-call …`, `FOO=1 mcp-call x`, a function named `mcp-call` called after its definition (confirmed, arguably wrong but harmless). Not matched, deliberately or acceptably: `sudo`/`exec`/`nohup`/`time`/`command`/`bash -c "mcp-call x"`, `/usr/bin/mcp-call`, heredoc body, comments, strings. No over-match found against `negatives.yaml` (all pass) or my 18 snippets, other than F1. Under-match on prefix wrappers (`exec mcp-call reindex`) is a silent miss; mention in the step report.

## Judgements on the asked points

1. **Pack rules**: conformant to the format and P1-46 except the reason code (F2). `refresh.sh` derives LNK-19: `reindex`, `confirmed`, `key-match`, qname `scripts/refresh.sh::<script>` (executed, pinned). `dyn.sh` derives nothing (F2). Exact-name matching verified (`./mcp-call`, `mcp-calls`, `MCP-CALL`, `mcp-call ` do not match). Scope: `callKind: command` is emitted only by `extractors/shell/extract.py:614` (the config extractor emits launches, not callsites), so the scope is sound in practice, but it rests on an extractor convention, not the format; a later extractor emitting `command` callsites would match.
2. **`SERVED` guard**: meaningful and scans the right files (`ccg/packrt/*.py`, AST strings/names/attrs). Derives `mcp-call` (literal) and `eval` (token) from the pack; assertion added in `test_the_forbidden_words_are_derived…`. A runtime rename through concatenation (`"mcp-" + "call"`) dodges it, as for every existing word; same bound as before, not new.
3. **Leaf harness**: executed; 24 leaves, `FILE_PACK_SURVIVORS` empty. The survivor in the delegate's table (no negatives line) is real and explained by F3. Leaves killed by the validator rather than behaviour (id/version/status suffix) count as kills by design.
4. **`test_rule_pack_schema.py`**: not weakened. It now compares directory contents to `REAL_PACKS | NON_SPEC_PACKS` and per-pack files to an explicit allowlist (and iterates `on_disk`, a strict improvement); an extra file or pack still fails. `len(REAL_PACKS) == 7` retained.
5. **`status: paper`**: nothing reads it; only the schema const (`rule-pack-v1.schema.json`) and `test_the_pack_file_is_marked_paper` for the seven spec packs (grep of `ccg/` and tests). Kept per the plan; the shell pack's header comment explains it.
6. **Own mutants** (`behaviour_failures`, with and without negatives; all killed unless noted): `call` a0→a1 (silent ARGUMENT, killed by positive case 0); drop `callKind` (case 2); `eval-call` a1→a0, drop `args`, a0 value `mcp-calll`, callee→`exec` (all killed); status dropped (killed, F4); `*mcp-call` glob (killed only by negatives, F3); reason→`shell-dynamic` (validator).

## What's solid
Real-extractor tests (a comment is a comment, `echo "mcp-call x"` is an `echo` call); the literal/dynamic split pinned both ways incl. value-sets; negatives per rule with loosened-rule mutants; the schema test's allowlist; honest pinning of the not-derivable case rather than a runtime special-case.

## Open questions
- Architect: quoted eval (any form, F2), reason code on CONSUMES (F2), flag-first tool (F1).

## Routing
Merge-ready for the pack as is; F1 gets a pinned test + architect question; F2 stays with the architect; F3 nit to `tdd-engineer`.
