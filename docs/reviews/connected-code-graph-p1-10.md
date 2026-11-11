# Review: P1-10 shell extractor (implementation)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1) · **Reviews:** `docs/plans/connected-code-graph-p1-extraction.md` P1-10

## Scope & verdict

Reviewed the uncommitted `cpg/extractors/shell/{__init__,extract}.py`, `README.md`, `cpg/tests/test_shell_extractor.py` and `cpg/tests/data/shell/` against the P1-10 step, `-bundle-spec3.md` (§4.1 rule 4, "Shell", §5.8, §5.9, §5.10, §7), `-p1.md` (D11, D12), `-p1-linking.md` P1-44 and the L0 test plan. The delegate wrote no REPORT.md, so its claims were treated as unverified. Executed vs read is marked per finding. Everything ran foreground under `ulimit -v 1000000/1500000` and `timeout -s KILL`, in scratch copies under `cpg/.ccg-work/gate-p1-10/` (removed). No tree-mutating git, Docker, network or FalkorDB.

**Verdict: needs changes.** One blocker: the variable-resolution cache lets a 40-line script build an exponentially large string, and the delegate's memory test passes without catching it. Everything else is conforming or a suggestion.

CPG: considered, not relevant — the code under review is a new, untracked extractor module and its fixtures; no loaded CPG models it and the question is its behaviour on hostile and real shell input, which I ran.

## Findings

### B1 (blocker, verified by execution) · an exponential string defeats every size cap and ends in `MemoryError`

`extract.py:552-563` (`_def_value`) caches a definition's value in `sc.cache` keyed by the definition only, while the chain-depth limit (`depth >= MAX_RESOLVE_DEPTH`, :553) applies only to a value computed fresh. A script where each use is followed by the next doubling definition therefore builds on the cached full value each time. `_assemble` (:217-237) merges adjacent consts into one string and checks only the *piece count* (`MAX_TEMPLATE_PARTS`, :234); the 8,192-char `MAX_CONST` is checked only later in `_screen` (:577), after the string exists.

Reproduced (`probe.py`, `ulimit -v 1000000`, 60 s KILL): `f(){ v0=x; v1="$v0$v0"; echo "$v1"; v2="$v1$v1"; echo "$v2"; … v39 …; }` (82 lines) raises `MemoryError` after 2.7 s. At 1 GB it died near 2^29; without a limit it would take the host's RAM before any exception. A script of this shape is untrusted repository content and is well under the 2 MiB file cap, so the README claims "variable chains 8 deep … one literal 8 192 characters" (README "Input is untrusted") are false for this path.

Why the delegate's test misses it: `test_a_hostile_script_stays_within_a_memory_bound` (`test_shell_extractor.py:598-605`) chains 400 definitions but has exactly one use (`use "$x399"`), at depth 0. The depth cut-off then fires on the first descent and nothing is cached at full depth. The test passes (58 passed, not skipped under ulimit) while the hazard is live. The `a=$a$a…` line in it hits the self-reference path, not this one.

Fix (the owner implements; one variant run by me): cap the assembled text length, not just the piece count, in `_assemble`. In a scratch copy I changed `if len(pieces) > MAX_TEMPLATE_PARTS:` to also fire when the summed length of string consts exceeds `MAX_CONST`. `fn.sh` then returns 81 records in 0.01 s, and the 30-script corpus keeps identical record counts (8,857 / 96 / 2,564). Whether the assertion that would catch a wrong fix exists is the point: the new test must be the doubling script with a use after each definition (above), run under `ulimit -v`, and must **fail on the unfixed code** (I just did: `MemoryError`). Also make the test fail rather than skip when no `ulimit` is set, or run the doubling case in a subprocess with `RLIMIT_AS` set by the test itself, so the gate cannot go green by skipping.

### M1 (minor, verified) · the value of a use depends on which uses came before it

Same cache as B1. `v0=lit; v1=$v0; … v11=$v10; echo "$v11"` gives `[{"unknown":"runtime-built"}]`; the identical chain with an `echo "$v_i"` after each definition gives `[{"const":"lit"}]` for the same final `echo "$v11"` (`t2/a.sh` vs `b.sh`, executed). The chain cap is then an accident of evaluation order, not a documented limit. When fixing B1, key the cache on the definition only and bound by size/steps, not by the depth of whichever use arrived first; add a test asserting the two scripts above agree.

### M2 (major for recall, plan-conformant; verified) · one tree-sitter error discards every function and call in the file

`extract.py:720-726`: any error/missing node gives `file` + `<script>` only. The plan's text says exactly this, so the code conforms, and the 30-script corpus is clean. But the wider real corpus is not: `git ls-files '*.sh'` is **113** files and **5** hit `parse-error`: `cpg/sandbox/offline-run.sh:430` (`(( 0$(stat …) & 02 ))`), `cpg/sandbox/tests/test_sandbox.sh:134`, `cpg/eval/headless/tests/test_launcher.sh:39`, `claude/teco/hooks/test-ensure-environment.sh:128`, `skills/joern-cpg/scripts/test-stamp-wiring.sh:351` (arithmetic command with `$()`, `<>/dev/tcp`, `${V:0:12}`, `<<<`). That is 4.4 % of scripts, including the sandbox runner, with all function symbols and call sites lost; calls into their functions then fall to `name-match` at best. Not a plan violation, but the plan's smoke number hides it and the README does not mention the rate. Suggested: `architect`/`teco` decide whether P1 accepts this (record the 5/113 figure in the plan and README) or extracts the functions and commands outside the error ranges, with the error lines as the warning detail. No code change is required to approve this unit.

### M3 (minor) · the local `SourceRefused` rationale is wrong, and the P1-44 mapping does not name it

- The delegate says the writer-rules tripwire forbids importing another extractor's internals. It does not: `tests/test_writer_rules.py:193` forbids only `ccg.valuehash` and the gated `bundle`/`atrest` names. `extractors/config/extract.py:27` imports `extractors.joern.normalize.core.SourceRefused` today. Two extractors in the same feature now disagree.
- `-p1-linking.md` P1-44 (line ~195) lists `normalize.core.SourceRefused` only. The shell class is a different class, so it falls under "any other exception → extract:failed": the code is right, but the plan does not say so and the P1-44 "one planted exception per row" test will not cover it. The delegate's claim that the plan already says both classes map is not borne out.
- Suggested: one class. Either re-export `normalize.core.SourceRefused` (as config does) or move it to `extractors/common/` and have the three modules import it; if the local class stays, add a P1-44 row and test for it. The 2 MiB cap (`MAX_FILE_BYTES`, the largest real script is 171 kB per the code comment) is a reasonable choice the plan never specified; it should be written into P1-10.

### M4 (minor, ruling needed) · secret withholding goes beyond the spec and the Python/TS extractors

Verified: neither `normalize/` nor spec3 withholds `valueSet`/`constant` literals; `-bundle-spec2.md` F-1 says the bundle "may hold source literals in clear". Shell withholds on `screen_fails`, name-rule variables, secret-named flags and `--flag=value`, and over 8,192 chars (`:565-590`, `:660-663`). It is conservative and consistent with FR-15's direction, and the valueSet is a legal `unknown` (§5.8). Measured on the 30-script corpus by differencing the output with the masks disabled: **7 of 5,118** valueSet/constant records change (the delegate said 3 of ~4,900 values; the counts depend on the definition, order of magnitude agrees; 4,930 valueSets confirmed). The changed values are `99999999999999`, a `pass=0` counter (name rule on `pass`) and a template using it. **No Cypher literal is withheld**: the `gquery` test passes and its 50 values are intact. So AC-3 role lifting loses nothing here. Cost is only that variables like `SALESPERSON_…` are masked by the name rule's substring list. Ask `architect` to ratify in the plan (or drop for parity); do not change code for this review.

### M5 (minor) · `<dynamic>` and `shell-dynamic` for a computed command name is a plan extension

Verified in code (`:636-639`, `:655`) and a probe: `$DIR/run.sh a` gives `<dynamic>`, `possible/shell-dynamic`. The plan and spec3 §7.3 list `eval`, indirect expansion and computed `source` only. A computed command name is a sensible fourth case (the spec's `calleeName` rule already defines `<dynamic>`); add it to the §7.3 `shell-dynamic` row so P1-44 and the linker read the same list. Also: `a0` for such a call is the first *argument*, not the command, which is correct and consistent.

### Rulings on the delegate's points

- **(a) `constant` for shell:** accept. §5.9 defines a constant as a module-level variable assigned once whose value is a value set; a script-level `NAME=…` is the shell analog, `qname` is `path::NAME` (last segment the variable name, §4 grammar), the record is a bundle-only record, and `ref` is used exactly as §5.8 defines it. The code's exclusions are sound: counted across all scopes, conditional, appended, function-named and name-rule-failing variables are never constants (`:437-446`); the ref is read only inside functions. Checked on `bootstrap_schema.sh`: `HOST` is read as `{"ref": "…::HOST"}` and the test asserts it, and `PORT=6379` is a constant. A script that assigns a variable twice is correctly not a constant (`t1/a.sh`: `HOST` set twice, read as `unresolved-ref`). Add one sentence to spec3 "Shell" so the next reader does not have to re-derive this.
- **(c) no `cond`/`ret`, `complexity` 1:** accept; spec4 §2 is `1 + own cond records`, README states it. Re-open only if a later AC needs shell branches.
- **(d) cap and error mapping:** see M3.
- **(e) unparseable file keeps `file` + `<script>` and one warning:** conforms to the plan text; see M2 for the real-corpus rate.

## Checks run

- **Tests (executed).** `tests/test_shell_extractor.py`: 58 passed, none skipped (under `ulimit -v 1500000`). With `tests/test_writer_rules.py`: 324 passed, which includes the AST tripwire over `extractors/**` (it passes, and the shell module contains no `gate_argv`/`valuehash` reference, pinned by `test_the_module_imports_no_bundle_rule_deciding_name`). `audit_records` is vacuous here: the module writes no `configEntry`/`launchFact` (read, :1-34, and the sweep `grep` of the module for those types).
- **Corpus smoke (executed, two definitions).** Glob (30 files, `falkor-chat/scripts` + `cypher-mcp`): 8,857 records, 96 symbols, 2,564 call sites, 188 constants, 91 params, 958 data, 0 warnings, validates as one bundle: matches the delegate. `git ls-files '*.sh'`: 113 files, 19,693 records, 344 symbols, 5,729 call sites, **5 parse-error warnings** (M2). `bootstrap_schema.sh`: 50 `gquery` call sites, all `resolved` to `…::gquery`, valueSets are Cypher consts or a template with the dimension (test asserts and passes).
- **Hostile input (executed).** 10,000 nested `$(`: `too-deep` warning, 0.04 s. 500 nested `if`: `too-deep`. 2 MiB + 1 byte: `SourceRefused`; exactly 2 MiB: accepted. FIFO, symlink to `/etc/passwd`, non-UTF-8: `SourceRefused`, messages carry the file name and byte offset but no content. 40,000 commands: `record-cap` warning, 44,442 records, 1.2 s. 40 conditional `x=$x…` definitions: bounded. Doubling template through a cached definition: **MemoryError (B1)**. A search of the module finds only one `open()` (`:147`, after confinement and `is_file`) and no `subprocess`, `eval`, `exec`, `os.system` or `popen`. Race between `resolve()` and `open()` (a symlink swapped in between) is not closed; low risk for a staged unit, named only.
- **Secrets (executed).** `TOKEN=…`, `DB_PASSWORD=…`, `--password x`, `--secret-key=zzzz`, `--api-key <long>`: all withheld as `unknown`; `https://x` kept.
- **Mutants (executed).** Delegate's batch files list 37 results (not 44); one line, `M24-regular-file-check-off SURVIVED`, is stale: I re-ran it and it is killed by the FIFO test. Re-ran 6 of the delegate's: M13, M24, M25, M27, M33, M43 all killed; M37 and M43 are killed only by the corpus smoke test, as claimed (M43 re-run: `test_the_real_scripts_parse…`). Four-plus of my own, in fresh copies (`.ccg-work/gate-p1-10/gm.py`), 11 in all, **all killed**, including silent argument mutations: G1 (flow path index taken from the argument position `i` instead of the param), G2 (`def` id fixed to param 0), G5 (value-set id always `a0`), G6 (callsite ordinal always 0), G8 (`use`/`def` swapped), G9 (constant line +1), G11 (substitution flow `fromPath` = `p0`). The delegate's tests are strong on identity and index arguments; the gap is the resource bound (B1), where the mutant that matters (removing the length cap) has no test because the cap does not exist.
- **Not modelled (read).** README lists `shift`, `set --`, `getopts`, `$#`, arrays, `declare -n`, `export -f`, `bash -c`, sourced libraries, cross-file calls, and says the subshell/pipeline/`&&` assignment is "added" (over-approximated as a conditional definition), `+=` likewise. Matches the code (`_CONDITIONAL`, :69).

## What's solid

- Interface and record shapes match the plan and the spec: one `<script>` per file (`:311`), functions always top level with `#N` redefinitions, params `$1..$N` index `N-1` and `$@` at index N, `${v:-d}` as a value alternative and never a `bindValue` (W3, no `configEntry`/`launchFact` written), `possible/shell-dynamic` for `eval`, `${!n}`, computed `source`.
- Id-bearing and index arguments are well pinned by tests (all 11 of my mutants on them died).
- Confinement is correct: name checks, `resolve()` + `is_relative_to`, `is_file`, bounded read, UTF-8 refusal (D12), no process or exec surface, explicit-stack walks, parse timeout via the read callback.
- The README states the limits plainly.

## Open questions

1. `architect`: accept the 5/113 parse-error loss (M2), and ratify the withholding rule (M4) and the 2 MiB cap and `<dynamic>`/`shell-dynamic` extension (M3, M5) in the plan/spec?
2. `architect`/`teco`: one shared `SourceRefused` or an explicit P1-44 row for the shell class (M3)?

## Routing

- **B1 + M1 → `tdd-engineer`/`coder`** (same delegate): the doubling-script test first (fails today with `MemoryError`), then the length cap and depth-independent cache; re-gate by `analyst` (re-run `fn.sh`, the corpus counts and a mutant removing the new cap).
- **M2, M4, M5 → `architect`** (plan/spec text; M2 may spawn a follow-up unit). **M3 → `architect` + P1-44's owner**, plus the shell/config extractors' shared-class edit.
- Approval of the unit's other behaviour stands once B1 is closed.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, "Always check". Source: B1 (and the delegate's own memory test, `test_shell_extractor.py:598`, which passed over it).

- A size or depth cap on untrusted input is checked against the **cache and merge paths**, not only the first descent: plant a case where each step reuses the previous step's cached result (a doubling chain with a use after every step) and run it under `ulimit -v`. A cap that holds only when each value is computed once, or a memory test that skips without `ulimit`, is a bound, not proof.

## Pass 2

Re-gate of the coder's fix for B1, M1 and M3. Same caps (`ulimit -v`, `timeout -s KILL`), foreground, scratch copies under `cpg/.ccg-work/gate-p1-10/` (removed). The coder's PROGRESS.md notes were read; its claims were re-run, not trusted.

**Verdict: approve with suggestions.** No open blocker.

- **B1: fixed.** `extract.py:230` now also fires the cap when merged const length exceeds `MAX_CONST`. My 40-step doubling script (a use after each step), re-run under `ulimit -v 1000000`: 81 records in 0.02 s (was `MemoryError`). Corpus counts unchanged: 30-file glob 8,857 records / 96 symbols / 2,564 call sites / 0 warnings; `git ls-files '*.sh'` 113 files / 19,693 records / 344 symbols / 5,729 call sites / 5 parse-error warnings. The new test (`test_a_doubling_chain_with_a_use_after_every_step_…`) runs in a child under `RLIMIT_AS` and fails by assertion, not skip: against a scratch copy of the module with the length cap and the depth key reverted it fails (3 failed: the doubling test, the length-cap test, the order-independence test), and passes on the shipped code. README "chains 8 deep" and "one literal 8,192 characters" now hold on the cache and merge paths (the cap is checked at merge, the cache key carries the depth).
- **M1: fixed.** The cache key is `(definition, chain depth)` (`:553`). My `a.sh`/`b.sh` pair (same final `echo "$v11"`, with and without earlier uses) now both give `runtime-built`; a test pins it. Residual: a deep chain that used to resolve when a prior use cached it now consistently truncates at depth 8: consistent, documented.
- **M3: fixed.** `extract.py:53` imports `SourceRefused` from `extractors.joern.normalize.core`; `extractors/common/errors.py` now exists and `core` re-exports it. Executed: `shell.SourceRefused is core.SourceRefused` and `is errors.SourceRefused` are both `True`. The import works either way. Whether P1-44's mapping row covers it is P1-44's: the one class is the one it already names.
- **M2, M4, M5: rulings are in the plan** (P1-10 "Rulings", `-p1-extraction.md:378-380`, read). Still open and small: `cpg/extractors/shell/README.md` does not yet record the 5/113 parse-error rate or the five paths (grep finds neither), which the ruling says it will. Coder owes that one README edit; no re-gate needed for it.
- **Tests (executed):** `tests/test_shell_extractor.py` 61 passed, none skipped. I did not re-run `test_writer_rules.py` (under concurrent edit, as agreed).
- **Mutants (executed, fresh copies, all killed):** H1 cache depth ignored (killed by the order-independence test); H2 length cap removed (killed by the length-cap test, and the doubling test fails on the same revert, above); H3 assignment-in-command check disabled (killed by the read-before-assign test); H4 silent argument mutation, `$N` flows to param `N-2` instead of `N-1` (killed by the positional-flow test).

Routing: coder adds the README line (parse-error rate and the five paths); nothing else open on P1-10. `teco` may close the gate once that line lands. The proposed `REVIEW.md` line in Pass 1 still stands for `teco` to apply at feature close.
