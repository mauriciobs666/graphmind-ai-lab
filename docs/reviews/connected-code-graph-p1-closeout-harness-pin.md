# Review: P1 close-out harness pin round and the Q8 withheld-shape predicate

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1 close-out)

## Scope & verdict

Reviewed, as committed and executed against a `git archive` of HEAD (`8599a8f9`) with a throwaway `.git` whose object store aliases the repo's, so the `EVAL_COMMIT` dev tests ran instead of skipping:

- (a) the harness pin round `425c2c0c`: `cpg/eval/tools/gate.py` (`is_dev_rehearsal`, `confirmed_key_errors`, `dev_probe_check`, `write_dev_probe`), `cpg/eval/decisions/ccg-dev-key-errors-v1.json`, `cpg/tests/test_gate_dev_probe.py`, `test_gate_dev_probe_writer.py`, `gate_helpers.py`;
- (b) the Q8 predicate `cpg/eval/tools/withheld_shapes.py` (`48844873`) with `cpg/tests/test_withheld_shapes.py`.

Baselines: `docs/reviews/connected-code-graph-p1-32b-impl.md` (Pass 7: P7-1, P7-2, stopping condition), `docs/plans/connected-code-graph-p1-gate-marker-ml.md`, `docs/plans/connected-code-graph-p1-closeout-ml.md`, `docs/reviews/connected-code-graph-p1-closeout-ml.md` `## Q8 ruling`, `docs/plans/connected-code-graph-p1-closeout.md` §3.5. No held-out data and no custodian directory read; only builder-visible dev gold and the repo's own files as corpus, counts reported.

**Verdict: needs changes.** (a) alone is approve with suggestions: P7-1 and P7-2 are fixed, one major (F4) and two minors. (b) is the blocker: F1 (the predicate is neither pinned nor wired, and the dev probe will refuse to start with it as it stands), plus F2, F3 and F5 on rule coverage and tests. Nothing here is wrong in the shipped state of an unused module; it is wrong to freeze on it.

CPG: considered, not relevant — the review is of harness scripts and a 150-line pure function whose evidence is execution on the tree and dev gold, and no loaded `cpg_*` graph covers `cpg/eval/tools`.

Executed: `tests/test_gate*.py` + `tests/test_withheld_shapes.py` in the archive, **902 passed, 15 deselected**, no skips; 20 mutants of `gate.py` and 45 of `withheld_shapes.py` (Appendix A); 19 `is_dev_rehearsal` probes; ~85 predicate decoys; three corpus differentials (shell, Markdown, dev gold).

## Findings

### F1 (blocker for the freeze; executed). The predicate is unpinned and unwired, and the dev probe refuses once it is applied

Two facts, one cause (the module has no consumer).

1. `G.HARNESS_FILES` is the import closure of `gate.py`/`gate_run.py`; executed, it lists 10 `.py` files and `EVAL_COMMIT` and not `withheld_shapes.py` (`grep -rn withheld_shapes cpg` finds only its own test). The ledger says so (`docs/plans/connected-code-graph-coordination.md` l.1235, "wire before the prereg record"). Consequence: the custodian's one pre-ask run of the predicate on the held-out list is code that no sha in `ccg-prereg-v1`, the marker or the probe covers. A rule edit after the freeze is invisible, which is the adaptation the pin exists to expose.
2. Nothing in `gate.py` applies the predicate, so `dev_probe_check` still demands "every dev item of a gated stratum is a hit or a confirmed key-error". Executed on the real dev gold with the 10 shape items non-hit and the 3 key-errors recorded: refused, `scripts has 36 hits of 49 dev items (3 stakeholder-confirmed key-errors)`. Plan §3.5 says a withheld shape "means the criterion is not met for those items", so by the plan's own text the rehearsal exits 1 and the gate cannot start. The text and the code agree; the design has no route.

Needs a ruling and a build, in this order: (i) `data-scientist` rules whether predicate-withheld dev items are excluded from the hit-or-key-error requirement, reported withheld with counts (my reading of the closeout-ml Q8 effect paragraph and the withheld-share cap); (ii) in a round before the wiring, add `withheld_shapes.py` to the expected names in `tests/test_gate.py::test_m3_the_real_harness_pin_covers_...` (red); (iii) the wiring round satisfies it by importing the module where the held-out list is withheld (`gate_run.withhold_strata`'s neighbour) and in the probe, so the closure picks it up, and adds the cap and the withheld-count report; (iv) regenerate the rehearsal probe, since `harnessSha256` changes. Putting the assertion in the wiring round itself would let the declared set contain the module from birth.

### F2 (major; executed). J, M and I are narrower than their families

The ruling withholds "a `-m` package launch", "string-embedded code" and "`bash -c`"; the regexes cover the dev spellings. Real corpus (167 `.sh` files, comment lines skipped): **28 of 35** `python … -m <pkg>` lines are J; the 7 misses are `"$VENV/bin/python" -m pip …` (5, the path ends in `python"`) and `"$PY" -I -m ccg.broker.server` (a flag before `-m`). Decoys: `"$HERE/.venv/bin/python3" -m x`, `"${PY[@]}" -m x`, `"$PY" -u -m x` all `None`; M: `"$PY" <<EOF` (no `-`) and `"$PY" -c '…'` `None`; I: `bash -lc '…'`, `sh -ec '…'`, `"$SHELL" -c '…'` `None`. The engine builds none of these, so each held-out analogue is a certain miss on the one-shot list, not withheld. The J patterns equal the two dev lines (`exec "$PY" -m …`), which is a fit, not a rule.

Fix: do not add these spellings as a list; vary the axes in a parametrised test with a twin per axis: launcher word (literal, versioned, `$V`, `${V}`, `"${V[@]}"`, quoted path ending in an interpreter, `env`/`exec`/`uv run` prefix), flags between interpreter and `-m`/`-c` (`-u -I -X x`, combined `-lc`/`-ec`), and the opener (`<<`, `<<-`, `-`, `-c`). Expected values from an independent probe, not the regex: the corpus differential in Appendix B (`shdiff.py`/`jprobe.py`) is that probe. Keep the false-positive twins (`grep -m 1`, `git commit -m`, `"$X" -m name` is currently J and should be reconsidered).

### F3 (major; executed). The Markdown fence and front-matter grammar differs from the extractor's, both ways

The contract is "a fence is not a call" as the extractor ships it (`extractors/markdown/extract.py` `_fence_run`, `_front_matter_end`). The predicate re-implements a different grammar. Over the 1162 tracked `.md` files, outside front matter:
- engine-fenced, predicate-not: **22 lines in 6 files**, container-nested or deeper-indented fences (`AGENTS.md:248`, a 4-space fence in a list; `- ```bash`; `> ```bash`). A held-out item in such a block is a certain miss, unwithheld.
- predicate-fenced, engine-not: **589 lines in 2 files**: a line opening with ```` ```` ```` followed by inline code (`falkor-chat/docs/reviews/k027-parse-robustness.md:287`) opens a fence in the predicate (CommonMark and the extractor refuse a backtick info string containing a backtick) and swallows 492 lines to EOF; withholds items the engine builds.
- Front matter: 0 disagreements on the corpus, but decoys differ: a leading `---`, prose, `---` (a horizontal rule, no YAML first line) is `C` in the predicate and prose in the extractor; a 250-line block is `C`, the extractor's `FRONT_MATTER_MAX_LINES = 200` makes it prose.

Fix: re-state the same three rules (container prefixes, info-string rule, YAML-first-line and 200-line cap) in the predicate, still importing nothing; the differential in Appendix B (`mddiff.py`) becomes a test over the repo's `.md` corpus that compares the two line sets and expects 0/0. The independence the ruling asks for is about not reading engine output; mirroring the shipped contract text is what "build-invariant" means for B and C.

### F4 (major; executed). The key-error record can be widened after the freeze

`confirmed_key_errors` demands working copy equals HEAD, which stops an uncommitted edit (`test_a_confirmation_record_edited_after_its_commit_is_not_read`, mutant M11 killed). It does not tie the record to the record commit. `run_context` exempts everything under `cpg/eval/` from its "changed since the record" refusal, and the decisions file is not in `HARNESS_FILES`. Executed (`late_ke.py`, Appendix B): freeze at commit R with `confirmed: []`; a later commit adds `{"itemId": <a miss>, "confirmedBy": "user", "date": "2026-11-24"}`; `dev_probe_check(probe_for_R, repo, R)` returns without refusal. So yes, the record can be forged by a plain commit, and `confirmedBy == "user"` is a self-asserted string: the validation is a format guard (real date, distinct ids, the one allowed word), not provenance. The three entries themselves are sound (below).

Fix: read the record's bytes at `record_commit` (`git show <rc>:cpg/eval/decisions/ccg-dev-key-errors-v1.json`) and require them equal to HEAD's and the working copy's; the red test is the scenario above. Failing that, put the file's sha256 in `ccg-prereg-v1`. Say in the docstring that `confirmedBy` is not a signature; the control is the commit and the pin. I ran the scenario (red now), not the fix: the implementer owns the choice of `rc` blob versus prereg sha.

### F5 (major; mutation). The predicate's tests leave 23 of 45 mutants alive

Run with `python -B` and `__pycache__` cleared between mutants (Appendix A; without that, same-size edits within one second reuse stale bytecode and mislead in both directions). Survivors that are not equivalent:
- the module's own "judged on EVERY gold line" invariant is pinned for B and J but not C, M, I, K: `all`→`any` survives for each (W10, W12, W13, W15), confirmed by hand with `sed`;
- fence grammar edges: indent bound 0-2/0-4 (W1, W2), two-backtick opener (W3: a line starting with an inline double-backtick span would be a fence), closer of the other char (W5), closer with trailing text (W6);
- heredoc terminator with `<<-` indentation (W17), `Popen` (W25), `zsh` (W37), `-c` spacing (W38), `\r`-terminated front-matter fence lines (W39, W40), `.markdown` and upper-case extensions (W42, W43).
Equivalent or benign: W9, W28 (lines[-1] is empty after a trailing newline), W31, W45 (`SHAPES` is unused: nit).
Each needs a test whose expected value is written by hand (the dev-count tests derive nothing from the rules, which is good); the `any` four are one parametrised test over the four families with a two-line gold, one plain.

### F6 (minor; executed). Over-withholding edges

Decoys: `python3 - <<< "abc"` and `python3 -c 'print(1<<3)'` match `_HEREDOC` (`<<3`, `<<< "abc"`), whose body never terminates, so every later line to EOF is `M` including a plain `"$HERE/x.sh"` the engine builds; `# bash -c '` in a comment opens an `I` region; `bash -c 'a' ; "$HERE/x.sh"` withholds the whole line though the launch is outside the string. The quote-parity scan marks 12 lines of `cpg/eval/headless/tests/test_guard_paths.sh` (165-235) including a python heredoc opener at :173. Corpus check that this does not matter today: of 732 withheld shell lines (M 594, I 108, J 30), 10 carry a call site with a dynamic or path callee, all `"$PYTHON" -m venv` or `/usr/bin/python3` (interpreters, not script launches), so no script launch is over-withheld on this tree. Fix with the tests, not the heuristic: a `<<<`/arithmetic twin and a comment twin.

### F7 (minor; mutation). The P7-1 exemption test covers one stratum

`is_dev_rehearsal` iterates `STRATA` (5). Restricting it to `STRATA[:1]` survives all 902 tests; the dev rehearsal list (`11_dev_probe.sh`, `dev-all.jsonl`) spans strata, and `[1:]` or `('scripts',)` are killed only by incidental tests. Add one test: a list holding a committed record of each stratum is exempt; replace one with a content twin and it is not.

### F8 (minor; inferred for the engine, executed for the predicate). K is defined by what the extractor can read

K withholds `subprocess.run([...])` when either of the first two list elements is a name, attribute or `str(name)`. Executed: `subprocess.run(["bash","scripts/x.sh"])` is `None` (not withheld), `subprocess.run([sys.executable,"x.py"])` and `subprocess.run(["echo", str(P)])` are `K`. I found no pack or extractor in `cpg/packs`, `cpg/extractors`, `cpg/ccg` that reads `subprocess` argv at all, so the literal twin is probably also unbuilt, which would make the literal/indirect cut the "product chooses the cut" pattern the ruling rejects for G/H. I did not run the Python path of the engine (Joern); the plan's `-p1-linking.md` owner should state whether a literal argv is built. One dev item, so the dev count cannot tell.

### F9 (nits)

- `tests/test_gate_dev_probe.py:151` asserts `"rehearsalId" not in probe(e)` where `probe` is the test helper's own document; it cannot fail from the product. The real pin is `test_gate_dev_probe_writer.py:53`.
- `write_dev_probe` still format-checks `run["listKey"]` (64 hex) and then ignores it; harmless, say so or drop it.
- The real-record test pins `ids <= dev` but not the three ids; acceptable (the ids are a stakeholder fact), but then the file's sha belongs in the prereg (F4).

## Answers to the questions

1. **P7-1: fixed.** `is_dev_rehearsal` is True only when every record's `_mk.canon` (sorted-key compact JSON) is in the set of committed dev canons. Executed on the real gold: all dev records True; one record True; reversed key order True; duplicates True; tuple-for-list `lines` True; escaped-unicode re-encoding True (same value). False: empty list, an extra key, a dev `itemId` with changed `lines`, with another `impacted.path`, a held-out-shaped record reusing a dev id, a mixed list (one honest, one twin), a trailing space in a string, `1.0` for `1`, a non-mapping, `None`, NaN, a cycle, `bytes`, a 5000-deep nest (returns False, no raise), reordered `lines`. The working tree is required clean before the marker (`pf["firstContact"]`), so the dev gold read from it is the committed gold. Test gap: F7. Mutants M1-M4, M20 (killed M1-M4; M20 survives only as F7).
2. **P7-2: sound.** The id was `run["listKey"]`, compared with nothing; the file still binds record commit, harness map, gold hashes and the build's snapshot id. Repo sweep for `rehearsalId`: only two tests asserting its absence, plus review and ledger prose. The new docstring states the residual (a hand-composed `run.json` on a store holding that snapshot yields a probe), which is what Pass 7 recorded as acceptable for the stated threat.
3. **Key-error record.** The three ids are real dev items, all `derivedStratum: scripts`: `f7a509…` is `skills/joern-cpg/scripts/pipeline.sh` impacting `test-stamp-wiring.sh` lines 142 (`awk … "$HERE/pipeline.sh"`) and 462 (`rq_scan "$HERE/pipeline.sh"`, whose body greps its input): both gold lines are reads, so the confirmation covers the whole item; `f33cbe…` is `mcp-monitor/run.sh` impacting `demo_falkor_chat.sh:117` (`"$ROOT/run.sh"`); `e8c36e…` is `cypher-mcp/server.py` (`::<module>`) impacting `cypher-mcp/run.sh:31`. Entry criterion executed over the real dev gold for http+mcp+scripts: every item hit or key-error accepted; any one item removed refused; the three ids with a hit mislabelled as key-error are rejected by mutant M14/M12. Validation is enough for a malformed or agent-typed file, not for a deliberate commit: F4.
4. **Predicate.** Pure: stdlib only (`ast`, `re`, `typing`), no I/O, inputs unmutated, the same output under four `PYTHONHASHSEED` values and under an import hook that bans `ccg`, `engine`, `gate`, `match_keys`; `locator["path"]` is ignored. Dev counts: **J 2, K 1, M 1, C 2, B 2, I 2 = 10 of 49**, on the items `9180d6`/`d65e21` (`model-bench/run.sh:19`, `mcp-monitor/run.sh:19`), `a7b52a` (`seed_eval_corpus.py:599-600`), `aa752d` (`demo_falkor_chat.sh:145-146`), `106e87`/`b0f627` (`tdd-engineer.md`, `tico.md` front matter), `c649a7`/`d7b414` (`SKILL.md:206`, `AGENTS.md:80`), `0fbf43`/`d105f0` (`test-stamp-wiring.sh`); the other 39, including G/H 1-segment and the awk item, are `None`. Independence from the extractor builder: the ledger names the coder as author (same git author, so the repo cannot confirm it). Rule decoys and the withholds-what-the-engine-builds / fails-to-withhold answers are F2, F3, F6, F8; on the dev gold there is no built item withheld and no unbuilt shape missed.
5. **Pin: a gap, and what to do** is F1; the freeze/prereg record must carry the predicate's sha (`88af5c26…` today), which only means something once the module is in the closure, and any later rule change (F2, F3, F5 will change it) moves that sha and the gate's own, so wire first and regenerate once.

## What's solid

- The `gate.py` guards are well tested: 17 of 20 mutants die (M1-M4, M6-M14, M16-M19; M5 and M15 are equivalent because `_dev_gold` already maps `ValueError` and the `TypeError` branch refuses anyway; M20 is F7), including the fresh key-error validation (`confirmedBy`, date regex, `fromisoformat`, distinct ids) and the writer's hit-wins rule.
- The dev-count test derives its expected counts from the ruling's letters and checks which items, not only how many; the purity test runs the module under a banned-import hook rather than reading its own import list.
- `gate.py` sha `03c0f814…` and `withheld_shapes.py` sha `88af5c26…` equal the ledger's.

## Open questions

1. (F1) Are predicate-withheld dev items excused in the entry criterion, with counts, or does the plan's "not met" stand and require a stakeholder waiver? `data-scientist` / stakeholder.
2. (F8) Does the engine build a literal-argv `subprocess` launch of a script? Engine owner.

## Stopping condition

This review closes when all five hold, each checkable by the command named: (1) `python3 -B -c "import gate; assert 'cpg/eval/tools/withheld_shapes.py' in gate.HARNESS_FILES"` from `cpg/eval/tools`, with the assertion landed red in an earlier round; (2) `11_dev_probe.sh` exits 0 on the real dev gold with the withheld counts printed and the 10 shape items excused only through the predicate; (3) the F4 scenario refuses; (4) the Appendix B differentials read 0 for `python -m` lines not withheld (7 today), 0 engine-fenced-not-predicate lines (22) and 0 predicate-fenced-engine-not lines (589), with the false-positive twins kept; (5) mutants W10, W12, W13, W15, W1-W3, W5, W6, W17, W25 killed. A next pass is owed only if the predicate's rules change (they will, for F2 and F3: a pinned-file diff) or the wiring adds a guard; a pass limited to `gate.py` edits that touch no check, or to tests, ends the series. The Pass 7 condition ("P7-1/P7-2 do not reopen the series unless a new guard") is met for (a): the diff tightens `confirmed_key_errors` and `is_dev_rehearsal`, adds no new guard class.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: F2, F3 (this review).

- A predicate that stands in for another component's grammar (a fence, front matter, a launcher spelling) is run over the real corpus against that component's own parser, differences counted per direction (withheld but built, built-looking but not withheld); a rule that matches only the dev spellings is a fit, not the rule.

## Appendix A: mutants

`gate.py` (in place in the archive, restored after each, runs 2-5 s apart, `tests/test_gate_dev_probe.py`, `_writer.py`, `test_gate.py`): killed M1 itemId-only exemption, M2 `all`→`any`, M3 empty list exempt, M4 unsorted canon, M6/M7 confirmer check, M8 date regex, M9 `fromisoformat`, M10 distinct ids, M11 HEAD equality (re-run at the right site: the first `str.replace` hit another `!= text` in `find_prereg_line`, which made the first M11 result meaningless), M12 key-error subset, M13 every-item rule, M14 key-error even when hit, M16 schema, M17 gold sha, M18 record commit, M19 non-dev ids. Survived: M5, M15 (equivalent), M20 `STRATA[:1]` (F7).

`withheld_shapes.py` survivors (23 of 45, `-B`, `__pycache__` cleared): W1, W2, W3, W5, W6, W7, W9, W10, W12, W13, W15, W17, W18, W25, W28, W31, W37, W38, W39, W40, W42, W43, W45; the first run, without that care, showed W12 killed and W14 alive, both wrong.

## Appendix B: probes

Scripts (scratchpad, read-only inputs): `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/` — `p71.py` (is_dev_rehearsal), `probe_probe.py` (entry criterion with the 10 shapes), `late_ke.py` (F4), `dev_run.py` (dev counts per item), `decoys.py` (~85 predicate decoys), `shdiff.py`, `jprobe.py`, `mprobe.py` (shell corpus), `mddiff.py` (Markdown corpus), `mut_gate.py`, `mut_w.py`. Run with `cpg/.venv/bin/python -B` from the archive.

## Pass 2

Scope: HEAD `88f0ef08` (predicate `c6edc26d`, sha256 `152f1285…`; harness `88f0ef08`, `gate.py` `034f05a8…`, `gate_run.py` `2d0db67b…`; all three shas match). Executed on a fresh `git archive` with a `.git` aliasing the repo objects. `tests/test_gate*.py` + `tests/test_withheld_shapes.py`: **1196 passed, 15 deselected**. No held-out or custodian data read.

**Verdict: approve with suggestions.** No blocker, no major. Teco's ruling (withheld dev items excused, counts beside them) is implemented without a hole I could open. CPG: considered, not relevant (same reason as Pass 1).

### Dispositions

- **F1 fixed.** `gate.HARNESS_FILES` now has 12 entries including `cpg/eval/tools/withheld_shapes.py` (executed). Replacing the import in `gate.py` with a stub turns `test_m3_the_real_harness_pin_…` red. Process note: the assertion landed in the same commit (`88f0ef08`) as the wiring, not a round before, so red-first is shown by that mutant, not by history. Dev probe: on the real dev gold (http 18/0 withheld, mcp 20/0, scripts 49 with exactly 10 excused: `0fbf43 106e87 9180d6 a7b52a aa752d b0f627 c649a7 d105f0 d65e21 d7b414`; the G/H items and the three key-errors are not excused) 36 hits + 3 key-errors + 10 withheld is **accepted**; one hit-able scripts item missing, one http miss, a withheld id listed as a key-error: each **refused** with the counts. The excuse set is computed in `dev_probe_check` from the committed gold and `git show <EVAL_COMMIT>:<path>` text, never from the probe file; a non-scripts stratum is never excused (mutant "any stratum" killed). The inputs a forger would need are the committed gold (hash-pinned in the probe), `EVAL_COMMIT` (in `HARNESS_FILES`, so an edit is a harness diff), and the predicate (pinned); an unreadable `EVAL_COMMIT` withholds nothing. Residual: the excuse is exactly as wide as the predicate, so a false-positive rule excuses a real miss. On dev that is bounded by the 10 ids above; for held-out, F2/F3's differentials are the guard.
- **F2 fixed.** Corpus `python … -m` lines withheld: 35 of 35 (was 28); the Pass 1 decoys `"$VENV/bin/python" -m`, `"${PY[@]}" -m`, `"$PY" -u -m`, `"$PY" <<EOF`, `bash -lc`, `sh -ec`, `"$SHELL" -c` are now J/M/I; the false-positive twins (`grep -m`, `git commit -m`, `mkdir -m`, `"$X" -m name`, `python3 script.py -m opt`) are `None`. Shell over-withholding differential: 45 withheld shell lines carry a dynamic or path callee, all interpreters (`"$PYTHON" -m venv`, `/usr/bin/python3 -I -c`), none a script launch. Residual (minor): a multi-line `bash -c "…\" …"` with an escaped quote and the gold inside the string is `None` (parity heuristic).
- **F3 fixed.** Markdown differential over 1163 files: engine-fenced-not-predicate 0 (was 22), predicate-fenced-engine-not 0 (was 589), front-matter end disagreements 0; the HR-led, 250-line and backtick-info decoys now agree with the extractor.
- **F4 fixed.** `confirmed_key_errors(..., record_commit)` requires the bytes at the record commit equal the working copy. Executed end to end (helper repo): record present at F and used at F **accepted**; widened after F, widened with the probe listing only the old id, narrowed after F, and a nonexistent `record_commit` each **refused**. Mutants removing the `record_commit` read and the writer's pass-through are killed. Residual (minor): `record_commit=None` skips the check and `dev_probe_check(doc_without_recordCommit, repo, None)` is accepted (executed); production cannot reach it (`run_context` refuses a missing record commit first, `score` skips the probe check), but the default fails open: make `None` a refusal.
- **F5 fixed in substance.** My fresh batch, 54 applicable mutants of the new `withheld_shapes.py` (`-B`, `__pycache__` cleared): 50 killed, including every Pass 1 survivor (`any` for C/M/I/J/K/B, fence bounds, closers, `<<-`, `Popen`, `.markdown`, case). Survivors: `ksh` in the shell-word pattern, the lower bound `g < 1` (benign, as before), CRLF handling in `_fence_run` and in `_shell_embedded` (two untested paths, minor).
- **F6 fixed.** `<<<` and `1<<3` after `python` no longer open a heredoc, a `# bash -c '` comment opens nothing, and `bash -c 'a' ; "$HERE/x.sh"` is not `I` (executed decoys).
- **F7 fixed.** The `STRATA[:1]` mutant of `is_dev_rehearsal` now fails 5 tests (executed on `test_gate_dev_probe*.py`).
- **F9 closed in part.** The `rehearsalId` test now plants `rehearsalId="not hex"` into a probe and expects acceptance, so it goes through the product. `write_dev_probe` still format-checks `run["listKey"]` (`gate.py:1471`) and ignores it otherwise; harmless nit.

### The cap and the P7-1 interaction

`check_withheld_cap` refuses `withheld > 0.25 × scriptsCounting`: executed (19, 79) pass, (20, 79) refuse, (0, 0) pass, (12, 49) pass, (13, 49) refuse. In `_run_gate` it sits after `score_inputs_check` and `withhold_strata` and before `lock.take`, the build and the marker; `PrecheckRefusal` maps to `EXIT_RESOLVE == 5`; the test asserts no marker and no build call. Mutants `>=`, `0.30`, cap skipped, record-commit read off, no excuse, any-stratum excuse, writer without record commit: all killed. **Gap (minor):** moving the cap below `lock.take` is not killed (executed: 553 passed), because the tests run with `lock=None`; the "before the lock" half of the claim rests on reading. Plant a lock double that records `take` and assert it was not called on a cap refusal.

Interaction with P7-1: the cap and the withheld counts run on every list including a dev-only one, so a dev rehearsal list is also capped; the real dev list is 10 of 49 (20.4%, headroom of 2 items), so a dev subset heavier in shapes would exit 5, which is harmless but should be written down. The exemption is content-based on the committed gold, the probe's excuse set is recomputed from the same gold, and a mixed list is not exempt, so no combination I ran lets a held-out-shaped record skip the probe and still be excused. `withheldShapes` is stored in the marker but is not in `_pins`; a resume cannot change it without a harness-sha change, which `_pins` already compares.

### Stopping condition

Pass 1's five items: (1) met, (2) met, (3) met, (4) met (corpus 0/0/0 and 7 → 0), (5) met except the four survivors above. The series ends when: the `None` default of `record_commit` is a refusal (planted: the executed scenario above refused); a lock double pins "cap before lock"; CRLF fence and CRLF heredoc cases kill the two CRLF mutants. These are test and one-line edits; they need no further pass unless the predicate's rules or `HARNESS_FILES` change again (then a rule change re-runs the Pass 1 differentials: `python -m` unwithheld lines 0, fenced-line disagreements 0/0, dev counts J2 K1 M1 C2 B2 I2 and the 10 ids). F8 stays an open question for the engine owner and does not block.
