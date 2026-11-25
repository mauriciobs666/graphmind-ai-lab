# Close-out CO-2 / CO-2a implementation gate (scripts resolver, `recv` record, `;` retry)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1 close-out CO-2, CO-2a)

**Scope.** Commit `b1748405` (read with `git show b1748405:<path>`; a throwaway `git archive` of that sha served for execution, so the in-tree CO-2b edits were not touched). Baseline: `docs/plans/connected-code-graph-p1-closeout.md` §3.1-§3.6 and the CO-2 row; evidence `cpg/eval/results/co2/` and `docs/test-reports/connected-code-graph-p1-closeout-co1-report.md`. Root `REVIEW.md` applied; no component `REVIEW.md` exists for `cpg/` or `docs/`. No held-out or custodian files read; no store or container started.

**Verdict: approve with suggestions.** No blocker: no probe produced a wrong `confirmed` edge (none is ever produced, see m2), the offline fixture and answer suites are unmoved, and the dev claim is reported honestly. Two majors should be closed before the freeze commit F: a wrong-edge path for literal paths (M1) and surviving mutants over real guards (M2). The noise cost of the 324 `possible` edges is unmeasured by design and is the biggest open risk (M3).

CPG: considered, not relevant — the change is a single assembler pass and an extractor edit, read directly; the proving-system graph is the dev store, not a loaded `cpg_<component>` of this code.

## Checks run

| Check | Result |
|---|---|
| `tests/test_fixture_gate_p1.py`, `test_fixture_gate_p1_ac.py`, `test_fixtures.py` (from `cd cpg`, on the sha, `-m "not live"`) | exit 0, 3 / 8 / 80 passed; the live tiers are deselected (not required) |
| 13 answer-engine, projection-golden, `test_answer.py`, `test_engine_bfs*` files | exit 0, 1665 passed, 3 skipped (stored slice absent) |
| the 3 skipped `test_engine_answer_v18.py` tests re-run with the real gitignored `.ccg-work` linked read-only | exit 0, 278 passed, 0 skipped |
| `test_script_resolver.py`, `test_assemble_rows.py`, `test_shell_extractor.py` | green (104 shell-extractor tests; 123 resolver and assembler) |
| fixture trees through the real build and assembler with the new pass | 13 call sites, 0 `CALLS` edges into a `<script>`, 0 drops: no fixture row can move (`an/fx.py`, base trees only; H0 ops not applied) |
| whole `tests/` directory | **not completed**: stalled at 43% in a slow test and was stopped; the targeted sets above are the evidence, not a full green |
| `recv` consumers: grep of `ccg/packrt`, `ccg/linker`, `ccg/engine`, `ccg/assemble`, `packs/` | only `packrt/extract.py` (`value_sets`, looked up by `(callsite, path)`; no shipped pack names `recv` for a shell site) and the new assembler read; `bundle.py:285,324` admits the head. The negative pin runs the packs with and without the records and asserts `recv_sets(recs)` is non-empty |

Own mutants: 30 on `rows.py` (`an/mut.py`), 8 on the extractor (`an/mut2.py`); results in M2 and m5.

## Findings

### M1 (major, verified by probe): a literal path that misses tier 1 falls into tier 2, so a missing or repository-escaping path matches an unrelated script

`rows.py:_script_calls`: `if target is None:  # tier 2` runs for a literal site too, and `_segments` drops a leading `..`. The brief and the CO-2 row say a missing file gives no edge. Planted trees, each through the real build:

- `run.sh` calls `./scripts/gone.sh`; only `lib/scripts/gone.sh` exists: edge `run.sh -> lib/scripts/gone.sh`, `possible/name-match`. The shipped test of this case (`test_a_missing_file_gives_no_edge_and_is_counted`) has no decoy, so it cannot see this.
- `a/b/run.sh` calls `../../../x/t.sh` (one level above the repository): tier 1 refuses it (`startswith("..")`), tier 2 strips the leading `..`s and matches `x/t.sh` inside the repository. `../../sibling-repo/scripts/t.sh` matches `lib/sibling-repo/scripts/t.sh` the same way.

Why it matters: a wrong-`possible` edge, in the family the AC-5/AC-9 wrong-`possible` bound counts. Plan §3.1 tier 2 drops a *computed* head; the docstring's "(or a tier-1 miss)" is the delegate's extension, not in the plan. Markdown tokens do rely on tier 2 for a literal (plan §3.1 item 3), so the rule is language-dependent.

Suggested change: tier 2 only when `dynamic` or the site is Markdown; a shell literal that misses tier 1 is counted `script-call-unmatched` and skipped. Run: with that guard (counting the drop) `test_script_resolver.py` and `test_assemble_rows.py` stay green apart from the "each drop reason is reached exactly once" meta-test, which sees the second `b.drop` literal (restructure to one site). So the suite does not constrain the point at all. **Not verified:** whether any of the 32 dev `shell name-match` edges, or any of the 30 built hits, depend on the fall-through; the fix needs one dev re-ask (cheap: ask only, the assembled rows can be re-derived offline) before it is adopted. Add the decoy test above first.

### M2 (major, mutation-run): 6 genuine surviving mutants over real guards; the delegate's "1 equivalent survivor of 20" is a narrower set

Run on a copy of `b1748405` (`tests/test_script_resolver.py` + `tests/test_assemble_rows.py`; 30 runs, 19 killed, 11 survived of which 10 distinct (the `zsh` mutant ran twice); one of the 30 is the M1 fix variant). Survivors, each classified:

| Mutant | Verdict |
|---|---|
| `p == "/".join(tail) or` removed (tier-2 match on a script whose whole path is the tail, e.g. root `scripts/t.sh`) | genuine: untested, the common layout |
| `known = scripts.get(repo, {})` replaced by every repository's scripts (tier 1 across repositories) | genuine: the cross-repo test covers only tier 2 |
| `lang != "shell"` guard removed (a Python or JS site named `exec`/`source` with a literal argument) | genuine: untested; inferred reach, not planted |
| `zsh` dropped from `_SCRIPT_LAUNCHERS` | genuine: no test |
| `..` no longer pops a real segment in `_segments` (`$HERE/a/../scripts/t.sh`) | genuine: only leading `..` tested |
| tier 1 also run for a dynamic-head site | genuine |
| `_PATH_TOKEN_REFUSED` emptied | inert as written: no later step can match a path holding those characters; it can only refuse a real file name containing `(` or `[` (nit) |
| `"/" not in text` guard removed | changes only the drop counter (one-segment text reaches `unmatched` instead of being skipped) |
| `startswith("..")` in tier 1 removed | equivalent (a known path never starts with `..`) |
| Markdown `callKind == "mention"` check removed | equivalent: the Markdown extractor emits only `mention` (`extractors/markdown/extract.py:313`) |

Suggested: add one test per genuine row (planted decoy or root-level target), then re-mutate. For the extractor (`an/mut2.py`, real-scripts test included in the run): survivors are "retry accepted even if still bad" and "drop the `${`/`;` pre-gate" (equivalent) and "nested `{` depth branch ignored" (genuine, untested; see m5).

### M3 (major risk, unmeasured by design): 324 `possible` edges, 610 returned items for 49 gold items

`summary.json` states noise was not computed (correct: a dev run has no labels) and prints no reduction. Proxy from `co2-asks.json` (`anchorAnswers`, read, not labelled): the 22 dev script anchors return **610 items** (496 `possible` at distance 2, 96 `possible` at distance 1, 18 `confirmed`) against 49 gold items; one anchor returns 124 (`bootstrap_schema.sh`), five return 51 to 57. The distance-2 items are the V-both callee rows through a Markdown caller that mentions many scripts (plan §2 names this cost). Gold is incomplete, so these are not all wrong, but nothing bounds the share that is.

Why it matters: the wrong-`possible` bound (AC-5/AC-9, entry item 3) is the criterion this change spends, and four of the 34 dev hits exist only because of that fan-out (M5/§Honesty). Suggested: CO-9's rehearsal must report wrong-`possible` for script anchors separately from http/mcp; ask `data-scientist` whether a distance-2 `possible` item reached through a prose mention should count towards recall at all (a hit that the resolver did not name is a coincidence, as the ledger already says). No code change is proposed here.

### m1 (minor, verified): the self-call exclusion compares the owning symbol, so a call inside a function is not excluded

`if target == site["symbolId"]` compares the call site's owner. `scripts/t.sh` holding `f() { ./scripts/t.sh; }` (and the same with `"$HERE/scripts/t.sh"`) gives an edge `t.sh -> t.sh`, `possible`. Fix: compare against the owner's file (`sym_path[site["symbolId"]] == path of target`), plus a test with the call inside a function. A script path in the same file is the plan's "self-call" case.

### m2 (minor, plan drift, verified): tier 1 never yields `confirmed` on real data

Plan §3.1 tier 1 says `confirmed`/`resolved-call` and the CO-2 row says "`confirmed` only at tier 1"; the code and the plan's own opening line take "the call site's own status", and every unresolved shell site is `possible/name-match` (`extract.py:645-649`). `summary.json` shows 0 confirmed. The only test of the `confirmed` branch (`test_a_tier_1_edge_is_confirmed_only_when_...`) feeds a hand-rewritten bundle the extractors never emit. The behaviour is the safer one; the drift is unrecorded. Suggested: the architect or the stakeholder rules which sentence stands, and the test is labelled as a fabricated-input check (it passes for any status carried over).

### m3 (minor, verified): a flag's value is read as the script

`bash --rcfile ./cfg/rc.sh ./real/run2.sh` gives an edge to `cfg/rc.sh` and none to the real script; `bash -o pipefail ./real/run2.sh` and `bash +x ./t.sh` give no edge (the value or `+x` is the first non-flag word, no `/`, so the site is skipped). `exec cat ./real/data.sh`, `exec env X=1 ./t.sh`, `exec -a name ./t.sh`, absolute paths, `bash -c '...'` and `bash -c "$x"`: no edge (checked). `bash -xc './real/run2.sh'` gives an edge (a combined `-xc` is not recognised as `-c`; the edge happens to be right for a string holding only a path). Suggested: skip the value of `-o`, `-O`, `+o`, `--rcfile`, `--init-file`; treat `+x` as a flag; recognise a `-c` inside a combined short flag. Low stakes: all `possible`.

### m4 (minor, verified): a computed piece glued to the first segment is read as a whole segment

`"${D}foo/scripts/t.sh"` records `[unknown, const "foo/scripts/t.sh"]`; `_static_path` returns text `foo/scripts/t.sh` with a dynamic head and the match requires only a segment boundary before `foo`. If `$D` is `x`, the real path is `xfoo/scripts/t.sh`. Probe: edge to `foo/scripts/t.sh`. Fix: for a dynamic head, require the literal text to start with `/` (else drop its first segment before matching). Rare in real scripts; included because the brief asks for any wrong-edge path.

### m5 (minor, verified by plant): the `;`→`,` retry can change structure, and recorded callee text, in a file that previously kept no calls

The recorded text is read from the original bytes (verified: `echo "${a:-x; y}"` keeps `x; y`, test pins it), and a real error survives (`if then fi fi`: still one warning, no calls). But `_hide_expansion_semicolons` tracks only `${`/`{`/`}`, not `$(`, quotes or comments:

- `x="${v:+$v; }"` then `y="${z:-$(foo; bar)}"`: first parse fails, retry succeeds, and the commands are recorded as `foo;` with `a0=bar` instead of `foo` and `bar`.
- `x="${v:+$v; }"`, then `echo '${'`, then `./a.sh; ./b.sh`: an unbalanced `${` inside single quotes keeps depth above zero, so later `;` become `,`: recorded call `./a.sh;` with `a0=./b.sh`.

Impact: a file that failed to parse before gets wrong call names instead of none; `;` in a name makes `_PATH_TOKEN_REFUSED` refuse it in the resolver, so no wrong edge follows today. In the real corpus only `test-stamp-wiring.sh` retries. Suggested: stack contexts (`${`, `$(`, backtick) and hide a `;` only when the innermost open context is `${`; I ran that variant against the first plant (correct: `foo`, `bar`); the quote/comment axis is **not** fixed by it and needs quote tracking or a refusal (record a warning and keep no calls when an unbalanced `${` remains at end of file). Add both plants as tests; the `{`-depth branch is an unpinned survivor today.

### m6 (nit, verified): `recv` text exposure is the argument screen's, no more

`recv` goes through `_screen`/`bundle.screen_fails` like an argument. Planted: `API_TOKEN=ghp_...` then `"$API_TOKEN" run` gives `unknown runtime-built`; a token inside a template tail gives `unknown`; `"/home/john.doe@example.com/bin/x"` is a literal callee (no `recv`). What is new: a command word that contains whitespace (it became `<dynamic>`, previously unrecorded) is now recorded as a `const`, and the screen is token-shaped only (`"password is hunter2 for x/y.sh" run` is stored whole). The same holds for arguments today, so no new class; worth one line in the extractor README.

## Honesty of the dev claim (item 5): sound, with two reproducibility notes

Verified from `co2-asks.json` rows (`an` script over `strata.scripts.rows`): the 34 hits are 30 at distance 1 (A 13, D 4, E 5, F 1, G 7) and 4 at distance 2 (G 1, H 1, I 2), all `possible`. The report and the ledger keep them separate ("built families 30/30; 4 coincident"), the `note` in `summary.json` names them, and the 15 misses are reported as `not-listed`. The claim "the resolver names none of them" is borne out by the distance (a Markdown caller reached the anchor first). Notes: (a) `summary.json` splits G only by family, so "7 two-segment G hits" is checkable only through the distance column; (b) the measured clone was `9728c632` plus an unrecorded private commit, and the ledger says one dead local was removed afterwards: the measured code is not addressable by sha. Neither changes a number; record the clone commit sha in `summary.json`.

## What's solid

- Tier boundaries: segment-aware suffix, one-segment rejection, ambiguity (including the root-equality case) and cross-repository suffix are killed by tests; `confirmed` never reaches tier 2.
- No decoy, case, unicode or symlink probe produced a wrong edge (they give no edge: false negatives only). `..` handling within a repository, `//`, trailing `/` and `./x/../y` normalise correctly; absolute, `~`, `$`-bearing and `bash -c` inputs are refused.
- The `recv` carrier: accepted by the validator, no data record, no pack or linker reads it, written only for a non-`eval` `<dynamic>` site (the mutants for the id, `eval` and the screen are killed).
- The fixture, gate and answer suites are unmoved; the fixtures hold no script-to-script call, so no golden could move. The delegate did not edit any golden.

## Open questions

1. Is tier 2 meant to apply to a *literal* shell path that missed tier 1 (the docstring) or only to a computed head (the plan text)? M1 assumes the plan.
2. Which sentence stands for tier-1 status (m2)?
3. For `data-scientist`: should a distance-2 `possible` hit reached through a prose mention count as recall at the gate (M3)?

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: M1 (the shipped missing-file test has no decoy) and m5 (retry tracks one nesting kind).
- A resolver or matcher that returns "no edge" for a missing target is checked with a decoy that shares the unmatched path's tail in another directory, and with a path that climbs out of the root; a no-match test without a decoy cannot fail.
- A text rewrite that tracks nesting with a depth counter is probed with each other nesting kind the grammar has (command substitution, backticks, quotes, comments) inside and outside the counted one.

Growth-stops check: not applicable (a new review, not a revision of a requirements, plan or earlier review document).

## Pass 2 (CO-2c, commit `bb073e55`)

**Verdict: approve with suggestions.** Every Pass 1 finding that CO-2c claims to fix is confirmed fixed by execution; no blocker. Two small residuals (P2-1, P2-2) and one design decision for the owner (P2-3). Executed on a `git archive` of `bb073e55`: `test_script_resolver.py` + `test_shell_extractor.py` + `test_assemble_rows.py` 255 passed 1 skipped; fixture gate suites exit 0 (3 / 8 / 80); 12 answer-engine and golden files exit 0 (1661 passed, v18 run with the real `.ccg-work` linked read-only); the Pass 1 probes re-run (`an/probe_3.py`, `p2_3.py`, `semi_3.py`).

### Dispositions of earlier findings

- **M1** fixed. `./scripts/gone.sh` with a decoy `lib/scripts/gone.sh`, `../../../x/t.sh`, `../scripts/t.sh`, `../../sibling-repo/...`: all give no edge and count `script-call-unmatched`. Markdown still matches by suffix, by design (see P2-2).
- **M2** fixed. My six survivors re-run as S1-S6 on the new code: all killed. A fresh batch of 25 on `_resolve_script`/`_site_path`: 22 killed; survivors N13 (`len(word) > 1`, a bare `-`/`+` word; negligible), N19 (Markdown flag `lang != "shell"`, equivalent: only shell and Markdown reach it), N24 (absolute-path refusal removed: genuine, the guard is untested for Markdown, P2-2).
- **M3** unchanged by this round and still unmeasured (`co2c/summary.json`: `noise: not computed`); the pass removes 6 edges of 432. Carry to CO-9.
- **m1** fixed (call inside a function, literal and computed: no self-edge; a function calling another script still edges).
- **m2** see item 5 below: the sentence that should stand is recorded in the code docstring; the plan is not yet amended.
- **m3** fixed: `--rcfile F ./t.sh`, `-o pipefail ./t.sh`, `+x` resolve to the script; `-c`/`-xc`/`-ec` give none. A cost: `bash -xc './real/run.sh'` now gives no edge (was a right edge); accepted, a command string is not a path.
- **m4** fixed, by refusing rather than dropping the segment (item 3).
- **m5** fixed for the `$(...)` axis (`${z:-$(foo; bar)}` now `foo`, `bar`); the quote axis is documented as out of scope and still produces `./a.sh;` with `a0=./b.sh` (P2-1). Extractor mutants (6 on `_hide_expansion_semicolons`): 4 killed; survivors "paren pushed outside an expansion" and "`}` pops any top" are unpinned (nit).
- **m6** unchanged (nit).

### Answers to the five questions

1. **The 6 removed edges are true cwd-relative calls.** `falkor-chat/scripts/test_queries.sh:114` runs `EMBEDDING_DIM=4 ./scripts/bootstrap_schema.sh test`, with no `cd` anywhere in the file; the suite is documented to run with `falkor-chat/` as cwd. The 5 evidence copies under `falkor-chat/docs/reviews/` carry the same line. I reproduced the corpus: an offline assemble of every tracked `*.sh`/`*.md` (1332 files) gives 8 shell-literal misses, of which exactly these 6 have an *exact* path under an ancestor of the caller's directory (`an/corp.py`). So the removal loses 6 true edges for 0 dev hits (the implementer's count; hit set unchanged). Acceptable for the freeze as a conservative loss, but it is a recall loss on a real, common idiom (cwd = component root). A plan ruling is the right route, and I ran the candidate: **tier 1 also tries the caller's ancestor directories, exact path only, no `..`-leading text, status the site's own (`possible`)**. With it the resolver and assembler suites stay green (145 passed), all M1 plants stay edge-free (the decoy `lib/scripts/gone.sh` is not an ancestor-relative path of `run.sh`), and the 6 edges return. It is not pinned by any test today, so adopting it needs its own decoy test (an ancestor path vs a sibling directory path). Not a precondition for F unless the stakeholder counts these 6 edges as wanted.
2. **Mutation**: see M2 above.
3. **m4 deviation is better than the brief.** The brief said drop the first segment; the implementer refuses the whole path when a computed head is glued (`"${D}foo/x.sh"`). A trailing-slash idiom (`ROOT=/x/; "${ROOT}scripts/t.sh"`) has a whole first segment, so dropping it would leave the one-segment tail `t.sh` and hand it to the own-directory tier, a wrong sibling edge; refusing loses at worst a true edge. Cost measured: 0 glued sites in the 1332-file corpus (`an/corp.py`). Accept.
4. **Interactions, executed.** m3 with m4: `bash -o pipefail "${D}foo/x.sh"` is refused, `bash -o pipefail "$D/scripts/x.sh"` resolves (value skipped, then tier 2). m1 with the own-directory tier: `"$HERE/run.sh"` inside `run.sh` gives no edge (path compare, not symbol compare), and the one-segment tail never leaves the caller's directory (tests present, N14/N15 killed). M1 with tier 3: tier 3 requires a computed head, so a literal never reaches it. A literal path `..foo/t.sh` (a directory whose name starts with two dots) now gives no edge (tier 1 refuses `startswith("..")`, tier 2 no longer retried): a harmless false negative.
5. **m2: the sentence that should stand is "the edge takes the call site's own status; never `confirmed` on real data".** The shell extractor marks every unresolved site `possible` (`extractors/shell/extract.py` call-site construction), so the plan's "confirmed at tier 1" can never fire, and the requirement is "at least 95% of confirmed items real" (`docs/requirements/connected-code-graph.md` ~l.349, ~l.596): a cwd-dependent literal path cannot meet a confirmed bar (it is exactly the cwd question of Q1). Amend plan §3.1 and the CO-2 row (architect); keep the fabricated-input test, labelled as such (it is).

### New findings

**P2-1 (minor, verified by plant): the quote axis of the retry still yields `;`-polluted call names.** `x="${v:+$v; }"` then `echo '${'` then `./a.sh; ./b.sh` records `./a.sh;` with `a0=./b.sh`. The docstring discloses it and the resolver refuses a name with `;`, but the recorded call is wrong. I ran a one-condition remedy: if the context stack is non-empty at end of file (an unbalanced `${` or `(`), return the original bytes, so the retry fails and the file keeps no calls and the parse-error warning; all plants then give the right result (`unbalanced single-quote + semis` -> no calls, the `$(`, baseline and still-bad plants unchanged). Suggested for the same unit as a test; not a precondition for F (corpus: one retrying file).

**P2-2 (minor, verified by mutant and inferred by reading): Markdown paths keep the repository-escape and the absolute-path guards untested.** Mutant N24 (the `text.startswith("/") and not dynamic` refusal removed) survives: no test plants a Markdown mention of an absolute path, which would otherwise suffix-match an in-repository script. A Markdown token such as `../../x/t.sh` still strips its leading `..` and suffix-matches (the M1 escape class, by design for prose mentions, `possible/prompt-mention`). Suggested: one Markdown test for an absolute token and one for a climbing token, each asserting the intended outcome whichever way the owner rules.

**P2-3 (decision, not a defect): ancestor-directory tier for shell literals.** See answer 1: the owner chooses between accepting the 6-edge loss and adding the ancestor tier (verified safe against all M1 plants, 145 tests green, unpinned).

### Stopping condition (falsifiable)

This gate is closed when: (a) a test plants an absolute and a `..`-climbing Markdown token and asserts the ruled outcome, and the N24 mutant is killed; (b) the owner records, in `docs/plans/connected-code-graph-p1-closeout.md` §3.1, either the ancestor tier (with a planted ancestor-vs-sibling test) or the ruling that the 6 `./scripts/bootstrap_schema.sh` edges are lost by decision; (c) plan §3.1 tier-1 status wording matches the code. If a re-run of `an/probe_3.py`-style plants (decoy, escape, flag value, glued head) shows any edge or call the plant says must not exist, the gate reopens.

Growth-stops check: this pass appends one section to a review revised in place (allowed); earlier findings carry one line each, no restated prose.
