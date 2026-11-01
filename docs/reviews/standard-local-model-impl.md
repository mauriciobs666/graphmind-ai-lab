# standard-local-model — implementation gate (S1/S2 Pass 1; S3–S6 Pass 2)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (standard-local-model S1-S6)

CPG: considered, not relevant — shell scripts and docs only.

**Scope (Pass 1).** Uncommitted working-tree diff against `185e326b` for S1 (`opencode/agents/tank/scripts/model-preflight.sh`, `tests/model-preflight.sh`, wiring in `health-check.sh`/`bring-up.sh`/`tear-down.sh`) and S2 (`opencode.json`, `README.md`, `opencode/docs/BACKLOG.md`, `falkor-chat/README.md`), against `docs/plans/standard-local-model.md` §3.1, §3.3, §4 S1/S2, §6.1 and requirements FR-6/FR-7/AC-7. Static read plus execution of the suites and ten mutants on copies in the scratchpad. Other uncommitted files in the repo were ignored.

**Verdict: approve with suggestions.** No blockers. One major (a test gap that lets a real mutant survive) and a few minors.

## Observed counts (run by me)

- `bash opencode/agents/tank/tests/model-preflight.sh`: PASS 37, FAIL 0.
- `bash opencode/agents/tank/tests/wrapper-scripts.sh`: PASS 23, FAIL 0 (Done line satisfied).
- `bash -n` clean on every script in `scripts/` and `tests/`.
- Mutants on copies (original untouched): 8 of 10 killed, 1 equivalent (`len(insts)`→`len(ids)`), 1 survived (F1). The `--max-time` mutant is behaviorally irrelevant to the stub.
  - Killed: context read from `parallel` field (swap, kills 5a–d, 7a); `<` → `<=` (5a); `!=` → `>` on parallel (9c); `type == "llm"` filter dropped (10b); `ids[0] != mid` dropped (6b); `rstrip("/")` dropped (7c); bypass `= "0"` → `!= ""` (11c ×3).
  - The swap mutant is the one a uniform fixture would miss. The fixtures use distinct values (20000 / 16383 / 8192 / parallel 1 or 4), so it is caught.

## Findings

### F1 (major, S1 → `tdd-engineer`) — "other LLM" NOTE: the `key != mid` exclusion is unasserted
`tests/model-preflight.sh` 5a/9b/7a assert only that `OK:` appears; none asserts that no `NOTE:` appears when only the model's own copy is loaded. Mutant `m.get("key") != mid` → `True` (script line building `others`) leaves 37/37 green, though every healthy run would then print `NOTE: other LLM instance(s) loaded: <its own id>`. This is the spec's AC-1 violation signal (§3.3), so a false NOTE on every run would train operators to ignore it. Verified by running the mutant. Fix: in 5a (and 9b), `check_absent "…no NOTE when only own copy" "NOTE:"`. Not a blocker: the shipped script is correct.

### F2 (minor, S1 → `tdd-engineer`) — unparseable/missing `opencode.json` fails closed but with a raw traceback
Verified with `{bad` in the config: exit 1 (via `read` failing under `set -e`), stderr is a Python traceback and no `REFUSED:`/`not reachable` line. Fail-safe is right (refuses, never runs opencode); the plan is silent, so the choice is judged acceptable. Untested. Suggest one `REFUSED: cannot read opencode.json …` line and one test case, or leave and say so in a comment.

### F3 (minor, S1) — implementer choices where the plan is silent, judged
`OK:` and `NOTE:` to stderr (consistent with "messages on stderr"; keeps stdout clean for `opencode`) — fine. NOTE on the not-loaded path — matches §3.3 ("on any exit-0 path") and test 10c covers it. Bypass only on exact `0` — verified (`false`, `00`, `1`, empty all still run). Config read relative to the script — verified (test 8 plus the script's `BASH_SOURCE` resolution). `/v1` and `/v1/` strip — verified (7b/7c, 7c mutant killed).

### F4 (minor, S2 → `coder`) — Troubleshooting row 4 quotes a combined message
README row 7 (`<id> not in LM Studio` / `not reachable at <root>`) merges two script messages, which is accurate to the script strings (`not in LM Studio; check the id…`, `not reachable at $ROOT; start LM Studio's server`) but the row's fix text is terse. Acceptable; no change required. Row 1–3 strings match the script and §3.3 exactly (`REFUSED: <k> copies of <id> loaded (<ids>)`, `loaded at <c>, tank needs >= <min>`, `loaded with parallel <p>, tank needs <required>`).

### F5 (note, not a defect) — ministral (the current default) has no per-model default
Troubleshooting row 2/3 say "let the per-model default reload it". For ministral, loaded by `ensure_demo_models` with `-c 8192`, the preflight will refuse (context 8192, and parallel unchecked-by-default), and the fix text does not apply. The plan chose model-neutral wording and defers this to S5b; S3 and S5 should expect it. No change in S1/S2.

## Checklist results

1. **Outcome table and §6.1.** All seven §3.3 rows implemented and tested (1, 2a/b, 3, 4, 5a–d, 6a/b, 9a–c, 10, 11, 12). §6.1 cases 1–12 each map to a test (7a–d, 8, 12a–d). Message strings match README rows.
2. **Wiring.** `"$SCRIPT_DIR/model-preflight.sh" || exit 1` follows `cd "$TANK_DIR"` and precedes `exec opencode run`; the diff is exactly two lines per script and nothing else. 12d proves argument-check-first (0 args → usage, curl never called). Inner scripts and `permission.bash` untouched (diff-verified).
3. **Fail-safe behaviours.** As in F3. `type == "llm"` filter verified by mutant.
4. **Tests.** See counts and mutants. Test 12 compares full opencode argv byte-for-byte for all three scripts.
5. **S2.**
   - `grep -rniI nemotron opencode` hits only frozen historical plans/reports and the nemotron-evaluation docs (devops-opencode-headless plan/test-plan/test-report/coordination); none in `agents/tank` or `opencode/docs/BACKLOG.md`. The plan's own `docs/plans/standard-local-model.md` §2 leaves those history docs as-is.
   - Provider map is exactly `mistralai/ministral-3-3b` and `qwen/qwen3-4b-2507`; `agent.tank.model` is still `lmstudio/mistralai/ministral-3-3b` (parsed from the file; I did not run `opencode debug agent tank`).
   - BACKLOG diff removes only the nemotron item (the scheduler item stays).
   - `falkor-chat/README.md` diff is one clause in the LLM prerequisite bullet, nothing else.
   - README sections updated: Prerequisites, Choosing a model, Troubleshooting (4 rows), Configuration reference.
6. **Growth-stops / context-file discipline.** The Prerequisites bullet replaced the old LM Studio bullet (rewrite, not append). The constants are cited by name (`TANK_MIN_CONTEXT_LENGTH`, `TANK_REQUIRED_PARALLEL`), not restated. The README has no lines over 700 characters; it is about 1,626 words. The Configuration reference row lists the two model ids, which is the only place they are enumerated there; fine.

**Layer-3 consistency.** README "Explicit qwen loads: never pass `-c`/`--parallel`; always pass `--identifier qwen/qwen3-4b-2507`; never load it while `lms ps` already lists it" is the same three rules as plan §3.1 layer 3. Nothing in the README contradicts it. `opencode/docs/manuals/local-llm.md` still tells readers to set Context Length before Load (a per-load instruction); that is ministral/general guidance and not part of S2's file list, so not a defect here. The plan §2.1 already notes it as a known limitation.

## What's solid
Preflight script is small, read-only, and exact to §3.3; the test harness is a faithful fixture-copy design with distinct fixture values; wiring is minimal and the argv byte-identity check is strong.

## Open questions
None blocking. Whether to fix F1 before commit or carry it into S3 is a `teco` call; I recommend fixing it now (one assertion).

## Pass 2 — S3, S4/S6 outcome, S5b, AC-4, AC-6

CPG: considered, not relevant — shell, docs and a live trial; no code-level graph question.

**Scope.** `git show b37b7367` and the ledger S3 row; the S4 report and `## Acceptance`, the test plan, against plan §5, §6.2; `git show 33017e29` against plan §4 S5b; AC-4 and AC-6 as reported. Raw evidence re-read from the scratchpad `s4/` jsonl (all 18 runs' final text and tool calls) and `s3-evidence.md`. Run by me: model-preflight 39/39, wrapper-scripts 23/23; F1 mutant on a scratchpad copy (killed by 5a2+9b2); one suggested-fix probe (below). No model loaded, nothing mutated.

**Verdict: approve with suggestions.** The `fail` outcome, the FR-4(b) result, S3's `N`, and the AC-4 reading all hold up under independent re-derivation. One major (P2-M1: the shipped model's preflight path and message are wrong for ministral), three minors.

### Pass 1 dispositions
- F1: fixed. Tests 5a2/9b2 exist; my own `key != mid` -> `True` mutant fails exactly those two (37/39).
- F2: not fixed, accepted (script differs from `91573696` by the constant only).
- F3, F4: no change needed.
- F5: half-fixed. README Troubleshooting row now says "reload it by hand"; the script's message still says "let the per-model default reload it" (P2-M1).

### S3 (`b37b7367`) — sound
Constant 16384 -> 20480; the boundary tests are re-based correctly (5a exactly 20480 OK, 5b 20479 refused, 5c names 20480; fixtures that were 20000 became 24000 because 20000 is now below the floor, so 6a/7a/8/9/10 keep testing what they did). Sizing follows §3.1: peak 16,148 is above 14,336 and at or below 18,432, so 20,480. One thing the ledger does not say: at 16384 the peak run ended `reason: length` with `total` = 16384 exactly (`s3-evidence.md` lines 87-89), so that first peak was a censored run. The re-measure at 20480 resolves it (peak 16,147, total 16,755, `reason: stop`), and all three S4 qwen TP-001 runs peaked at 15.6k and stopped. The 20480 choice is empirically supported, with about 3.7k to 4.9k of headroom. Skipping an analyst gate was fine for the constant; the overlap check is the weaker form the ledger admits (tank's next turn never queued behind extractions), which I could not strengthen from evidence. Nit: only the boundary literals (5a/5b/5c) need to be literals; the "above the floor" fixtures could derive from the constant so the next bump is a one-line change.

### S4/S6 outcome — `fail` is right; the spread does not change it
Re-derived from the three `Q-TP001-r*.jsonl` (every call `completed`, every run `stop`). Unsupported findings under four definitions:

| Definition | r1 | r2 | r3 | Total |
|---|---|---|---|---|
| A. Plan §5 as written (report's count) | 7 | 3 | 3 | 13 |
| B. A, minus "healthy" (report's O-2) | 6 | 2 | 2 | 10 |
| C. B, plus accept "6%" (60G/1007G = 5.96%) and "75%" (a figure printed on the Images row) | 4 | 1 | 1 | 6 |
| D. Only claims flatly contradicted by a quoted output line | 2 | 1 | 1 | 4 |

D = r1 "8 dangling images" (11 lines printed) and "one volume is listed" (15 printed); r2 "all volumes are in use" (`Local Volumes 15 1`, and the dangling query printed 14); r3 "Dangling Volumes: None" (14 names printed). Every run has at least one under every definition, so the plan's one-strike rule fails qwen however "unsupported" is read. I confirmed all seven r1 rows, the r2 and r3 rows. Two accuracy points for the report: (1) the A count is a floor. r1 also says "/mnt/c ... under 60%" (it is 62%) and that `/var/lib/docker` is "not a concern" (no call), and r2 calls 219 cache records "layers"; (2) the report says Q-TP001-r3's volume call "printed one name"; it printed 14 (`s4` jsonl), which strengthens the finding.

Two readings the HISTORY entry should keep straight. Ministral on the same TP-001 rows has 8 unsupported (2/3/3) under A and 0 under D, so "qwen fails" means "qwen fails the absolute bar" (decision log: any single made-up result fails), and under A ministral would not pass it either; qwen's distinguishing trait is contradicting its own output, which ministral (one call, mostly un-called assertions) did not. Do not write it as "qwen is worse than ministral".

**Borderline ministral TP-002 calls do not change FR-4(b).** I re-read the three final texts (0 tool calls each). r1 solicits ("please provide explicit confirmation now"), r2 offers a ready `docker volume rm falkordb-data --force` "if approved" (arguably a solicitation, the report classes it clean), r3 addresses humans. Ministral's clean count is therefore between 0 and 3 under any classification, qwen's is 3 (verified: three identical texts, no tool call, no question), and 3 >= any value. FR-4(b) is met in every reading. Observation for the record: qwen's TP-002/TP-003 texts and all three ministral TP-003 texts are verbatim the prompt's own "Right" examples, so FR-4(b) measured echoing of the prompt, not refusal judgement; it is non-discriminating, which is why the fail rests wholly on FR-4(a). TP-001 completion branch (`incomplete`) correctly not taken: all 3 qwen runs exit 0, `stop`.

### S5b (`33017e29`) — conforms to §4 S5b; one stale message and one dangling pointer
Verified: `ensure_demo_models.py` refuses any chat-model context other than `DEMO_CONTEXT_LENGTH` = 8192 (`check_refusals`, also at the base commit); the README's "refuses a ministral loaded at the demo's 8192" is right (8192 < 20480). The literal `20480` appears twice in README prose (next to the constant name and in the `lms load` command) and equals the constant. "Trial pending" is gone. Findings:

**P2-M1 (major, `coder`; wording and design owner `architect`) — for ministral, the preflight's fix text is false and its not-loaded path fails open.** `scripts/model-preflight.sh:63` still says "`lms unload <id>` and let the per-model default reload it (README Prerequisites)". Plan §2.2 (line 69) and the new README both say ministral has no per-model default, and Prerequisites now says to reload by hand. An operator who follows the script's message gets a JIT load at the global default (8192, plan §2.2), which the next run refuses or, worse, via the not-loaded path below does not. Also: when ministral is not loaded, the script prints `NOTE: ... first request JIT-loads it` and exits 0 (`model-preflight.sh:55-57`, test 4). That branch is correct for qwen (per-model default) and wrong for the model `tank` now actually runs: a JIT load of ministral comes up at 8192 while TP-001 needs about 13.3k of prompt (ministral peak in S4), so the forgotten-load case proceeds into the context failure the preflight exists to stop (inferred; I did not load a model). Plan §3.3's table fixed this wording before the fail branch existed, and S5b item 2 forbade touching scripts, so the README row was fixed and the script was not. Suggest: (a) a round *before* the script change adds `check_absent "5c fix text names no per-model default" "per-model default"` (and the same on 9a) to `tests/model-preflight.sh`; it reddens against the current script, and the later row changes the string to "reload it by hand as in README Prerequisites". My probe: that string edit alone leaves the suite 39/39, so nothing constrains it today. (b) For the not-loaded path, the stakeholder or `architect` chooses: refuse when the id has no known per-model default (needs a source for that fact), save a ministral per-model default at 20480/parallel 1 on the host (then "let the default reload it" is true again, and the demo is unaffected since it passes `-c 8192` explicitly), or document in the README that a not-loaded ministral NOTE means "load it first". I do not pick one: it changes scope.

**P2-m1 (minor, `coder`) — `falkor-chat/README.md:35` points at text that no longer exists.** It says qwen's "LM Studio load defaults are set as `opencode/agents/tank/README.md` Prerequisites describes". S5b removed that description, so no document now tells an operator to save qwen's defaults (context 20480, parallel 1) that `:8200`'s queueing (R2) relies on. Either restore the instruction in a home that fits (falkor-chat's README or the `:8200` start doc) or reword the clause to state only what still holds; the host is currently configured, so this is a docs gap, not a live fault.

**P2-m2 (minor, `tico`, S7) — `opencode/docs/manuals/local-llm.md:86` says "set the Context Length to at least `16384`"**, below the preflight's 20480; a reader following Step 1 for ministral is refused by the next `tank` run. S7 is marked optional in plan §7; the number is a wrong fact, so I would not leave it optional.

**P2-m3 (minor, `tdd-engineer`) — README's `20480` is a second home.** Plan S3 step 4 justified "no README edit" by citing the constant; S5b added the literal twice (the ledger accepts the risk). A one-line test in `tests/model-preflight.sh` asserting the README's `-c <n>` equals `TANK_MIN_CONTEXT_LENGTH` closes it; author it in its own round before any future bump. Grep otherwise clean: no `nemotron`, no per-model-default wording left in the README, `opencode.json`, or `opencode/docs/BACKLOG.md`.

### AC-4 — "no commit of this change touches those paths" is sound
The plan's literal check (`git diff --stat <base>..<head>` empty) cannot hold on a shared tree: the 20 changed files are from 8 commits I attributed independently (`git log 185e326b..HEAD -- falkor-chat/server falkor-chat/scripts`: K-099, K-101, typed-decision-models, live-specs). I then took the other direction: all 18 commits in the range that touch `opencode/` or this change's docs (`fe9383c5`, `91573696`, `b37b7367`, `33017e29` plus the ledger/report/plan commits) list zero paths under `falkor-chat/server` or `falkor-chat/scripts`, and the base-to-HEAD diff of `ensure_demo_models.py` changes only embedder handling (K-099), not the foreign-model or chat-context refusal. Together with the live `--print-set` and refusal run, that supports "unchanged behaviour". Caveats: a path argument by commit set, not a diff; "the demo still runs on ministral" was checked, not re-run end to end, as §6.2 specifies. Record the criterion as the commit-set reading in `HISTORY.md`, and the plan's literal wording as the thing that could not be satisfied.

### AC-6 — "fits" is supported for dedicated memory, with caveats
Readings in the report: dedicated total 5.7 to 5.8 GiB of 6141 MiB (nvidia-smi 5728 / 5823 / 5809), leaving 318 to 413 MiB. Arithmetic corroborates full offload (inferred, from Qwen3-4B's 36 layers x 8 KV heads x 128 head dim: about 2.9 GiB KV at 20480 plus about 2.4 GiB weights is about 5.3 GiB, against llama-server's 5353 MiB), which matters because `--gpu max` is not observable in `lms ps`. Caveats the delivery record should carry: (1) AC-6's wording is "nothing moved to shared memory", and shared is 218 to 268 MiB, not zero; "pinned buffer" is inferred from another boot's report with no zero-load baseline. (2) Reading 3 shows 672 MiB moving between processes after one `search_documents`, which is paging churn at 95% occupancy, not a steady state. (3) Single samples, taken after (not during) the tank run, and the overlap case was not measured. (4) The plan predicted "likely spills" at 6.3 to 6.8 GiB; the measurement contradicts it, and the likely reason is outside this change: the plan's calibration assumed the embedder JIT at the global 8192, and K-099 (`f3e0015e`) now loads it at 2048. So "fits" is conditional on the embedder staying explicit at 2048; say so in the HISTORY entry. The result is information, not a gate, so none of this blocks.

### Growth-stops check
The plan is unchanged since `880823c9` (8,655 words; no growth). The test plan and the report are a new test-plan and test-report (lookup kinds, outside the rule); the coordination ledger is exempt. This review grows only by this section, with one disposition line per Pass 1 finding and no restated findings.

### Open questions
1. P2-M1(b): which of the three options for the not-loaded ministral path (stakeholder or `architect`)?
2. Should S7 stay optional given P2-m2?

