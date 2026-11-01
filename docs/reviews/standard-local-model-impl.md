# standard-local-model — S1/S2 implementation gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (standard-local-model S1, S2)

CPG: considered, not relevant — shell scripts and docs only.

**Scope.** Uncommitted working-tree diff against `185e326b` for S1 (`opencode/agents/tank/scripts/model-preflight.sh`, `tests/model-preflight.sh`, wiring in `health-check.sh`/`bring-up.sh`/`tear-down.sh`) and S2 (`opencode.json`, `README.md`, `opencode/docs/BACKLOG.md`, `falkor-chat/README.md`), against `docs/plans/standard-local-model.md` §3.1, §3.3, §4 S1/S2, §6.1 and requirements FR-6/FR-7/AC-7. Static read plus execution of the suites and ten mutants on copies in the scratchpad. Other uncommitted files in the repo were ignored.

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
