# model-bench defects — live-spec close check

> **Status:** archived · **Owner:** `analyst` · **Tracks:** GAL-F0028 (model-bench defects)

**Scope.** Close step 3 (root `AGENTS.md`): the uncommitted live-spec edits (`model-bench/docs/DESIGN.md`, `docs/BACKLOG.md` item GAL-F0028, comment edits in `tests/test_scoring_extraction.py`) against the shipped code of `3aafd1bb`, `6c7dda13`, `f9773ffb`. Root and `model-bench/REVIEW.md` applied. Not reviewed: the manual (`tico`'s later unit; stale lines listed below). Every claim below was executed, in a scratch root under the session scratchpad (a copy of stored runs, a stub LM Studio, a copied tree for mutants); nothing in the repo was edited.

**CPG:** considered, not relevant — a spec-vs-behaviour check settled by running the CLI.

**Verdict: approve with suggestions.** Every shipped contract is stated correctly. One true statement was deleted and not replaced (F1, major), plus a wrong table-shape phrase (F2) and a stale manual line (F3). No contradiction of shipped code.

Suite: `.venv/bin/python -m pytest -q` in `model-bench/`: 3674 passed, 5 deselected, 2 xfailed.

## Findings

### F1 (major, verified) — the deleted "verdict compares the first two arms" sentence is still true under `--models`

The edit removed "The verdict compares the **first two** arms; the arms table lists all of them." It is wrong for plain `compare` but still correct for `--models`. With `--models qwen/qwen3-4b-2507,smollm3-3b,stablelm-zephyr-3b` over three stored arms the CLI exits 0, the arms table holds all three (stablelm at line 17) and the headline pairs the first two (paired-n line names only qwen and smollm3). §2.3 now says nothing about how many arms a verdict covers or what happens past two on `--models`. Suggest: add one clause to the `--models` sentence: "more than two keys are all tabled; the verdict pairs the first two." Also state the arm order for plain `compare`: first appearance of each key in file-name order (`list({r.armKey: r …}.values())` keeps first-seen position).

### F2 (minor, verified) — "The arms table has one row per arm" is not what the report prints

The table has one row per arm and metric (four rows per arm in the `chat-responder-grounded-answers` reports I generated). What the fix guarantees is that each arm appears once. Suggest: "each arm appears once in the arms table (its newest run)".

### F3 (minor, verified) — the manual's line 146 is stale (for `tico`, not edited here)

`docs/manuals/small-model-benchmarking.md:144-146` says the final plain `compare` "lists all three models in its arms table, but its verdict covers only the first two by file-name order". Plain `compare` over three arms now exits 2. Lines 119-120 (`run --reference`), 461 (`validate --strict`) and 473 are already updated in the working tree. I read no other manual lines.

### F4 (nit) — wording

`DESIGN.md:106` "A LM Studio failure" reads "An LM Studio failure". `runner._start_of_run_read`'s docstring cites steps 3, 3a, 7, 8; those are the runner-spec numbers, DESIGN §3.2's are 4, 8, 9 (see check 5). No change needed unless the docstring should say which document it numbers by.

## Checks (executed)

1. **`run --reference`.** `python3 -m modelbench run --pack x --model m --reference k` exits 2 ("unrecognized arguments: --reference k"); `rank --pack … --reference smollm3-3b` ran and exited 0; `rank --help` lists it. §2.9 and the §2.9 usage line are true. The usage line matches the parser (`run.add_argument("--reference")` is gone).
2. **Exit 5 text and `results/transcripts/`.** The message is now "residentModelsAtEnd could not be captured after three retries" (`runner.py:2105`), with no path. The spec states no message text except row 5's paraphrase, which fits. §3.8 ("no run writes raw model replies there"), §2.2's file table (`DESIGN.md:1177`) and `.gitignore` are still true: `grep -rn transcripts modelbench/*.py` finds only docstring mentions, no writer.
3. **`compare`.**
   - 16 distinct keys, plain: exit 2, one stderr line naming `--models`, no file written.
   - Four runs of two keys (qwen ×3, smollm3 ×1): exit 0, two arms; qwen shows 23/30, not the oldest run's 15/30.
   - One arm: exit 0, "fewer than two arms" verdict. Zero arms: same, exit 0.
   - Three keys, plain: exit 2. Same root with `--negative-control`: exit 0, first record duplicated (the BACKLOG's third bullet is true).
   - Limitation: a copy of the 21:04 run renamed `…T00:00:00Z.json` sorts before the 18:19 run; both plain `compare` and `--models` then print the 18:19 run (15/30). The spec sentence is accurate, including "under `--models` too".
4. **`validate --pack packs/nlq-structured-query --strict`**: exit 2, one stderr line, no traceback. Matches §2.8.
5. **Exit 3.** DESIGN §3.2's own numbering: step 4 = residency + catalog, step 8 = baseline residency, step 9 = catalog after load (+ instance config on a candidate arm). The spec's "steps 4, 8 and 9" names all four reads correctly; the code's 3/3a/7/8 are the runner-spec's numbers. Row 3 (start-of-run reads, item-1 baseline residency, `loadedContextLength` catalog read, instance config) and the sentence under it agree. Executed with a stub server: HTTP 500 and a malformed body on the start-of-run reads both give exit 3 with "nothing was run or stored". I could not drive steps 8 and 9 live without a chat endpoint. Instead, in a copied tree, I replaced each wrapped call (step 7, then step 8) with the bare call: `test_run_exits_three_when_a_read_between_the_probe_and_item_one_fails` goes red for each (`residency-2-unreachable`, `catalog-2-unreachable`).
6. **Other stale statements.** Unfiltered grep of `DESIGN.md` and `FORMATS.md` for `reference`, `NotImplemented`, `strict`, `first two`, `file-name`, `traceback`, `exit 1`, `transcripts`, `newest`: no remaining false statement except F1's omission. Row 2 ("§2.3–§2.9 name each case") holds: §2.3 names the >2 refusal, §2.8 `--strict`, §2.9 `--reference`. `README.md:55` documents `rank --reference` only, which is true. `FORMATS.md:292` (file-name order of `load_history`) is true.
7. **Live-spec rules.** Grep of the added lines for fixed/no longer/previously/formerly/used to/now: no hit. Numbering kept; no Version or revision note.
8. **Growth-stops.** `DESIGN.md`: +16/-17 lines. The §2.2 and §2.3 growth is new scope (the `compare` arm rules). The two deleted defect paragraphs are replaced in place. The one restated fact is the "newest" rule, which §2.3 states and §2.4 cites by pointer. No finding.
9. **BACKLOG item.** Forward-looking only: the five delivered bullets are gone, three residuals are left, all true (limit verified in check 3; `referenceKey` exists at `runner.py:166`, set to `None` at `cli.py:759`, 8 occurrences in `tests/test_runner.py` and `tests/test_runner_decisions.py`; negative-control verified in check 3). One imprecision: the fix suggestion says "the one place both branches share", but `_select_arms` has two (`by_key` and the `candidates` collapse). Say "in both." Delivery must be in `HISTORY.md` (the coordination plan's close step; not yet written; not checked).
10. **ID reuse.** `.work-items/registry.jsonl:123` carries title "Defects the live spec recorded", and `wi set` refuses to change a title (`skills/work-items/docs/SPEC.md:59`), so the registry title stays stale. Acceptable: the item is the same work (the remaining defects of that record), `wi check` does not compare titles, and a retire-and-mint would orphan the `Tracks:` cites in `reviews/model-bench-defects.md` and the coordination plan. Cost: those two documents still say "defects the live spec recorded" (`reviews/model-bench-defects.md:3`, `plans/model-bench-defects-coordination.md:3`). Alternative if the owner prefers a clean title: retire GAL-F0028 and mint a new F for the residuals.
11. **`wi.py check`** (`python3 skills/work-items/scripts/wi.py check`): exit 1, one problem, `family-no-item: claude/docs/reviews/coordination-graph-guard.md: family of 'coordination-graph' lacks Item:`. Unrelated to model-bench.
12. **Test comments** (`test_scoring_extraction.py`): both now describe the `--strict` exit-2 refusal accurately; no behavioural change.

## What's solid

§2.2 row 3 and its sentence, §2.8, §2.9, the `compare` collapse/refusal/exempt/limitation statements and §3.8 all match what I executed. The history-word and numbering rules are met.

### Proposed REVIEW.md lines

None. F1 is a one-off omission of an unchanged contract, which root `REVIEW.md`'s "read the unchanged lines" and "contract the diff changed that the spec omits" lines already cover.
