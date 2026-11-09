# audit-team.sh personal-identifier leak triage

> **Status:** active · **Owner:** `devops` · **Tracks:** —

Triage of the two FAILs of check 7 in `claude/scripts/audit-team.sh` (run from the repo root), and the script change the stakeholder approved on 2026-11-09. Identifier values are deliberately not quoted; locations only.

## What the check matches

- Scan set: `git ls-files --cached --others --exclude-standard`, text files only (`grep -I`), case-insensitive, **minus the exemptions below**.
- Home path: the runtime `$HOME` in four forms (absolute, Claude Code slug, JSON-escaped, URL-encoded), end-bounded; plus Windows-side `Users\<winuser>` forms under WSL (no hits from that half).
- Hostname: `hostname`, `grep -i -w -F`.

## Decision and implementation (stakeholder, 2026-11-09: approved)

1. **Frozen documents skipped by header**: a `> **Status:** archived|superseded` line in the first 12 lines (first token only; never by path). Verified with scratch files: archived and superseded copies skipped; `active` and header-less copies containing `$HOME` FAIL and are named. Scratch files deleted.
2. **Recorded data skipped by path**, in the script variable `LEAK_EXEMPT_PATHS` (rationale comment: recorded evidence, replayed by tests): `cpg/tests/fixtures/headless/` (captured stream/transcript JSONL), `cpg/spikes/frontend-fitness/results/` (recorded JSON), `cpg/spikes/frontend-fitness/sample/` (briefs and JSONL); each sampled and confirmed data, no `.py` inside. 
3. **Left un-exempted, because they are scripts with hardcoded absolute paths (unit U2, `coder`)**: `cpg/spikes/frontend-fitness/adjudication/tools/` (44 `.py` files, 58 lines; the approved list named it as data, but it is code), `cpg/spikes/frontend-fitness/labeller-scripts/` (5 files, 9 lines), `cpg/spikes/frontend-fitness/labels/superseded/javascript/` (`build.py`, `jsview.py`: 2 files, 5 lines; the sibling `javascript.pre-census.jsonl` has no hits), and `cpg/tests/test_eval_tools.py` (2 lines, constants at lines 66 and 450).
4. **Hostname under 4 chars: stricter form, not a skip.** A 3-char whole word collides with finding IDs (the `typed-decision-models4-impl.md:2254` false positive), so a short hostname is matched only as `@<host>` (prompt context, case-insensitive, regex-escaped); 4+ chars keep the whole-word scan. Verified with scratch files, since deleted: `u@<host>:~$` FAILs and is named; a bare ID in a table row does not.

## Counts

Baseline run before the change (own output): home path 743 lines in 146 files; hostname 2 lines in 2 files. (The earlier draft said 744/147; the live-document genericization by another unit landed in between, so live-doc hits are already 0 in the tree now.)

Final run after the change (other units were fixing scripts meanwhile): home path **12 lines in 8 files** (`adjudication/tools/` 4, `labeller-scripts/` 1, `labels/superseded/` 5, `test_eval_tools.py` 2); hostname **1 line** in `docs/reviews/audit-team-leak-exemptions.md` (the analyst review, not this unit). `RESULT: FAIL` remains until those land.
