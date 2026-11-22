# Connected code graph P1 — AC-24 dev2 custodian log

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (C-1001 P1)

Counts-only custodian log for `cpg/eval/dev/ac24-r2.jsonl` (spec: `docs/plans/connected-code-graph-p1-interpreter-model2-ml.md` §4). No question text appears here.
Custody root: `~/.local/share/ccg-eval/cf7b2042cce7/` (roles `QW-dev2`, `AN1-dev2`, `AN2-dev2`; work files in `work/dev2/`).

## 1. Infrastructure check (before any session)

- `run-writer.sh status`: pinned binary `claude 2.1.312` sha256 `4f4178ce...5d01d` verified; `settings.json` sha256 `8fd894ba...`, `agents.json` sha256 `f88cfb46...` unchanged. No session running.
- EVAL_COMMIT `cf7b2042cce7bad5d0ab7876940bcea00343d914` (`cpg/eval/EVAL_COMMIT`), ancestor of HEAD. Kit `kit-src/`: 1448 files, every file sha256 equals `cpg/eval/kit-manifest.json`, none extra or missing.
- Agents in `agents.json`: `ccg-question-writer` (QW, opus 2 h), `ccg-annotator` (AN1 sonnet 1 h, AN2 opus 1 h): present.
- Control 2 (blindness): the writer's cwd is the role root and `guard-paths.sh` denies any path outside it and any write outside `out/`; tools are `Read,Grep,Glob,Write,Edit` only (no Bash, no web, no MCP). The kit has no `cpg/` (`cpg/ccg/front/`, `cpg/eval/dev/`, `cpg/eval/results/` absent) and no `connected-code-graph*` doc (EXCLUDED_PREFIXES/EXCLUDED_DOC_GLOB, content check). **Deviation noted:** the kit does contain `docs/plans/` (and `docs/reviews/`, etc.) of OTHER components as of EVAL_COMMIT (2026-11-05, before the interpreter existed; zero ccg/interpreter/ac24 hits). The kit is sealed to EVAL_COMMIT and shared with the P0 writers, so it was not altered; judged not blindness-relevant.
- `heldout-manifest.json` sha256 before: `ce2bb8b1c1c7941ede76e742b03ce3a0e7777128cfe376303ae07eccfc6f800d`.

## 2. QW round 1 (stratum `dev2`)

- Brief: the P0 QW brief (`~/.local/share/ccg-eval/cf7b2042cce7/briefs/QW-impact.BRIEF.md`) with only the counts block changed (6 per cell, 6 ambiguous in all, 3 per language) and the session name. No mention of the interpreter, its prompt or a re-check. Brief file `briefs/QW-dev2.BRIEF.md`.
- Session: opus, exit 0, 1 attempt. `audit.py` (S52/S54 pins, settings and agents hashes): **clean**, 52 tool calls, 0 refused, 0 void. The writer made no call outside its role root.
- Candidates: 30 valid of 30 raw (0 malformed). Unambiguous per cell: code-location/en 6, code-location/pt 6, contract-point/en 6, contract-point/pt 6; depends-on 2 in every cell. Ambiguous 6: en several-symbols 1, two-use-cases 1, unknown-name 1; pt several-symbols 1, two-use-cases 1, no-anchor 1.
- qids: `QW-R2-<seq>` (distinct from the P0 `QW-<seq>`, so unique across dev/held-out/reserve). `strip_gold.py` + `--verify`: ok, 30 items, 0 errors.
- AN1/AN2 started in parallel with `QW-nogold.jsonl` in each role root.

## 3. AN1/AN2 and filters, round 1

- AN1 (sonnet) and AN2 (opus): both exit 0, `audit.py` clean, 0 refused (14 and 4 tool calls). Raw agreement 29/30, Cohen's kappa 0.8889.
- §10.3 outcomes (of 30): unambiguous 23, ambiguity 4, drop-disagree 1, drop-writer-ambiguous/annotators-unambiguous 1, drop-writer-unambiguous/annotators-ambiguous 1.
- Filters (pre-committed strict reading: any hit drops; held-out and reserve checked through `cpg/eval/tools/match_keys.py check-report`, run once per candidate against `heldout/ac24.jsonl` + the five `ac5-*` + `ac5-reserve.jsonl`, hit counts only; dev1 by in-process gold-anchor-set equality and normalized-text near-equality (equal or word-set Jaccard >= 0.8)). Drops are per reason and may overlap on one item:

| cell | after join | dev1 anchor-set | held-out text shingle | held-out anchor probe | survivors |
|---|---|---|---|---|---|
| code-location/en | 6 | 0 | 0 | 3 | 3 |
| code-location/pt | 6 | 0 | 0 | 4 | 2 |
| contract-point/en | 5 | 1 | 2 | 3 | 1 |
| contract-point/pt | 6 | 1 | 3 | 4 | 1 |
| ambiguous/en | 2 | 1 | 0 | 1 | 1 |
| ambiguous/pt | 2 | 0 | 1 | 1 | 1 |

  No dev1 text near-duplicate. Survivor ambiguous kinds: one unknown-name, one no-anchor (none of the two kinds selection needs). Depends-on among survivors: 1 per group. The held-out anchor-probe reading is what removes most items; waiving it (information only, not used) would leave 6/6/2/2 unambiguous and 1/1 ambiguous.
- Shortfall: code-location/pt (2 of 3), contract-point/en (1 of 3), contract-point/pt (1 of 3), both selection ambiguous kinds (0), depends-on per group (1 of 2). One top-up round launched (stratum `dev2b`, qids `QW-R2b-<seq>`), same brief with counts only: 12 code-location/pt, 12 contract-point/en, 12 contract-point/pt (>= 4 depends-on each), 8 ambiguous (4 per language, two two-use-cases and two several-symbols each). Larger counts than round 1 because the observed survival rate was about 40 percent.

## 4. Top-up round (stratum `dev2b`) and combined result

- QW (opus): exit 0, `audit.py` clean, 47 tool calls, 0 refused. 44 candidates, 0 malformed: code-location/pt 12, contract-point/en 12, contract-point/pt 12 (depends-on 5/4/4), ambiguous 8 (en and pt: two-use-cases 2, several-symbols 2 each). `strip_gold.py --verify` ok.
- AN1/AN2: exit 0, `audit.py` clean, 0 refused. Agreement 42/44, kappa 0.7755. Outcomes: unambiguous 36, ambiguity 4, drop-disagree 2, drop-writer-ambiguous/annotators-unambiguous 2.
- Combined (74 candidates, two QW sessions, same filters; unambiguous per cell after join / survivors): code-location/en 6/3, code-location/pt 18/5, contract-point/en 17/2, contract-point/pt 18/5; ambiguous en 4/1, pt 4/2. Drops by filter over both rounds: dev1 anchor-set 12 (ambiguous/en 2, code-location/pt 2, contract-point/en 4, contract-point/pt 4), dev1 text 1, held-out anchor probe 41, held-out text shingle 15, intra-pool duplicates 0 (items can hit several filters).
- Depends-on among survivors: code-location 2, contract-point 4.
- **Selection result: SHORTFALL, file not written.** (1) `contract-point/en` has 2 survivors, 3 needed. (2) `two-use-cases` has **0** final-ambiguous items: all 6 candidates of that kind (3 en, 3 pt, across both rounds) did not survive the §10.3 join (annotators judged them unambiguous or disagreed), before any custodian filter. Survivors of kind: unknown-name 1, no-anchor 1, several-symbols 1. (3) With the information-only waiver of the held-out anchor-probe filter, (1) disappears (contract-point/en 6) but (2) remains. Per protocol (control 5) no second top-up and no shape change; the decision is the caller's.
- `cpg/eval/dev/ac24-r2.jsonl`: **not created**. No repo file other than this report was edited.
- `heldout-manifest.json` sha256 after: `ce2bb8b1c1c7941ede76e742b03ce3a0e7777128cfe376303ae07eccfc6f800d` (identical to before). `s47_scripted_steps.py`, `split_manifest.py` not run; no model other than the writer/annotator sessions was run.

## 5. Selection redone under the item-duplication reading (no new session)

Per `docs/reviews/connected-code-graph-p1-interpreter-design-fault-ml.md` Pass 3: the held-out/reserve anchor-probe drop is replaced by full-gold equality (use case, direction, anchor set, scope), compared in-process against the held-out AC-24 items (the only held-out file with item-level gold); the 6-word text shingle check (through `match_keys.py check-report`) and the dev1 filters are unchanged. Same 74 candidates, same seed. Drops (an item may hit several):

| cell | dev1 anchor-set | dev1 text | held-out text shingle | held-out full gold | pool duplicate | after join | survivors |
|---|---|---|---|---|---|---|---|
| code-location/en | 0 | 0 | 0 | 3 | 0 | 6 | 3 |
| code-location/pt | 2 | 0 | 0 | 4 | 0 | 18 | 12 |
| contract-point/en | 4 | 0 | 7 | 7 | 0 | 17 | 2 |
| contract-point/pt | 4 | 1 | 7 | 5 | 1 | 18 | 6 |
| ambiguous/en | 2 | 0 | 0 | 1 | 0 | 4 | 1 |
| ambiguous/pt | 0 | 0 | 1 | 0 | 0 | 4 | 3 |

Depends-on among survivors: code-location 6, contract-point 5. Ambiguous survivors by kind: unknown-name 1, no-anchor 1, several-symbols 2, two-use-cases 0.

**Result: still short.** `contract-point/en` has 2 survivors, 3 needed (the cell is short under the item-duplication rule too; its after-join count is 17, 15 of them removed by dev1 anchor-set 4, text shingle 7 and full gold 7, with overlap). Per Pass 3, no file written. `cpg/eval/dev/ac24-r2.jsonl` does not exist. `heldout-manifest.json` sha256 before and after: `ce2bb8b1c1c7941ede76e742b03ce3a0e7777128cfe376303ae07eccfc6f800d`.
