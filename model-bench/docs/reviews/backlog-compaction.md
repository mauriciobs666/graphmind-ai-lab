# model-bench BACKLOG compaction audit

> **Status:** archived · **Owner:** `analyst` · **Tracks:** —

CPG: considered, not relevant — a documentation audit over a ~15-file Python component; every claim was checked by reading and running the code and docs directly.

**Scope.** `model-bench/docs/BACKLOG.md` (205 lines, 15 bullets under `## Open items` plus the 5-bullet `### Defects the live spec recorded`) against the code at the working tree (HEAD `9ad829f1`; `modelbench/` unmodified per `git status`), `docs/HISTORY.md`, and the live spec (`docs/DESIGN.md`, `docs/FORMATS.md`). Read-only: nothing but this file was written. Verdict on the backlog: **no delivered item is hiding in it; every item is still open. Compaction is therefore a rewrite job, not a deletion job** — plus one false statement in the defects intro (G-1) and nine stale line/count references.

Line numbers are BACKLOG.md's current ones.

## Table

| # | Item (first words) | Lines | Class | Evidence (what I ran/read) | Action |
|---|---|---|---|---|---|
| 1 | Judged reply quality for the `chat-responder` pack | 8-15 | **B** | `grep -rn judgeModel modelbench` → 0 hits (no judge exists); DESIGN §6.5 (l.646) says "deterministic layer only". No HISTORY entry delivers a judge (correct: it is deferred). | keep |
| 2 | +22 harder retrieval queries | 17-20 | **B** | `wc -l packs/embedder-graphrag-retrieval/queries.jsonl` → 38; `report.py:1477` still prints "recall@10 = 37/38 … only 1 item is available to win". | keep |
| 3 | `tools/exec.py`'s `order_by` sort raises `TypeError` | 21-35 | **B**, trail in text | `exec.py:376-377` `deduped.sort(key=lambda r: r.get(order_prop), …)` with no None handling; `pytest tests/test_tools_exec.py` → 45 passed, 2 xfailed (the strict xfail on `…places_nulls_the_way_falkordb_does` is live, so the defect reproduces); pack `packVersion` 1.2.0, and 18/18 `results/runs/nlq-structured-query-*.json` carry `packContentHash` `8f975842…` (item's "18 runs / 8f975842" confirmed). DESIGN §6.4 (l.631-633) already states the behaviour and points back at this item (`open: docs/BACKLOG.md`), so the item is the fix plan, not a duplicate. HISTORY has no delivery. | rewrite (shorter; keep — DESIGN's pointer needs it to exist) |
| 4 | `report.py`'s lucky-pass labelling (†/‡) covers k/n cells only | 36-48 | **B + E** | Code: `_lucky_mark` (`report.py:1360-1373`) returns "" unless metric starts `exactMatchBy` (l.1321); `compare_report` verdict path builds `v.text` from `stats.verdict` at l.3134 with no lucky handling; `_render_scorer_counts` (l.1396) has no `‡`; `_LUCKY_PASS_DISAGREEMENT_FOOTNOTE` (l.1309) is "The arm's …" singular while `{arms}` can list several. `pack.json` of `nlq-structured-query`: `verdictMetrics == headlineMetric == layer1ExactMatchRate`, so unreachable by a shipped pack. "Optional polish … Pass 2, N3/N4" is a second finding stacked onto the first. | rewrite |
| 5 | `results.Outcome`'s `Literal` … missing `"unrunnable"` | 49-59 | **C** | `results.py:38` still `Literal["pass","fail","n_a","parse_failure"]`; `"unrunnable"` is constructed in `classification.py:588`, `extraction.py:221`, `grounding.py:233`, `retrieval.py:421`, `agreement.py:357`, `runner.py:815`, `toolcalls.py:969` — so the defect stands (B), but the cites are stale: `classification.py:218-221` and `extraction.py:216-221` no longer point at the constructions (now 588 / 221), and the item names only two of seven sites. DESIGN §6.1 (l.554-555) already specifies the 4-member vocabulary **including** `unrunnable`, so the live spec is ahead of the type. No `mypy.ini`/`pyright*` anywhere (`ls`, `grep mypy pyproject.toml` → none). | rewrite |
| 6 | `checklist_pass` containment is plain substring | 61-75 | **B** (long) | `scoring/grounding.py:156-159` `_canon_str(c) in canon_reply` — still plain canonical substring; `items.jsonl` cr-11 ("4 retries"/"4 retry attempts") and cr-17 ("30 minutes" / "30-minute") contexts present. DESIGN §6.5 l.651-652 states "canonicalized substring match" as the current rule but not the paraphrase limit, so the item carries the only statement of it. | rewrite (trim S7-report archaeology) |
| 7 | `report.py` prints no per-script breakdown for `tool-caller` | 77-81 | **B** | `report.py` comment near l.1058: "`RunResult`/`ToolCallAggregates` stores no per-script length either"; `grep -n 'byScript\|per-script' report.py` → no table; `scripts/s6_walkthrough.py` exists. | keep |
| 8 | `ministral-3-3b`'s rung-2 censoring (69/81) not root-caused | 82-86 | **B** | `docs/test-reports/small-model-benchmarking-s6-report.md:256,270` state the same gap; no later HISTORY root-cause. Conditional ("only if `plaintext` is ever considered") — a trigger, not planned work, but still open and accurate. | keep |
| 9 | No paired significance test for a turn-pooled exploratory metric | 87-91 | **C (minor)** | Gap is real (DESIGN §10: turn/call-pooled counts print without an interval). But the example "duplicate-instruction rate" is the S6-report name; the shipped metrics are `duplicateWithinTurn`/`duplicateCrossTurn` (`toolcalls.py:326`, DESIGN l.672), `grep -i duplicate-instruction modelbench` → 0. | rewrite (rename example) |
| 10 | `validate_pack` … crashes when a declared data file is not yet authored | 93-104 | **B** (+ dated repro) | **Reproduced today** on a copy of `packs/nlq-structured-query` with `items.jsonl` removed: `load_pack` ok, `validate_pack` → `FileNotFoundError` via `packs.py:1415 → :949 (_answerability_stamp_problems) → :306 (iter_items)`. Siblings guarded: `awk` of `packs.py` shows `try/except (OSError, json.JSONDecodeError)` at l.709-711 (row count) and l.988-990 (cleanThroughTurnH). Callers `cli.py:691,724` do not catch. DESIGN §5.2 axis 7 lists the check but not the crash. | rewrite (drop 2026-09-18 narrative) |
| 11 | `stats.verdict()`'s "is better than" wording is polarity-blind | 106-120 | **B**, stale cites | `stats.py:1400-1405` still `winner, loser = (a_label,b_label) if diff >= 0 …` → "X is better than Y"; `compare_report` prints `v.text` unmodified (`report.py:3134-3148`), whereas `rank_report` corrects via `_polarity_corrected` (l.1619/1682) and `_LOWER_IS_BETTER` (l.159). `classification.py` cites `:209-210,259`/`stats.py:1320` are stale (assignments are described at `classification.py:509-512`; the verdict site is `stats.py:1400`). DESIGN §6.1 "Polarity" (l.571) states polarity only for `rank` and reference-anchored verdicts, so the `compare` gap is not in the spec. `ls results/runs/guard-judge-*` unchanged claim not re-run. | rewrite (fix cites) |
| 12 | `rank_report()`'s reference-not-found message is oddly worded | 122-134 | **C** | Message is at `report.py:1843-1846` (item says 1362-1368). More importantly the proposed fix ("tie the suffix to whether `session` was actually passed to `rank_report`") is impossible as written: `rank_report`'s signature (l.1758-1765) has **no `session` parameter**; `--session` is applied upstream by `cli._select_rank_arms` (cli.py:548). DESIGN §2.4 l.152 documents the exit-2 refusal but not its wording. Genuine cosmetic nit remains. | rewrite (or delete as a nit — decision below) |
| 13 | `rank_report()`'s zero-`n` banner double-counts | 136-173 | **B + E** | **Reproduced**: `_attempted_for_metric` (`report.py:213-236`) on 70 items with `scoreable={}` returns `70` for both `falseAdvanceRate` and `falseSuspendRate`; `classification.py:586-596` emits `scoreable={}` on no-response. `grep` over all `results/runs/guard-judge-understanding-*.json`: **19** files (item says 17), **0** empty-`scoreable` items, so still not observed live. Item is 38 lines with a "Verification round 2 — 2026-09-20" trail and its own fix design paragraph. | rewrite |
| 14 | `nlq-structured-query` pack no longer mirrors falkor-chat's production parse | 175-183 | **C (minor)** | `git log 7f3ad0e5` = K-066 U4; `falkor-chat/server/falkorchat/querygen.py:268 merge_duplicate_keys`, `:324 default_returns` exist. Pack side: `tools/exec.py:126-129` rejects a missing `returns`. But the JSON parse is **not** in `tools/exec.py`: it is `scoring/extraction.py:227 → classification.extract_own_line_json_object` (`classification.py:99-100` plain `json.loads`), i.e. **outside the content-hashed pack**; so "a pack version bump that copies the two normalizations" understates the change (a scorer edit is needed for the repeated-key merge). `PROVENANCE.md` has no note (`grep` → none). Not in HISTORY. | rewrite |
| 15 | An LM Studio outage met by an item's own call is not re-probed | 184-192 | **B** (partial D) | `grep -n '\.probe()' runner.py` → only `:381` (`_between_item_residency`) and `:1883` (pre-flight); item-call catches (`runner.py:806,914,482,631,1586`) score `timeout`/`no_response` with no re-probe. DESIGN §3.5 l.307-308 records the current behaviour ("does not stop the run") and l.363 the between-item re-probe; the *unruled* half (decisions/handoff, multi-turn `tool-caller`) lives only here. `docs/plans/typed-decision-models4.md:709` is the R-21 source. | keep |
| D1 | Defects: `run --reference` accepted and ignored | 199 | **B** | `cli.py:183` adds the flag, `cli.py:741` passes `referenceKey=args.reference` into `RunConfig` (`runner.py:166`); `grep -n referenceKey runner.py` → declared, never read. DESIGN l.211-212 states it. | keep |
| D2 | exit-5 message names `results/transcripts/<runId>.jsonl` | 200 | **B**, **not in DESIGN** | `runner.py:2082` (`transcript retained at results/transcripts/{run_id}.jsonl`, exit 5). DESIGN l.407 says only that no run writes transcripts; **nothing in DESIGN says the exit-5 message names it**. | keep; spec gap (G-1) |
| D3 | plain `compare` pairs first two arms by file-name order | 201 | **B** | `cli.py:223-240` (`_select_arms`: no `--models` ⇒ `candidates` in `load_history` order, oldest-first/file-name order). DESIGN l.125-126 states it. | keep |
| D4 | `validate --strict` raises `NotImplementedError` | 202 | **B** | `cli.py:680-682`; DESIGN l.192-193 states it. | keep |
| D5 | LM Studio timeout during `run`'s start-of-run residency/catalog read → traceback | 203 | **B**, **not in DESIGN** | `runner.py:1887,1890` (`residency()`/`catalog()` at steps 3/3a) are outside any `try`; `lmstudio.py:660-666,627` raise `LMStudioUnreachable` on no response; `cli.py:749-752` catches only `RunRefused`. Read from code, not reproduced (needs a live/mocked timeout). DESIGN §4.2 l.459 names the exception but not this consequence; the exit-3 row (l.102) implies the opposite. | keep; spec gap (G-1) |
| D6 | "**Done when:** each is fixed or removed…" | 205 | **E-adjacent** | A bullet inside the list carrying an obligation to edit the spec "in the same change"; also the intro (l.196-197) says each defect "is documented there as current behavior", **which is false for D2 and D5** (see G-1). | rewrite |

**Counts.** A (delivered, delete): **0**. B (open, accurate — keep or trim): **1, 2, 3, 6, 7, 8, 10, 11, 13, 15, D1-D5** (3, 6, 10, 11, 13 also get a rewrite for length/stale cites). C (stale text): **5, 9, 12, 14** (and 11's cites). D (belongs in spec): **none moved** — see "Decision on D" below. E (history/stacked clauses): **4, 13**, trail-only in 3, 6, 10. F (duplicates): **none** — see "Decision on F".

## Findings the table does not carry

**G-1 (major). The defects intro is false for two of five defects.** BACKLOG l.196-197 says each defect "is documented there [DESIGN.md] as current behavior until fixed". DESIGN documents D1 (l.211-212), D3 (l.125-126), D4 (l.192-193); it does **not** document D2 (the exit-5 text names a transcripts path no run writes) or D5 (a timeout on the start-of-run residency/catalog read is an uncaught traceback, exit 1, not exit 3 — the exit-3 row l.102 reads as if it were handled). Owner decision, two ways to close it: (a) a spec writer adds one sentence each to DESIGN §3.1/§4.2 (the live spec is a statement of current behaviour, so defects belong in it) and keeps the BACKLOG bullets; or (b) BACKLOG drops the "documented there" claim. This audit proposes (a) as the contract the "Done when" line already assumes; replacement intro text below states what is true either way. **I did not edit DESIGN.md.**

**G-2 (minor). Cross-reference that constrains this file.** DESIGN l.633 ("open: `docs/BACKLOG.md`") and the xfail `reason=` in `tests/test_tools_exec.py` (~l.396-401, "docs/BACKLOG.md: …") both point at item 3; keep item 3 (shortened) and keep its identifying words ("`order_by` sort raises an uncaught `TypeError`"). Also `tests/test_tools_exec.py` and `s4-gate-followups.md` are cited by path from the item, so the rewrite keeps both.

**G-3 (minor). Size.** BACKLOG is 2,334 words; the AGENTS.md 2,500-word bar is for context files, not a backlog, and no line exceeds 700 chars (`awk 'length($0)>700'` → 0). The weight is review archaeology (`Pass 1 — FX1, N2`, "Verification round 2 — 2026-09-20", "Flagged during Unit B's code-gate review"). The rewrites below drop it; every dropped fact is already in the cited review or HISTORY, which I confirmed exists for each cite except where noted.

## Decisions on the questions in the brief

- **A (delivered/fixed).** None. Checked each against code; every defect still reproduces or reads as described. Nothing needs a HISTORY check-before-delete.
- **D (live constraint → spec).** None moved. Closest candidates: #6's "canonicalized substring" (already DESIGN §6.5; the *limit* is a known defect with a design task, i.e. backlog-shaped), #3's null behaviour (already DESIGN §6.4 — and is the *defect* statement the item points at), #15's current behaviour (DESIGN §3.5). Each already has its constraint half in the spec; only the *forward* half remains here.
- **F (duplicates).** #3 is **not** a duplicate: DESIGN §6.4 records current behaviour, the BACKLOG item records the deferred fix + trigger; the defects section does **not** list it. #11 and #13 both involve `report.py` banners/verdicts but are different mechanisms (verdict wording vs. attempted count). No merge proposed.
- **#12 — rewrite or delete?** It is "not a live bug", the message is accurate, and the stated remedy is infeasible as written. Deleting would be a discard of a nit with no record; I propose **rewrite to one sentence** (below) rather than delete, since the nit is real and cheap; if the owner prefers a leaner backlog, deleting it loses nothing load-bearing. Low stakes, decided.

## Replacement text (verbatim)

### 3 — replaces lines 21-35

```markdown
- **`tools/exec.py`'s `order_by` sort raises an uncaught `TypeError` on a `None`-valued sort key**
  (DESIGN §6.4). No shipped golden item reaches it (every `order_by` item sorts on a never-null
  `Product.price`), but a sparse property in `tables.json`'s catalog half would. The fix is deferred
  to the next deliberate `packVersion` bump of `nlq-structured-query`: the file sits inside the
  content-hashed pack, so any edit changes `contentHash` (`8f975842`) and unpairs future runs from
  the 18 recorded `1.2.0` runs. It orders nulls the way FalkorDB does (last ASC, first DESC;
  live-verified, including the `WITH DISTINCT … ORDER BY` shape `querygen.compile` emits) and never
  raises `SchemaViolationError` — `compile_and_execute` does not raise past validation. The spec
  is `tests/test_tools_exec.py::test_order_by_a_sparse_property_places_nulls_the_way_falkordb_does`,
  `xfail(strict=True, raises=TypeError)`; the marker comes off with the fix. A test with tied keys
  or several nulls compares each tie group as a set: FalkorDB's order among ties is not mirrored
  between ASC and DESC (`docs/reviews/s4-gate-followups.md`).
```

### 4 — replaces lines 36-48

```markdown
- **`report.py`'s lucky-pass labelling (†/‡) covers the descriptive k/n cells only.** If a pack
  ever makes an `exactMatchBy*` metric a verdict metric, everything downstream goes unlabelled:
  `compare_report`'s verdict text and Holm row, `rank_report`'s reference-anchored family, and the
  `<!-- rank-report -->` marker that `scripts/consolidate_sweep_reports.py` carries into a sweep
  consolidation. No shipped pack reaches it (`nlq-structured-query`'s only verdict and headline
  metric is `layer1ExactMatchRate`). Whoever ships such a pack labels those sections (one line under
  the heading reusing the † footnote) or accepts the limit knowingly. Two polish items in the same
  file: a headline-only `rank_report` never names a stored-vs-placed `luckyPassCount` disagreement
  (suffix the value in `_render_scorer_counts` with `‡`), and
  `_LUCKY_PASS_DISAGREEMENT_FOOTNOTE` is worded for one arm though `{arms}` can list several
  (`docs/reviews/s4-gate-followups.md`).
```

### 5 — replaces lines 49-59

```markdown
- **`results.Outcome` (`results.py:38`) omits `"unrunnable"`.** Every scorer builds
  `ItemResult(outcome="unrunnable", …)` on a no-response call (`classification.py`, `extraction.py`,
  `grounding.py`, `retrieval.py`, `agreement.py`, `toolcalls.py`, `runner.py`), and DESIGN §6.1 lists
  it as an outcome, but the `Literal` admits only `pass`/`fail`/`n_a`/`parse_failure`. No type checker
  runs in this repo (no `mypy`/`pyright` config), so nothing flags it. Widening is one line, but first
  sweep every `match`/`if` chain over `Outcome` (at least `report.py` and `results.py`'s aggregation
  helpers) for one that silently assumes four members.
```

### 6 — replaces lines 61-75

```markdown
- **`checklist_pass`'s `mustContain`/`mustNotContain` check is plain, canonicalized (case/whitespace)
  substring matching, so ordinary morphological paraphrase fails it** (`scoring/grounding.py`;
  DESIGN §6.5). Live on `cr-11`/`cr-17` of `packs/chat-responder-grounded-answers/items.jsonl`:
  `"4 retries" in "4 retry attempts"` and `"30 minutes" in "30-minute"` are both `False` (different
  stems; a hyphen splitting one token). Word-boundary tokenization alone fixes neither; a real fix
  needs stemming or hand-rolled normalization, with its own held-out fixture set to bound the new
  false positives (as the `_ABSTENTION_MARKERS` widening needed its hedge-then-answer guard). A
  `data-scientist`-scoped design task, not a one-line change, and scope creep for a
  zero-runtime-dependency component if folded into a smaller fix
  (`docs/test-reports/small-model-benchmarking-s7-report.md`, mechanism 2).
```

### 9 — replaces lines 87-91

```markdown
- **No paired significance test exists for a turn-pooled exploratory metric** (e.g.
  `duplicateWithinTurn`/`duplicateCrossTurn`): `report.py` has no interval for a metric whose analysis
  unit is a turn rather than the role's conversation/item unit (DESIGN §10 prints it without one), so
  such a metric can be reasoned about but not measured
  (`docs/test-reports/small-model-benchmarking-s6-report.md`).
```

### 10 — replaces lines 93-104

```markdown
- **`validate_pack` raises instead of reporting when a declared data file is not yet authored.**
  `_answerability_stamp_problems` (`packs.py`, the `nlq-generator`-only `"answerable"` check) iterates
  `pack.iter_items()` unguarded, so a manifest declaring `data.items` before `items.jsonl` exists
  makes `validate_pack` raise `FileNotFoundError` out of both `cli.py` callers (`validate` and
  `run`'s pre-flight), breaking its own contract that `[]` means valid and anything else is a problem
  string. Reproduced on a copy of `packs/nlq-structured-query` with `items.jsonl` removed. The two
  sibling axes that read a data file are guarded (`_row_count_identity_problems`,
  `_clean_through_turn_h_problems`: `try/except (OSError, json.JSONDecodeError)` → one problem
  string, pinned in `tests/test_packs.py`); the fix is the same shape plus its mirror test. Low risk:
  only a pack mid-authoring hits it.
```

### 11 — replaces lines 106-120

```markdown
- **`stats.verdict()`'s "is better than" wording assumes a higher raw rate is the better outcome —
  untrue for `falseAdvanceRate`/`falseSuspendRate`.** `classification.score_item` stores the error
  itself as the metric's success (`counts["falseAdvanceRate"] = int(advanced)` on a `clear_suspend`
  item; `counts["falseSuspendRate"] = int(not advanced)` on a `clear_advance` one, documented at
  `classification.py:509-512`), so lower is better, while `stats.verdict`'s winner selection
  (`stats.py:1400-1405`, `diff >= 0`) is polarity-blind and `compare_report` prints its text
  unmodified (`report.py:3134`). A distinguishable guard-judge verdict on either metric would print
  "X is better than Y" when X has the higher error rate. `rank_report` already corrects for it
  (`_LOWER_IS_BETTER`, `_polarity_corrected`). Read-verified, never observed live: no shipped
  `reports/guard-judge-*.md` has a distinguishable verdict on either metric. Mechanism and a scoped
  fix: `docs/plans/small-model-catalog-sweep.md` §2.4 and §6.
```

### 12 — replaces lines 122-134

```markdown
- **`report.rank_report()`'s reference-not-found `ValueError` appends `" and session"` whenever any
  other run is stored** (`report.py:1843-1846`, `+ (" and session" if runs else "")`), not only when
  a `--session` filter hid the reference. Accurate but misleading wording, not a live bug. A
  one-line fix needs the CLI's `session` threaded into `rank_report` (it has no such parameter;
  `cli._select_rank_arms` applies the filter) or the suffix dropped; do it the next time this
  function is touched.
```

### 13 — replaces lines 136-173

```markdown
- **`rank_report()`'s zero-`n` exclusion banner double-counts items when two verdict metrics share a
  run and both have `scoreable={}` items.** `_attempted_for_metric` (`report.py`) counts an item
  toward `metric` when it declares `metric` **or** declares nothing (`scoreable == {}`; the second
  clause is what lets it read `nlq-structured-query`'s real parse-failure shape). But `{}` is also
  every scorer's **no-response** shape (`classification.py:586-596` returns before the tier→metric
  map is consulted). For guard-judge, the only pack with two verdict metrics on one `items` list, a
  run whose items from both tiers time out prints "70 item(s) attempted" under **both**
  `falseAdvanceRate`'s and `falseSuspendRate`'s banners where the true split is 40/30. Reproduced
  with a fixture built from `classification.py`'s own no-response `ItemResult`; never observed live
  (none of the 19 stored `guard-judge-understanding-*` runs has an empty-`scoreable` item).
  The fix is not to guess which tier an empty item belonged to (that is not in the stored record,
  and inventing it breaks "declared, never inferred", `AGENTS.md`): once more than one verdict
  metric shares a run's items and an ambiguous count exists, the banner prints the items that
  explicitly declared `metric` plus a separate, named count of items that recorded no response for
  any metric and cannot be attributed to it
  (`docs/reviews/small-model-catalog-sweep-impl.md`, Defect 1 fix, verification round 2).
```

### 14 — replaces lines 175-183

```markdown
- **The `nlq-structured-query` pack no longer mirrors falkor-chat's production parse** (filed from
  `falkor-chat/docs/reviews/salesperson-single-model.md` P5-4). Since falkor-chat `7f3ad0e5`,
  `QueryGraphDataTool` merges repeated list keys (conflicting scalars rejected) via
  `querygen.merge_duplicate_keys` and fills an absent `returns` via `querygen.default_returns`. Here,
  `tools/exec.py` still rejects a missing `returns` and the model JSON is parsed last-wins by
  `scoring/extraction.py` through `classification.extract_own_line_json_object` — **outside the
  content-hashed pack**, so the repeated-key half needs a scorer edit as well as a pack bump. The
  pack under-scores a model that repeats `filters` or omits `returns` (ministral on `nlq-12`).
  Decide between a pack version bump (plus that scorer change) that copies both normalizations, or
  a `PROVENANCE.md` note that the pack scores the stricter pre-K-066 contract. Copy, don't import:
  nothing under `modelbench/` or the pack may read from `falkor-chat/` at runtime.
```

### Defects section intro and tail — replaces lines 196-197 and 205

```markdown
Current behavior the live spec (`docs/DESIGN.md`) is meant to state, each until fixed; the last two
are not yet stated there (spec sentences owed: the exit-5 text in §3, the start-of-run read in §3.1
or §4.2).
```

```markdown
Done when a defect is fixed or removed, with the spec section updated in the same change.
```

(The "**Done when:**" is demoted from a list bullet to a plain closing sentence so it is not mistaken for a sixth defect.)

## What I could not fully verify

- D5 was read from code (`runner.py:1887/1890` outside any `try`; `lmstudio.py` raises on no response; `cli.py:749-752` catches only `RunRefused`); not reproduced — it needs a mocked/real LM Studio timeout. Confidence: inferred, high.
- Item 11's "no shipped `reports/guard-judge-*.md` reaches a distinguishable verdict" and item 8's 69/81 were not re-derived; they are carried from the item and the S6 report respectively.
- Items 1, 2, 7, 8, 15 are "still open" on negative evidence (an absence of code); I ran the greps named in the table, not an exhaustive read of the call graph.
