# Connected code graph — AC-5 gate harness: once-only marker, resolve preflight, `blocked`, flow stratum (P1-32b)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1-32b)

Method rulings for the fix round on `cpg/eval/tools/gate.py` / `gate_run.py`, answering the open questions of `docs/reviews/connected-code-graph-p1-32b-impl.md` (§5, §9, ruling (a), B5). Sources: `docs/plans/connected-code-graph-eval-protocol-ml.md` (EP) §12.1 step 7, §6.2 is **not** the contact rule (the contact rule is `docs/plans/connected-code-graph-p1-ml.md` §6.2, with EP §8.1), EP §5.2.1, §3.7; `docs/plans/connected-code-graph-p1-gates.md` P1-32 step 1. No held-out data was read.

## 0. Rulings in brief

| # | Ruling |
|---|---|
| 1 | The run is one logical run. A run with no complete answer set has no result, so the list has not "failed"; it may be **resumed** under strict equalities. The analyst's marker design is valid with the amendments in §1. |
| 2 | The pre-marker resolve dry-run is acceptable under five conditions (§2). A refusal is not free: the failing anchors' clusters go to dev. |
| 3 | `blocked` and `fail` are distinct. A harness error yields `blocked`, ahead of `fail`. A completed or deterministically-failed run is **spent**, whatever its verdict. |
| 4 | With the flag false, flow is not asked, not scored, not exposed; `pass` is over four strata. Never reported as an AC-5 flow result. |

## 1. The once-only marker and resume

**Reading of EP §12.1 step 7.** "The list that failed" is a list that produced a complete answer set and a verdict that triggered §12.1 steps 1–3. The rule exists to stop adaptation to observed held-out results: a fix, then a re-run on the same items. A run killed before its last answer has no result, scored nothing and showed nothing (the verb prints counts only), so no adaptation can have occurred. Resuming it is the same run, not a second run.

**Why re-asking an anchor does not contaminate.** `ask` is read-only on the store (`ccg_raw_<system>` role, `RoRunner`) and deterministic given (snapshot, request, engine). Its answer is a pure function of those pins. Re-asking the in-flight anchor therefore returns what the crashed attempt would have returned, and nobody observed the earlier attempt's answer. The one channel that could differ is an engine-side cache. §1.3 handles that. A recorded answer is never re-asked, so a retry cannot select a better answer.

**Resume does not cover anything the engine or harness could have changed.** Hence the equalities below.

### 1.1 Rule text (implement)

1. **Key.** `listKey` = sha256 (full 64 hex, in the file name) over the sorted `canon(record)` of the deduplicated records (`match_keys.canon`; sorted by `itemId`). Byte order, blank lines, key order and duplicate records do not change it.
2. **States**, in the marker file (atomic rename on every transition; payload counts and class only):
   - `started` (`O_EXCL`; payload: `recordCommit`, `HEAD`, sha256 of each pinned harness file, `snapshotId`, the four prereg fields, `coreTreeHash`, `startedAt`);
   - `asked` (after the last answer is on disk);
   - `failed:<class>`, written by `main` on any exception, with `<class>` one of `transient` or `deterministic`.
3. **Class.** `transient` is only: a connection or timeout error to the store or engine, `OSError`, `SIGINT`/`SIGTERM`/`SIGKILL` (no `failed` write is possible on `SIGKILL`; the state stays `started`). **Everything else is `deterministic`:** `HarnessError`, `ValueError`, `KeyError`, `TypeError`, a translation fault, an engine answer `{error: …}`, any `ClientError` that is not a connection or timeout error. An unknown exception class is `deterministic` (fail-closed).
4. **Answers.** One JSON line per `(anchor)` appended to `run-<listKey>.answers.jsonl`, `flush` + `fsync` before the next ask. A final line without a trailing newline or not valid JSON is a torn write: discard it and treat the anchor as unrecorded. `run.json` is derived from the file.
5. **`run --resume`** is allowed iff all of the following hold, else `Refusal` (counts only):
   - marker state is `started` or `failed:transient`;
   - `listKey` equals the marker's;
   - the current `recordCommit`, `HEAD`'s pinned-harness sha256s, `snapshotId` (read again from the `:Snapshot` node through the engine reader), the four prereg fields and `coreTreeHash` all equal the marker payload;
   - no `run.json` or score output exists;
   - the resolve preflight (§2) passes again.
   It asks only anchors with no line in the answers file. A `--resume` against `failed:deterministic` or `asked` is refused. `run` without `--resume` against any existing marker is refused.
6. **No removal.** For AC-5 no human may remove the marker (unlike `ac24_smoke.take_marker`). The only way forward from a spent list is a new list (EP §12.1 step 4).
7. **`score` requires `asked`.** It reads the answers file only and stays freely re-runnable.
8. **Before the marker** the run verb must: open the engine reader and read the `:Snapshot` id through it (equal to the `RoRunner` id); build every request from the records (pure translation); run the §2 preflight. A failure refuses before the marker; the list is not spent (but see §2.3).

### 1.2 What the report must disclose

`run.json` and the report's `context` carry: `resumed: true|false`, the number of resumes, the answer count recorded before each resume and the count asked after, the `failed:transient` class names (not messages with ids), and the statement "answers recorded before a resume were not re-asked". A resumed gate is **not** labelled "after adaptation": by the equalities above nothing changed. If any equality fails, the run is refused, not labelled.

### 1.3 The cache

EP §10.4's emptiness check is about not scoring a replayed answer. On a resume the cache may legitimately hold the in-flight anchor's entry. Rule: on resume, `cache_len` may exceed the number of recorded answers by at most 1 (the in-flight anchor), and a `cacheHit` on that single re-asked anchor is tolerated and recorded as `resumedInFlight: true`. Any other `cacheHit` follows the existing rule (`stop`). **Caveat:** this assumes the engine's answer cache serves the identical answer; if the engine cannot show that (answer-spec owner), the in-flight anchor is instead asked with the cache bypass the AC-24 repeat protocol already requires (EP §10.4), and recorded as such.

## 2. Resolve dry-run before the marker

**Ruling: acceptable, under exactly these conditions.** `resolve` returns a status and candidates for an anchor. It computes no link and no impact, so it is not the act that EP §8.1 / `-p1-ml.md` §6.2 calls contact ("computed, asserted or hand-walked a link or an impact"). The gate runner is an exposed dispatch already; the harness is a verifier and no builder reads the output. The residual risk is feedback: a builder who sees "N anchors failed" and fixes against it learns about held-out anchors.

1. It runs **only inside the `run` verb**, in the exposed (gate runner) dispatch. No stand-alone verb, no flag that prints it.
2. It covers every asked anchor that exists in the graph (contract and unit anchors by `ContractPoint` existence; symbol and config anchors resolve through `client.resolve(spec)`). An anchor absent from the graph keeps the `anchor-not-in-graph` miss and is not resolved.
3. Pass criterion per anchor: exactly one candidate, and its id equals the asked anchor id. A zero-, multi- or different-id result is a failure.
4. **Output is counts by status** (`ok`, `none`, `ambiguous`, `id-mismatch`) and a total; no anchor id, no locator, no candidate, no stratum-by-id breakdown. Counts by stratum are allowed (the strata are public figures).
5. **A refusal is a contact event.** The refusal count reaches the builder as feedback on held-out anchors. The custodian therefore moves the failing anchors' clusters to dev (EP §12.1 step 3 mechanics, §9.1 step 1 clusters; logged as `preflight-resolve`, reported as a count), and the run proceeds on a new list. That list was never run, so EP §12.1 step 7 does not apply. A shortfall against the minimums is SQ-P0-1's trigger, never a reason to waive the preflight. Fixes go on dev anchors only.

After the marker, an `AnchorNotResolved` or an id mismatch on an in-graph anchor is a **harness error**, not a miss. The `keyPath` mask (`gate.mask_keypath`) is applied before asking, as the build did.

## 3. `blocked` vs `fail`, and what is spent

**Vocabulary** (confirmed): `stopped` (C5 stamp or void), `fail` (at least one miss; triggers EP §12.1 steps 2–3), `blocked` (the verdict cannot be given or cannot be a pass), `pass`, plus the re-gate outcomes.

**Precedence:** `stopped` > `blocked` by a **harness error** > `fail` > `blocked` by an **engine defect or a listed masked comparison** > re-gate outcomes > `pass`.
- A harness error (EP §5.2.1 HA-1 (e), (g), (h); a translation fault; a store error) makes the item **neither hit nor miss** (`outcome: "error"`, not scored). EP says it "blocks the gate verdict", i.e. any verdict. No recall is claimed and §12.1 does not start: a fail procedure must not run on a result with unscored items. The misses found are still listed.
- Engine defects (HA-2 mismatch) and masked comparisons (HA-5) block a pass only (EP §5.2.1, rows HA-2, HA-5): a miss alongside them is `fail`.
- The report names the `blockers` and counts them. It does not print ids.

**What is spent.** The once-only marker is spent by any run that reached a result or a deterministic failure, **including `blocked`**:
- A `blocked` run holds a complete (or deterministically truncated) answer set, and any fix it prompts is made against what the run exposed. Re-running the list would be adaptation, which is what step 7 forbids. The marker stays; no human removal.
- **Two cures that do not re-run the list:**
  - *Masked comparisons:* lifted when the method owner has reviewed each listed entry (EP HA-5); the same answers are re-scored with the review recorded.
  - *A harness-only fault* (the answers are unchanged and valid; for example the harness lacked a row the spec already defines): fix the harness and **re-score the recorded answers** with the engine unchanged (`coreTreeHash` and `snapshotId` equal). The report is labelled "after adaptation" (`-p1-ml.md` §6.1), and states by re-scoring with both versions that no item that was not in error changes its score.
- **Everything else** (an engine defect, an answer with an `edgeType` outside the allow-list or a class-X hop, a deterministic ask-time failure): the list is spent and the engine changes. The next gate is a **re-gate** under EP §12.1 step 4: the clusters of every errored or missed item move to dev, the other clusters of the list are `kept` (conservatively: their answers were produced), fresh items are added, and the floor F = 35 applies. A `blocked` run is therefore not a free retry.
- The first run's verdict token stays `blocked`; it is reported in the test report as run (EP §12.1 step 1).

## 4. Flow stratum when the flag is false

Source: `docs/plans/connected-code-graph-p1-gates.md` P1-32 step 1: the `flow` stratum is run "**only for a language whose flag is true after P1-34** (otherwise 'flow-unavailable, not closed', held-out flow stays unexposed, flow diagnostics are on dev items only). Counting `H` is 159 beside `H_all` 258 (flow contributes 0 counting items): the floor of 75 is read against 159." Supporting: `-p1-ml.md` §3.1/§3.3 ("held-out `flow` is then gated only on items whose chain stays in flag-true units, and fewer than 8 such items means 'present, not gated'"); EP §3.7 (flow counts toward 75 only when it crosses a unit); `docs/requirements/connected-code-graph.md` decision of 2026-11-19 (closing flow is judged on the held-out flow stratum at the AC-5 gate).

**Rule text (implement).**
1. The flow flag per language comes from the committed `ccg-p1-flow-python-v2-decision` record (via the prereg record), never from a CLI flag. No record means false. Here: false.
2. Held-out `flow` items of a flag-false language are **withheld**: not in the list passed to `run`, not asked, not scored, not resolved by the preflight. Their count (not ids) is reported.
3. `STRATA_GATED` = `http, mcp, scripts, wiring`. `decide_gate` tests `strata_ok` over `STRATA_GATED` only; it must not test `flow ≥ 8` (the current code does, which makes `minimums-not-met` certain).
4. **`pass`** (gate 1) iff: every scored item in the four strata is a hit; `H ≥ 75` over counting items; each of the four `H_s ≥ 8` over all asked items; no harness error, engine defect or unreviewed masked comparison; no stop or void. Re-gates add EP §12.1 steps 4–5 unchanged.
5. Report, beside the §5.3 figures: `flow: "flow-unavailable, not closed"` with the flag and the withheld count; `wiring` labelled "screened, not certified" (exactly at its floor of 8); `H` beside `H_all`; the Wilson lower bounds as `-p1-ml.md` §6.1 gives them (item recall over 159: 0.976). **No flow recall, no flow Wilson interval** and no sentence that AC-5 passed for flow: `pass` is AC-5 over four strata, and FR-8 / AC-25's control half stay recorded as not met for flag-false flow.
6. If the flag is true for a language, flow is asked only on items whose chain stays in flag-true units; fewer than 8 such items is "present, not gated" (`-p1-ml.md` §3.3). Out of scope for this round; the function signature should take `gated_strata` so that case needs no rewrite.

## 5. Acceptance tests (planted inputs)

- Same records as `A,B` / `B,A`, with a trailing blank line, re-serialised with another key order, with a duplicated record: one `listKey`; second `run` refused.
- `SIGKILL` at ask k of n: k lines on disk; `--resume` asks n−k; a torn last line is discarded and re-asked; a harness file edited, a different `snapshotId`, a different `recordCommit` between the two: refused.
- A `ValueError` after the marker: `failed:deterministic`, `--resume` refused; a connection error: `failed:transient`, resume allowed.
- `score` on `started` or `failed:*`: refused.
- Preflight: an anchor with two candidates, none, and a different id each refuse before the marker with counts only and no ids on stdout or stderr.
- A harness error plus a miss gives `blocked`; a masked comparison plus a miss gives `fail`; a masked comparison alone gives `blocked`.
- Flow items present in the input with the flag false: withheld, never asked (the `ask` stub records no flow anchor); four strata at 8 and `H = 75` give `pass`; flow at 0 does not give `minimums-not-met`.

## 6. Open items

- §1.3 depends on the engine's cache semantics (answer-spec owner).
- The `kept` treatment of clusters in a spent `blocked` run (§3) is the conservative reading; the stakeholder may relax it only through a recorded decision.
