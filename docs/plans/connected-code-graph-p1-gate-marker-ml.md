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
   - **Repeat-stall conversion (the one exception to "a timeout is transient").** A `TimeoutError` raised inside the engine's `ask` on the same anchor in **k = 3** consecutive stops with no answer recorded between them is converted: the anchor is recorded `{outcome: "engine-error", exceptionClass: "TimeoutError", via: "repeat-stall", stalls: 3}` (a miss, §3) and the run continues. Conditions, all required: (a) the marker carries `stall: {index, count}` (`index` = number of recorded answers when the stop happened, a position, never an id; `count` increments only when the new stop has the same `index` and class, and resets on any other stop or any progress); (b) only a timeout converts, **decided on the cause chain of what the engine's `ask` raised, never on the outer type** (the real `Client.ask` wraps every fault in `StoreError`, so the outer type is never `TimeoutError`; the harness's own `EngineAskError`/`StoreError` wrappers are transparent). Walk `exc.__cause__ or exc.__context__` (the `Client` raises `from None`, which clears `__cause__` and leaves the original in `__context__`; `classify` already follows both), bounded (e.g. 16 links) with a seen-set. Convert iff a builtin `TimeoutError` (the socket timeout) or a redis `TimeoutError` (by class name and module) is on the chain and **no** `ConnectionError` (redis `AuthenticationError` included), `OSError` other than that timeout, or run-wide fault of the §3 table is on it. Connection refused/reset, other `OSError`s, store faults that are not a timeout and interrupts never convert and reset the count (a host or network fault is not the product's). The recorded `exceptionClass` stays `"TimeoutError"`. A server-side `Query timed out` reply is a different row (§3 table: the store answered; `engine-error` at once); (c) **host-health canary:** at each timeout stop the harness, in the same process and before writing the marker, runs one fixed non-held-out ask (a dev-list anchor, or the engine's trivial request, fixed in the prereg) under the same deadline; only if it answers within the deadline does the stop count toward `stall`. A failed canary writes plain `failed:transient` and resets the count; (d) `k`, the deadline and the canary request are prereg fields (pinned in the marker payload, so a resume with a different value is refused by the equalities above; §1.1.5). Why k = 3: one stop is a flaky host, two can be one bad minute; with a per-attempt spurious-timeout probability p the chance of three on one anchor is p^3 (p = 5% gives about 1e-4, an estimate, not measured), and the canary removes the host-wide share of it. Recorded: the conversion appears in `run.json` and the report as its own sub-line of `engine-error` (`repeat-stall`, count, per stratum), never merged into raised-exception engine errors. Rejected: transient forever (a hanging product holds the one-shot list, and the operator's only exit is a new list, i.e. a manual spend of held-out data to dodge a defect); k = 1 (a flaky host converts into a miss, the failure the marker exists to prevent); an operator-set override (a human choosing which anchors count as misses).
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

### 2.1 Stored-form http keys (`{?}`, `{#}`, `?query`): the engine's resolve is wrong, not the gold

**Verified in code.** The rewrite is not the harness's: `cli_request` sends `keyNorm` verbatim, and `ccg/engine/anchors.py::resolve_contract` runs `http_key(key, "consumer")` on it first. That function parses its input as a raw URL, so a stored-form key is mangled (`GET /x/{?}` and `GET /x/{#}/y` become `GET /x/{`; `GET /x?y=1` becomes `GET /x`) and the exact read misses; provider form is the identity. The 14 `{?}` points are in the store, so the gold is resolvable: the engine cannot be asked for a contract point by the key it stores and prints. That is a round-trip defect of the product (the key an answer returns cannot be re-asked), and a real consumer or provider key in the masked form is a legitimate AC-5 anchor.

**Ruling.**
1. **Fix it in the engine, before the gate, on dev evidence.** `resolve_contract`: if the consumer-normalised key has no exact hit, retry the key verbatim as a stored key (exact read, same `kind` filter) before the template fallback. A planted dev test over a store holding `{?}`, `{#}` and a `{}` key must resolve each to its own id; an ordinary key resolves exactly as before (the normalised read stays first). The evidence is the build's store and the rehearsal, not held-out data, so this is not contact and needs no re-freeze beyond the usual `coreTreeHash` pin. The harness stays as it is: it has no product-neutral way to fix this (the contract anchor spec carries only a key), and a harness rewrite that ships a different key than the record's would make the gate ask something other than what the custodian drew.
2. **Rejected: treat such an anchor as a miss or exclude it.** Excluding by key shape after seeing that the product fails on it lets the product shrink its own denominator; scoring it a miss charges the product for a defect found, and fixable, before the marker, and would spend the one-shot list on a known engine bug. Both are choices made on the shape of held-out anchors.
3. **Rejected: declare the list unusable and re-draw as the normal path.** It is the correct *fallback* only if the engine is not fixed before the gate: the refusal stands (counts only, §2.5), the clusters go to dev, and the cost is the ~20 min build plus a stratum shortfall (SQ-P0-1). With the fix this refusal does not occur on this cause.
4. **`?query` keys.** A gold key carrying a query is not a stored form (the build strips it), so its node is absent: `anchor-not-in-graph`, a counted miss under HA-3, as already ruled. No change; the custodian should not draw such keys, and a draw that holds any reports them as that count.

**To state before the held-out run (prereg text):** no resolution exception by key shape; a preflight refusal on any cause is a contact event handled by §2.5; post-marker, an in-graph anchor that does not resolve is a harness error (§2, last paragraph), never a miss.

**Stakeholder decision:** not needed (a defect fix against the live contract and rules already ruled). The engine owner owns the change; the gate does not start until the planted test passes and the dry-run, rerun on a built store holding those key shapes, exits 0.

## 3. `blocked` vs `fail`, and what is spent

**Vocabulary** (confirmed): `stopped` (C5 stamp or void), `fail` (at least one miss; triggers EP §12.1 steps 2–3), `blocked` (the verdict cannot be given or cannot be a pass), `pass`, plus the re-gate outcomes.

**Precedence:** `stopped` > `blocked` by a **harness error** > `fail` > `blocked` by an **engine defect or a listed masked comparison** > re-gate outcomes > `pass`.
- A harness error (EP §5.2.1 HA-1 (e), (g), (h); a translation fault; a store error) makes the item **neither hit nor miss** (`outcome: "error"`, not scored). EP says it "blocks the gate verdict", i.e. any verdict. No recall is claimed and §12.1 does not start: a fail procedure must not run on a result with unscored items. The misses found are still listed.
- Engine defects (HA-2 mismatch) and masked comparisons (HA-5) block a pass only (EP §5.2.1, rows HA-2, HA-5): a miss alongside them is `fail`.
- **An engine exception on one anchor is a miss** (`outcome: "miss"`, reason `engine-error`), tier `fail`. Why: EP §5.2.1 sets "neither hit nor miss" only where the harness cannot decide hit or miss (HA-1 (e), (g), (h): an answer it cannot replay or interpret). A raised exception is an answer the harness *can* judge: it is not a hit. It is the same class as HA-3 "not resolved: a miss", and a product that cannot answer a held-out anchor has missed it. Scoring it neither would let a product shrink its own miss count by crashing, and would turn a product crash into `blocked`, which is cheaper than `fail` only in appearance (the marker is spent either way, §3 below).
  - **Which exceptions.** Engine iff raised from inside the `ask` call after the harness built and validated the request, and neither in the narrow transient set (connection refused/reset, timeout, per §1.1 `failed:transient`) nor a run-wide fault (credential, snapshot, disk, preflight). Classified by *where* it was raised, never by the exception's type or message (the product must not pick its tier). An exception the harness cannot place on one side of that boundary is a harness error (conservative: it blocks a verdict that would start the irreversible fail procedure on an unplaceable crash).
  - **Disposition table (planted-test spec; classification is by the cause chain over the real wrapped exception, never by the outer type, and the exception axis of the test is produced by the real `Client` over a `CypherReader` whose db raises the family on every call, through `make_ask`, `_Persisting` and the marker).** Chain order of precedence: run-wide credential/store-state fault anywhere on the chain > connection/OS/host fault > timeout > everything else.

    | # | raised at the reader's db | disposition | marker / record |
    |---|---|---|---|
    | 1 | redis `ConnectionError` | transient | `failed:transient`, nothing recorded, resumable |
    | 2 | redis `TimeoutError` (builtin `TimeoutError` cause: a hang) | transient on stops 1-2; stop 3 with canary answering: engine-error `via: repeat-stall` (§1.1.3); canary failing: transient forever | as stated; count by position |
    | 3 | `BusyLoadingError` | transient | `failed:transient` |
    | 4 | `OSError` (ENOSPC, quota, fd limit) | transient (host fault) | `failed:transient`, resumable once the host is fixed |
    | 5 | `ResponseError("Query timed out")` | engine-error (the store answered: the product's query exceeded the deadline) | per-anchor entry, `exceptionClass` by the chain's innermost class name |
    | 6 | `AuthenticationError` (and `AuthorizationError`, `AuthenticationWrongNumberOfArgsError`) | **run-wide** | `failed:deterministic`; nothing recorded for the anchor; no miss, no recall claim |
    | 7 | `NoPermissionError` | **run-wide** | as 6 |
    | 8 | `ReadOnlyError` | **run-wide** | as 6 |
    | 9 | any other redis `ResponseError` (e.g. `WRONGTYPE`, a Cypher error) | engine-error (raised inside `ask`, placed on the engine's side) | per-anchor entry |
    | 10 | `StoreError` from `answer.AnswerError` ("the answer does not validate") | engine-error (the product's answer is invalid) | per-anchor entry |

    Why 7 and 8 are run-wide: the engine reader is provisioned as a read-only role (`ccg_raw_<system>`); a `NOPERM` or `READONLY` reply says the store's role or topology is not what the run pinned, which no anchor's content can cause and a product cannot pick. Charging it to every remaining anchor would spend the one-shot list as a product failure, the same reason as a refused credential. A mid-run failover to a replica is also a store-state fact, not the product's. The cost is real and accepted: `failed:deterministic` refuses `--resume` (§1.1.5), so the list is spent in the marker sense, but with no miss recorded and no recall claimed, so §12.1's fail procedure does not start on it. Rejected: making 6-8 `failed:transient` (resumable): the equalities in §1.1.5 pin the snapshot and harness, not the role or ACL, so a resume after a changed credential would run under an unpinned condition. Unlisted exception on the chain: the existing default stands (raised inside `ask` and not run-wide or host: engine-error; not placeable: harness error). The test names any row a later ruling changes as open, not omitted. Add a `NoPermissionError`/`ReadOnlyError` row to `CREDENTIAL_FAULTS` (or a sibling set) and test it over the whole chain, not `type(inner).__mro__`.
  - **Recorded as an answer.** The per-anchor line carries `{outcome: "engine-error", exceptionClass}` (class name only, no message: messages may embed ids). `--resume` never re-asks it, exactly as for any recorded answer; a retry would let a crash be re-rolled into a hit, which is the adaptation step 7 forbids.
  - **Triage (EP §12.1 step 2).** Entered as `linker-engine`, sub-label `engine-error`, one category as the step requires; the method owner may reassign only after reproducing the root cause on dev (`extraction`, `recognizer`, `key-error` with stakeholder confirmation). The cluster moves to dev as for any miss (step 3).
  - **Precedence.** It sits inside `fail`. With a harness error present, `blocked` still wins (nothing is triaged); a masked comparison or HA-2 defect beside it is `fail`, as for any miss. It counts in the denominator of recall as run (a miss), and recall excluding confirmed key errors is unchanged.
  - **Report.** The count of `engine-error` misses, per stratum and per `exceptionClass`; no ids, no messages. Never folded into a generic "errors" count with harness errors: the two are reported on separate lines.
  - **Decision.** No stakeholder decision needed: it applies EP §5.2.1/§12.1 as written. EP §5.2.1 owes one row (HA-6, planted input: an engine stub that raises `ValueError` on one anchor, expected miss `engine-error`, verdict `fail`, resume does not re-ask; the same stub raising a connection error is `failed:transient`; a stub raising `TimeoutError` on one anchor is `failed:transient` on stops 1 and 2 and `engine-error` (`via: repeat-stall`) on stop 3 when the canary answers, and stays `failed:transient` forever when the canary also fails or the failure is a connection error (§1.1.3)).
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
- Preflight (stored-form keys, after §2.1): planted `{?}`, `{#}` and `{}` http keys that exist in the store resolve `ok`; a planted `?query` key is absent and counted as a miss without resolve.
- Preflight: an anchor with two candidates, none, and a different id each refuse before the marker with counts only and no ids on stdout or stderr.
- An engine stub raising a `ValueError` on one anchor: miss `engine-error`, verdict `fail`, line recorded, `--resume` does not re-ask; a harness error beside it gives `blocked`.
- A harness error plus a miss gives `blocked`; a masked comparison plus a miss gives `fail`; a masked comparison alone gives `blocked`.
- Flow items present in the input with the flag false: withheld, never asked (the `ask` stub records no flow anchor); four strata at 8 and `H = 75` give `pass`; flow at 0 does not give `minimums-not-met`.

## 6. Open items

- §1.3 depends on the engine's cache semantics (answer-spec owner).
- The `kept` treatment of clusters in a spent `blocked` run (§3) is the conservative reading; the stakeholder may relax it only through a recorded decision.

## Scripts/wiring scope

Verified from the repo (not held-out): P1 defers shell launch/exec facts, the shell URL source, `LAUNCHES` beyond `launchUnit` and the script hop to P2 (`docs/plans/connected-code-graph-p1.md` close list; `-p1-linking.md` P1-17 (6), lines ~563, 954), while `STRATA_GATED` (`cpg/eval/tools/gate.py:56`; §4 item 3 above) still gates `scripts` and `wiring`. The dev gold mirrors the held-out writers' definitions (dev: http 18, mcp 20, scripts 49, wiring 11), so scripts is about half of the non-flow items and a held-out http+mcp subset is unlikely to reach the pooled `H ≥ 75` (held-out stratum counts not read; the custodian states them).

**Ruling: neither A nor C as stated; narrow by withholding (B, mechanism of §4), and do not spend the held-out list until the stakeholder rules.**
- A is a scope reversal of rulings the stakeholder already took for P2 (reopens P1-10/11b/17, new goldens and answer-spec version, new pre-registered measurements); not a gate-round fix.
- C as stated spends the one-shot list to confirm a known zero: AC-5 says any miss means a fix and a new blind list, so a pre-stated expected-zero cannot make the verdict anything but `fail`, and the run exposes held-out scripts/wiring items for nothing. Rejected.
- B by withholding: `scripts` and `wiring` held-out items are not asked, scored or resolved (like flow, §4 rule 2), so they stay blind for P2, when linking exists. Reported as "not delivered at P1, counted at P2" with counts only.

**Consequences.** (1) It is a requirements change (AC-5 reads "at least 75 ... every one appears", AC-2 wiring half, AC-4), so the stakeholder decides via `tico`/`teco`; the harness change (`STRATA_GATED`, `decide_gate`, the `test_gate_fail.py:371` assertion) waits for it. (2) A http+mcp run can be a P1 **interim certificate** only: never reported as "AC-5 passed". If http+mcp counting items are under 75 it is `minimums-not-met`/"present, not gated" and a held-out spend buys only a bound, not a verdict; then the better use is to defer the held-out gate to P2 (one combined gate over all four strata, one spend) and close P1 on dev plus the rehearsal. (3) Reuse: a spent http/mcp list contaminates those items for P2 (the engine changes after it), so P2 needs fresh http/mcp items anyway.

**Pre-register before any run:** the gated set (`http, mcp`) and the withheld set with the reason; the withheld counts per stratum; the expected-nonzero claim only for the gated strata; the interim label and the wording "AC-2/AC-4/AC-5 scripts and wiring halves: not delivered at P1"; Wilson bounds computed over gated items only (no pooled 159); the 75 and 8 floors stated against the gated set, with the `minimums-not-met` outcome pre-declared as "not a pass, not a miss". Separately and independently of the choice: add a dev-gold pre-gate probe (a gated stratum with 0 dev hits blocks the preflight unless listed as expected-zero), which would have caught this before the rehearsal.
