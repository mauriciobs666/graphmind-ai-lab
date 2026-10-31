# Typed Decision Models — Implementation Plan, Over-Limit and Loading Amendments

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 1.2 · **Extends:** `docs/plans/typed-decision-models2.md` · **Reviews:** `docs/reviews/typed-decision-models3.md`

*2026-10-31 — v1.2: answers review Pass 1; declined: m3's shared `policy-bounds.jsonl` (optional, and each copy's own refusal test pins it), and the body document's ingestion-status nit (no plan content).*

## 1. Goal & scope

**Goal.** Absorb the six "Handed to architect" items of `docs/plans/typed-decision-models-ml.md`
v1.6 ("the note"): over-limit inputs, their counts, chat timeouts, explicit loading, fitting, and
one offline test.

**How to read it.** `docs/plans/typed-decision-models2.md` ("plan2") and
`docs/plans/typed-decision-models.md` ("v1.3") stay authoritative. Each section below names the
plan2 section it amends. Where they disagree on a named point, this document wins.

It also rules the two open questions of the U1c code review
(`docs/reviews/typed-decision-models-impl.md` Pass 2): the hand-off-model rule (§3.2) and the
home of `eval_set_hash`'s line format (§3.8).

**Out of scope.** Everything else. The over-limit treatment is the stakeholder's decision of
2026-10-31 (note §Q4.3) and is not reopened.

## 2. Context & findings

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded, and the falkor-chat facts below were read from source.

Facts this plan rests on:
- **LM Studio.** The limit, the JIT eviction and the missing `/api/v0/decisions` route are in
  `ws:agent-team` document `480ef825c67d4f9c8e177b16174791c7` ("the probe document") and note
  §Q4.3/§Q6. Document `a1d388f459b74c4db5c1d13b2a22e66a` ("the body document") holds one 500
  body verbatim: 232 bytes of JSON, the text inside `error.message`. Neither records
  `lmStudioAppVersion`.
- **Loads (coordinator's read-only GETs, 2026-10-31).** `GET /api/v1/models` lists
  `loaded_instances[{id, config{…, physical_batch_size, …}}]`. It shows no load timestamp or
  instance nonce, so a load's identity is observable only to whoever performs it. LM Studio
  documents `POST /api/v1/models/load` (`{model, echo_load_config, …}` → `{instance_id,
  load_time_seconds, status, load_config}`), and an unload route; neither is probed here.
- **model-bench.** `lmstudio._raw_post` puts an HTTP error body only into the truncated message;
  `LMStudioCallFailed` carries `status` alone. Nothing in `modelbench/` retries.
- **falkor-chat.** `transport.make_http_transport` raises `ProviderCallError(kind="http",
  status=…, detail=<error body, truncated to 500 characters>)`; the marker starts at character
  65. `retry.is_transient` classes every 5xx as transient. `falkorchat/decisions.py` does not
  exist yet (U9).
- **decision-models** has its own fit-only client, `decisionmodels/lmstudio.py` (v1.3 §3.4). It
  cannot import model-bench (FR-8).

## 3. Design amendments

### 3.2 Candidate rule and chat timeouts (plan2 §3.2)

**Hand-off model rule, now one-way.** `Candidate.load` refuses a policy that can hand off (a
non-empty band, or `minConfidence > 0`) with `handoffModel: null`. It **accepts** an empty band
with a hand-off model: an overflow-only composite, whose only hand-offs are over-limit items.
- That composite is the runtime's own shape: a configured `handoffModel` takes over-limit inputs
  whatever the band. Refusing it would force the fit to drop the hand-off model whenever its best
  band is empty, and so measure a different composite from the one deployed.
- A candidate with `handoffModel: null` stays valid; its over-limit items follow note §Q4.3's
  "No hand-off model" rule (§3.3).
- U1d owns the change to built U1c code: the `_policy_problem` condition, and the flip of
  `test_candidate_load_refuses_an_empty_band_with_a_hand_off_model` into an acceptance test.
- U9's `DecisionPointConfig` validation applies the same one-way rule and refuses bounds outside
  `0 ≤ noBelow ≤ yesAtOrAbove ≤ 1` (or `0 ≤ minConfidence ≤ 1`). U6's `apply_policy` copy raises
  the same refusal. Each is pinned by its own suite.

**Chat timeouts.**
- The hand-off chat call never uses `decisionTimeoutSeconds`, an over-limit item's included.
- **No retry on any decision-point path, in either component.** The runtime has none (note
  §Q4.3), so the bench adds none. In falkor-chat, `DecisionClient` and the three fronts must
  never be wrapped in `call_with_retry`, because `is_transient` would retry the over-limit 500.

### 3.3 Over-limit decision calls (plan2 §3.3)

Note §Q4.3's over-limit row is the rule; this is its mechanics. It sits **above** plan2 §3.3's
kept decision-call failure, which no longer covers an over-limit call.

**Recognition, one rule in three clients.**
- **Marker.** `OVER_LIMIT_MARKER = "too large to process"`. A call is over-limit iff its status
  is `500` **and** its error body contains the marker (case-sensitive). Any other 500 stays a
  failure.
- **model-bench.**
  - `LMStudioCallFailed.__init__(message, *, status, body)`: `body: str | None` is required, like
    `status`. `_raw_post`'s HTTP-error rung passes the full error body decoded as UTF-8 with
    `errors="replace"` (`""` when the error-body read failed). Every other raise site passes
    `body=None`.
  - `decisions.is_over_limit(status: int | None, body: str | None) -> bool`, pure, beside
    `OVER_LIMIT_MARKER`.
- **falkor-chat.** `decisions.is_over_limit(status, body)` with the same signature, applied to
  a `ProviderCallError` of `kind == "http"` as `(exc.status, exc.detail)`.
- **decision-models.** `decisionmodels/lmstudio.py` carries its own `is_over_limit` (§3.5).
- **Vectors.** A new family, `model-bench/tests/fixtures/decision-contract/over-limit.jsonl`, has
  rows `{status, body, overLimit}`, derived by hand from this section:
  - the body document's body at 500 (true), and at 400 (false);
  - that body with the marker upper-cased (`TOO LARGE TO PROCESS`) at 500 (false);
  - another 500 body; the probe document's `/api/v0` HTTP-200 error body; `""`; `null` (all
    false).

  Each client asserts against a byte-copy, under v1.3 §3.2.2's guarded equality test.

**The bench's item, `handoff` surface.** On an over-limit decision call the item is handed off
exactly as a policy hand-off (v1.3 §3.3 steps 4–6). Its record, by what the hand-off chat call
does:

| Field | Hand-off succeeds | Hand-off fails, `fail-closed` role | Hand-off fails, `withhold` role (guard) |
|---|---|---|---|
| `detail.reason` | `over-limit` | `over-limit` | `over-limit` |
| `detail.decidedBy` | `handoff` | absent | absent |
| `detail.failedCall` | absent | `handoff` | absent |
| `calls` | `(chat,)` | `()` | `()` |
| `wallClockMs` / `withheldFor` | item clock (failed call + hand-off) / `None` | item clock at the raise / `None` | `None` / `"timeout"` or `"no_response"` |
| answer | the hand-off LLM's | fail-closed | none (dropped pairwise) |
| `handoffShare` | counts as handed off | counts as not handed off | not attempted |
| `overLimit` | k and n | k and n | k and n |
| `handoffCallFailures` | — | counts | not applicable (guard) |

Every row has `detail.probability = None` and `detail.inputTokens = None`, and no gap check
runs.

**The bench's item, `decisions` surface** (no `handoffModel`). It takes plan2 §3.3's kept
decision-call failure in full (`failedCall="decision"`, guard concordant, new abilities fail
closed), plus `detail.reason = "over-limit"`.

**Scorer hook delta.** The keyword `reason` (no default; `None` writes no key) joins plan2 §3.3's
set on the v1.3 §3.3 `ItemScorer.score_answer` Protocol hook, on `classification.score_answer`
(U1d, the old pack's `detail` staying byte-identical) and on U4's `agreement.py` scorer.

### 3.5 Fitting (plan2 §3.5)

- **Client.** `decisionmodels/lmstudio.py`'s `decide` returns `OverLimit(wallClockMs)` on a call
  that `is_over_limit` recognizes. It is asserted against the byte-copied `over-limit.jsonl`.
  Every other failure raises.
- **Ladder.** For a fit with a hand-off model, an over-limit calibration item is treated per note
  §Q2.1 in both ladders. Its `t̄_{L|H}` term is `OverLimit.wallClockMs` plus the hand-off call.
  For a `handoffModel: null` fit, which U6 produces whenever it emits a decision-only candidate,
  the item is scored by note §Q4.3's "No hand-off model" row: concordant on the guard, the
  fail-closed answer on new abilities, latency `OverLimit.wallClockMs`.
- **Fit report** (not the manifest) gains, per instruction text:
  - the over-limit count `k/n`, since a longer text forces more hand-offs (note §Q2.3);
  - on choice points, the calibration AUC of `max(probabilities)`, labelled exploratory (note
    §Q2.5).

  Over-limit items are excluded from both AUCs.

### 3.6 Counts (plan2 §3.6)

- **Metric.** The candidate arm's aggregates gain
  `BinaryMetric("overLimit", k, attempted, unit="item")` on both decision surfaces.
  `decisionCallFailures` counts `failedCall == "decision"` items **without**
  `reason == "over-limit"`.
- **Per-replicate block.** `overLimit k/n` prints beside `handoffShare` and
  `decisionCallFailures`. It also prints per cell: beside each FR-1 cell line on the guard (U3c),
  and beside each FR-10 class line on new abilities (U4). `overLimit 0/n` still prints.
- **Pairing.** `decision_stats.pair_items` pairs a handed-off over-limit item as any hand-off,
  with `y_i` = its `wallClockMs`. A `decisions`-surface one pairs as plan2 §3.6's guard decision
  failure.

### 3.7 falkor-chat and explicit loading (plan2 §3.7)

**Over-limit at runtime.** Each front checks `is_over_limit` on a decision-call
`ProviderCallError` before its error path:
- **Guard** (`_DecisionFrontedGuardJudge`): it calls `llm_judge(...)` and annotates
  `decidedBy="llm-handoff"` with no probability. The trace renders `[llm-handoff]`.
- **Join-in and message type:** the configured hand-off LLM answers, as on a policy hand-off.
  No fail-closed.
- **Every point:** `record_decision(point, "llm-handoff", None, reason="over-limit")`.
- **No `handoffModel`, or a measured-composite bypass:** the point takes its existing
  error path, with `reason="over-limit"` kept on the log line.

**Explicit loading in the bench: the runner loads (note §Q6).** LM Studio exposes no load
identity, so the evidence that a run is a fresh load is the runner's own act.
- **Load step.** On a decision-point pack, `run_pack` captures `residentModelsAtStart` as today.
  Then, before the warm-ups, it loads every model the arm uses that is absent from that snapshot,
  using `LMStudioClient.load_model(model, *, timeout_s=firstCallTimeoutSeconds) -> LoadedInstance`
  over `POST /api/v1/models/load {"model": id, "echo_load_config": true}`. The arm's models are
  the chat model, the decision model, or both. A load failure refuses the run (exit 3). The
  warm-ups then find the models loaded, so nothing is JIT-loaded by the run. A model resident at
  start is used as found.
- **Classification.** Plan2 §3.6's classifier is unchanged. Under U7, replicate 2 starts with the
  decision model absent, so the run is `load`.
- **Recorded.** The decision-surface fingerprint gains `decisionInstanceConfig`: the decision
  model's `loaded_instances[0].config` from `GET /api/v1/models` after the load step, verbatim
  (`_PRESENT`, `None` if absent). If it carries `physical_batch_size`, the over-limit threshold is
  a recorded run parameter. Raising that value is a different served artifact (note §Q7), so it
  is out of scope. This is additive in schema 1, under plan2 §4.2's U1e ordering rule.
- **Co-residency.** After both warm-ups, both models of a `handoff` run must be resident, or the
  run refuses (exit 3, an environment state) before any item.
- **Precondition.** Probe U2p (§4.2) must confirm the load route first. If it fails, the load
  step returns to `architect`, and the rest of U2 proceeds.

**Deployment (U9).** `SERVER.md` §1.8 states the explicit-loading precondition beside the
`timeoutSeconds` copy rule. It is documented, not checked at startup: residency can change after
startup, and `record_decision` latency shows a violation. v1.3 R-9's storefront follow-up
inherits it.

### 3.8 `eval_set_hash` line format (v1.3 §3.5)

This document is the format's home; it is stored data, and K compares it across runs. Each
`eval`-stratum item gives one line, `input_hash(item) + ":" + json.dumps(item["expected"],
sort_keys=True)`. The lines are sorted, each is terminated by `"\n"` (the last one included),
and they are concatenated. The hash is sha256 of that UTF-8 text, in lowercase hex. This is
`decisions.eval_set_hash` as built; changing it changes every stored `evalSetHash`.

## 4. Units

### 4.1 Amended briefs

Each brief **adds to** the plan2 §4 brief and the v1.3 brief for the same unit.

- **U1d.**
  - §3.2's one-way rule in `_policy_problem`, with the test flip (built U1c code).
  - §3.3's `LMStudioCallFailed.body` at every raise site (built U1a code).
  - `OVER_LIMIT_MARKER`, `is_over_limit` and `over-limit.jsonl`.
  - The `decisions`-surface over-limit item, and `reason` on the Protocol hook and on
    `classification.score_answer`.
  - The `overLimit` metric and the `decisionCallFailures` exclusion (§3.6).
- **U2.**
  - The `handoff`-surface over-limit records (§3.3's table).
  - §3.7's load step, `load_model`, `decisionInstanceConfig`, and the co-residency refusal.
  - `overLimit` on the `handoff` surface.
- **U1e.**
  - `tests/fixtures/lmstudio/decisions_over_limit_500.json`, holding `{_provenance,
    lmStudioAppVersion: null, status: 500, body}`. `body` is the body document's 232 bytes
    verbatim, as one string; a test pins its UTF-8 length at 232. `_provenance` says the version
    was not captured at probe time.
  - §5's offline test, plus `test_live_over_limit_body_is_recognised` (`-m live`). It posts an
    input over 512 tokens through the real `decide` and asserts `is_over_limit`.
  - Plan2 §4.2 U1e's `/api/v0` branch, runtime-field fingerprint delta included, is void: the
    route does not exist (§2). So is plan2's R-14.
- **U3c.** §3.6's per-replicate and per-cell `overLimit` prints, and `pair_items`' two
  over-limit rules.
- **U4.** The per-class `overLimit` print, and `reason` on `agreement.py`'s scorer. A new-ability
  over-limit item is scored on the hand-off LLM's answer, never fail-closed.
- **U6.** §3.5's client, the over-limit handling in both ladders, the fit-report additions, and
  §3.2's bounds refusal.
- **U7.** These amend plan2 §4.1 U7's procedure.
  - **Confirm** (first). Run `test_live_over_limit_body_is_recognised`; on a fail, halt, because
    the method has changed. Record the result and the attested `lmStudioAppVersion` in
    `model-bench/docs/HISTORY.md`. No raw body is captured: the fixture is verbatim already.
  - **Step 3.** Unload the decision model only (the unload route or `lms unload`) and confirm it
    is absent. Do not reload it: replicate 2's runner does (§3.7).
- **U9.**
  - falkor-chat's `is_over_limit` with its byte-copy of `over-limit.jsonl`.
  - §3.2's one-way config rule and bounds refusal.
  - `SERVER.md` §1.8's explicit-loading precondition.
- **U10, U11, U12.** §3.7's over-limit path in each front, each with its `record_decision` line.

### 4.2 New unit and order

**U2p — load-route probe** (`teco`, live and human-approved, because it loads a model on the
shared GPU). Against `http://localhost:1234`, recording every status and raw body in
`ws:agent-team`:
1. `GET /api/v1/models`.
2. `POST /api/v1/models/load` with `{"model": "laya", "echo_load_config": true}`.
3. `GET /api/v1/models`: laya's `loaded_instances[0]`.
4. If qwen is absent, `POST /api/v1/models/load` with `{"model": "qwen/qwen3-4b-2507"}`. Then
   one `POST /v1/decisions` with the probe document's request, and `GET /api/v1/models`.
5. `POST /api/v1/models/unload` with `{"instance_id": <step 2's>}`, then `GET /api/v1/models`.

**Done** when four answers are recorded:
- step 2 loads a decision model;
- step 3's `config` carries `physical_batch_size`;
- both models stay resident through step 4's decision call;
- step 5 leaves laya with no instance.

Only a "no" on step 2 changes the plan (§3.7's precondition). A "no" on step 3 leaves
`decisionInstanceConfig` without the batch size.

**Order.** Plan2 §4.3's graph gains `U2p --> U2`. `decisions.py` serializes U1c → U1d → U6a, after
U1c's review fix round.

## 5. Test strategy additions

These rows add to plan2 §5.

| Unit | Behaviour |
|---|---|
| U1d | An empty band with a hand-off model loads; a non-empty band with `handoffModel: null` refuses. `_raw_post` given an `HTTPError(500)` puts the full body on `body`, and a failed error-body read gives `""`. Every `over-limit.jsonl` row passes. Mutating the marker, dropping the status check or case-folding the body each reddens a row. A decision-only over-limit item gets `failedCall="decision"` and `reason="over-limit"`, counts in `overLimit`, and does not count in `decisionCallFailures`. |
| U2 | One item per column of §3.3's table, each asserted field by field and in the three counts. The hand-off chat call never gets `decisionTimeoutSeconds`. A 500 without the marker stays a kept decision failure. A decision model absent at start gets one `load_model` call before the warm-up; a resident one gets none. `decisionInstanceConfig` is recorded. A failed load, or a decision model absent after the chat warm-up, refuses with exit 3 and scores nothing. |
| U1e | **The offline over-limit test.** A fake opener raises `HTTPError(500)` with the fixture's body. Through the real `decide`, `LMStudioCallFailed.body` equals it and `is_over_limit` is true. Through `run_pack` on a guard fixture pack, it gives a handed-off item on the `handoff` surface and a kept decision failure on the `decisions` surface, each with `reason="over-limit"`. The fixture's body is an `over-limit.jsonl` row with `overLimit: true`. |
| U3c | A handed-off over-limit item pairs with `y_i` = its `wallClockMs`. A decisions-surface one pairs as concordant. `overLimit 0/n` prints when there are none. **U7's sequence:** with a fake LM Studio, `run_pack` the incumbent, then replicate 1 (laya absent, runner loads), then a fake unload, then replicate 2 (laya absent, runner loads). `cli._replicate_groups` gives kinds `(load, load)` and `loadCount == 2`. With laya left resident before replicate 2, it gives `same-load` and `loadCount == 1`. |
| U6 | An over-limit calibration item is a hand-off in every band, adds 0 to every `D_c`, and is absent from both AUCs. On a `handoffModel: null` new-ability fit, it scores fail-closed. The fit report carries per-text `overLimit` and, on a choice fixture, the `max(probabilities)` AUC. An out-of-range band refuses. |
| U9–U12 | Each front, given a fake `DecisionClient` that raises one over-limit `ProviderCallError`, calls the client exactly once, reaches the hand-off LLM and logs `reason=over-limit`. A non-marker 500 takes the error path. U9's config refuses a swapped band. |

## 6. Risks

These replace plan2 §6's **R-13**, which was false, and **R-14**, which is void (§4.1 U1e). They
add R-15 and R-16.

- **R-13: forced hand-off, a floor on `s`** (§3.3). Eval may hold no over-limit item, so the path
  is covered offline by U1e's test, by `overLimit 0/n` (§3.6), and in production by
  `reason=over-limit`.
- **R-15: explicit loading** (§3.7). The runner loads every absent model itself, and co-residency
  is enforced. A model already resident at start is used as found, so how it was loaded is the
  operator's responsibility. In a deployment the precondition is only documented.
- **R-16: the marker is one LM Studio version's wording.** A change silently turns over-limit
  into an ordinary decision failure; that fails safe. U7's confirm step is the check.

## 7. Ready to implement

Plan: `docs/plans/typed-decision-models3.md`. It extends `docs/plans/typed-decision-models2.md`,
which extends `docs/plans/typed-decision-models.md`.
- **Next unit:** U1d, after U1c's review fix round. U2p must run before U2.
- **Stated once here:** the candidate rule and timeouts in §3.2, over-limit mechanics in §3.3,
  fitting in §3.5, counts in §3.6, the runtime and loading in §3.7, and `eval_set_hash` in §3.8.
