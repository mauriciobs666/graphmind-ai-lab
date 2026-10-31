# Typed Decision Models — Review of the Over-Limit and Loading Amendments Plan

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/typed-decision-models3.md`

## Pass 1

### Scope & verdict

**Reviewed:** `docs/plans/typed-decision-models3.md` v1.1 (untracked, 2,262 words). It extends
plan2 v1.1 (`docs/plans/typed-decision-models2.md`), which extends v1.3. I checked it against:
- the method note v1.6 (`docs/plans/typed-decision-models-ml.md`): "Handed to architect" items 1–6,
  §Q2.1–§Q2.5, §Q4.3 and §Q6, each read in the note's own words;
- `docs/reviews/typed-decision-models-impl.md` Pass 2's open questions 1–2, and Pass 3's Claim 2
  and open question (the coordinator's added coverage question);
- `ws:agent-team` documents `480ef825…` (the probe) and `a1d388f4…` (the verbatim 500 body);
- plan2 v1.1 §3.2–§3.6, §4.1 U7 and §4.2 U1e;
- built code: `model-bench/modelbench/lmstudio.py` (`LMStudioCallFailed`, `_raw_post`) and
  `decisions.py` (U1c plus its uncommitted review-fix diff); `falkor-chat/server/falkorchat/transport.py`
  (`ProviderCallError`, `_read_error_body`) and `retry.py` (`is_transient`, call sites of
  `call_with_retry`).

**Executed:** I took the body document's text and counted its bytes: 232 with no trailing newline,
233 with one. I then ran plan3 §3.3's six `over-limit.jsonl` rows against the specified predicate
and against a body-lowercasing mutant (m1). I did not run any suite, because no code changed.

**Verdict: needs changes.** There is one blocker, B1. U7's new explicit reload before replicate 2
puts the decision model in `residentModelsAtStart`. Plan2 §3.6 then classifies the run as
`same-load`, so every group has `loadCount = 1` and every bar prints `not assessable`. There is
one major, M1: two rules both apply to an over-limit item whose forced hand-off chat call fails,
and they disagree on what it records.

The rest is sound:
- the six handed items and both impl-review questions each have an owner and a done-condition;
- the third over-limit copy, in `decision-models/`, is a real requirement (premise 1, below);
- the one-way hand-off rule agrees with note §Q2.1 and §Q4.3 (premise 2, below).

**Stopping condition (falsifiable, for a line check of the next revision with no Pass 2).** Accept
when all three hold, and nothing else changes design:
1. **B1.** Plan3 states how a replicate-2 run under U7's procedure is classified `load`. Either
   the runner performs the explicit load itself after recording the model absent at step 3, using
   a mechanism the plan names, or the fingerprint records other load-identity evidence that the
   plan names and plan2 §3.6's classifier reads. A §5 row drives U7's sequence (incumbent;
   replicate 1; unload; reload as U7 specifies; replicate 2) through `cli._replicate_groups` and
   gets `loadCount == 2`.
2. **M1.** §3.3 gives the item record for an over-limit item whose forced hand-off chat call
   fails, for each `HANDOFF_FAILURE_BY_ROLE` value. The record covers `decidedBy`, `failedCall`,
   `reason`, `calls`, `wallClockMs`/`withheldFor`, and how it counts in `handoffShare`,
   `overLimit` and `handoffCallFailures`. §5's U2 row tests one item of each kind.
3. **m1.** `over-limit.jsonl` gains a status-500 row whose marker is in another case, for example
   `TOO LARGE TO PROCESS`, with `overLimit: false`.

Every other finding is either fixed or declined with a one-line reason in the revision note.

CPG: considered, not relevant — there is no `cpg_modelbench`, and the falkor-chat sites (`transport.py`, `retry.py`, the `call_with_retry` callers) were read directly.

**Check 1 — coverage.**

| Note item | Owning unit(s) | Done-condition |
|---|---|---|
| 1 Detection | U1d (model-bench), U6 (decision-models), U9–U12 (falkor-chat), U1e (fixture) | §5 rows U1d, U6, U9–U12, U1e |
| 2 Counts | U1d, U2, U3c, U4 | §5 U1d, U2, U3c |
| 3 Timeouts | plan2 §3.2 (U1d); §3.2 adds the hand-off rule (U2) | §5 U2 "given 180 s" |
| 4 Explicit loading | U2, U7, U9; R-13 replaced | §5 U2 refusal row |
| 5 Fit | U6 | §5 U6 |
| 6 Offline test | U1e | §5 U1e |

Impl Pass 2 open question 1 is ruled by §3.2. The decision-only over-limit item is ruled there
too: it becomes a kept failure via §3.3. Open question 2 is ruled by §3.8. §3.8's text matches the
built `eval_set_hash` exactly (lines sorted, each `\n`-terminated, sha256 lowercase hex).

The seams I found are four:
- **B1**, between plan3 U7 and plan2 §3.6;
- **M1**, between plan3 §3.3 and plan2 §3.3;
- **m2**, between note §Q2.1 and a decision-only candidate's fit;
- **m6**, the `reason` keyword on the v1.3 Protocol hook and U4's scorer.

**Check 2 — premises.**
- **A third copy in `decision-models/`: a real requirement, not invented.** Note §Q2.1 makes an
  over-limit calibration item "a hand-off in every band", entering `t̄_{L|H}` with the failed
  call's time. `fit.py` runs Laya through its own client (v1.3 §3.4), and that client may import
  only stdlib and `decisionmodels` (v1.3 §3.4, FR-8 test). Without recognition, the fit crashes or
  miscounts on the first long calibration item. The copy is held equal by the byte-copied vector
  family under v1.3 §3.2.2, as the other contract functions are.
- **The one-way rule: consistent.**
  - Note §Q2.1 searches every `l ≤ h`, empty bands included. Every over-limit item hands off in
    every band, so an empty-band optimum still has `s > 0` and still needs the hand-off model.
    Refusing that composite would measure a different artifact from the one the fit chose.
  - Note §Q4.3 describes the same runtime: over-limit "hands off to the LLM at every point". Its
    "No hand-off model" row is §3.2's surviving `handoffModel: null` case.

**Check 5 — U2's exit-4 co-residency refusal.**
- It does not collide with plan2 §3.4's end-of-run load rule. The refusal fires before item 1, the
  end-of-run rule fires after the last item, and both read `residency()`.
- **It collides with replicate counting through U7's load step (B1).**
- Two smaller points are in m4 and the nits: the refusal enforces co-residency, not explicit
  loading; and its exit code is 4 where environment-state refusals use 3.

**Check 6 — growth-stops.** v1.0 is not available to me: the file is untracked and has no git
history. So I cannot attribute the ~760-word growth line by line. The revision note names three
changes:
- U1e pins the verbatim body. This is new scope, following the new body document.
- U7 only confirms the marker. This is rework.
- The `/api/v0` branch is void. This is a new fact.

Reading the current text, the restated material is small (m8). Most of the document is new scope
relative to plan2: the six handed items, and the two rulings from the review.

### Findings

**B1 (blocker) — U7's explicit reload makes every replicate 2 `same-load`, so no bar can be
`met`.**
- Plan3 U7: "Load both models explicitly with `lms load`. Plan2's step 3 then reloads the decision
  model explicitly, never by JIT." Plan2 step 3 unloads the model, confirms with `lms ps`, and
  runs replicate 2.
- After an explicit reload, the model is resident when replicate 2 starts. `run_pack` captures
  `residentModelsAtStart` at step 3, before the warm-up (`runner.py:801`). Plan2 §3.6's classifier
  marks a run `load` only if it is the group's first run, or if there is "no
  `residentModelsAtStart` entry whose `id == fingerprint.modelKey`". So replicate 2 is
  `same-load`, `loadCount = 1`, and `combine_replicates` returns `not assessable (only 1 load…)`
  for every bar of every candidate.
- The only sequence that still counts a second load is letting the warm-up load `laya` by JIT.
  That is exactly what §3.7 forbids. Its U2 check would not catch it either (m4).

*Fix (§3.7, U2, U7; plan2 §3.6's classifier).* Make "absent at start" and "explicitly loaded"
compatible. Two ways:
- (a) The runner does the load itself. It records the model absent at step 3, then issues an
  explicit load before the warm-up. The plan must name the mechanism, either an LM Studio REST
  load route (unprobed) or `lms load` via `subprocess`, which is stdlib. Note that
  `fingerprint.py`'s docstring records `lms` as retired from the harness.
- (b) The fingerprint records load-identity evidence that the classifier reads instead.

Either needs a short probe; this is `architect`'s call, possibly with `devops`. Add the §5 row
from stopping condition 1.

**M1 (major) — An over-limit item whose forced hand-off fails is governed by two rules that
disagree.**
- Plan3 §3.3 says the over-limit item gets `decidedBy="handoff"`, `calls=(chat,)` and an item
  clock that covers the failed call plus the hand-off. It counts as handed off. "A hand-off chat
  failure on such an item follows plan2 §3.3's `HANDOFF_FAILURE_BY_ROLE` row."
- Plan2 §3.3 says a kept hand-off failure gets `failedCall="handoff"` and "no `decidedBy`", and
  counts as **not** handed off. Its `calls` holds completed calls only. Here neither call
  completed: the 500 and the failed chat call are both raises (`runner.py:393-399` precedent), so
  `calls=()`.
- Both rules apply to the same item, so the following are each undetermined:
  - on `fail-closed` roles: `decidedBy`, `handoffShare`, `calls`, and whether the item counts in
    `overLimit`, in `handoffCallFailures`, or in both;
  - on the guard (`withhold`): whether the withheld item (`wallClockMs=None`, dropped pairwise)
    still counts in `overLimit`.

  `handoffShare` is AC-2's reported figure, and `overLimit` is printed as "a floor on `s`". An
  implementer has to guess both.

*Fix (§3.3, §5 U2):* one explicit sub-rule. The recommended answers follow note §Q4.3's columns:
- **Fail-closed:** `reason="over-limit"` plus `failedCall="handoff"`, with no `decidedBy` and
  `calls=()`. It counts in `overLimit` and in `handoffCallFailures`, and not in `handoffShare`.
  `y_i` = the elapsed time to the failure.
- **Withhold:** withheld as today, still counted in `overLimit`.

**m1 (minor; executed) — The case-sensitivity pin cannot redden.**
- §5 U1d claims that "lowercasing the body … reddens a row". The verbatim body is already
  lowercase (`…is too large to process…`).
- I ran §3.3's six rows (body document at 500 → true; at 400; another 500; the `/api/v0` 200 body;
  `""`; `null`) against `marker in body.lower()`. All six pass, so the mutant survives.

*Fix (§3.3 vectors):* add a status-500 row whose marker is in another case, for example the body
with `TOO LARGE TO PROCESS`, with `overLimit: false`.

**m2 (minor) — A decision-only candidate's fit handles over-limit items as a composite's would.**
- §3.5 applies note §Q2.1 ("a hand-off in every band") to every fit. A `handoffModel: null`
  candidate cannot hand off: per note §Q4.3's "No hand-off model" row, its over-limit items are
  decision-call failures.
- On the guard the two treatments coincide: concordant, so `D_c += 0`, and the latency is the
  failure plus `x_i`.
- On new abilities they do not. The fit would credit the LLM's answer where eval scores
  fail-closed (join-in `no`; message type, a disagreement). That inflates the calibration `T` and
  can mislabel `fitOutcome`.
- Where a decision-only candidate comes from at all is an older seam (v1.3 U6 builds one hand-off
  candidate per point, while plan2 U7 runs both).

*Fix (§3.5, U6):* state that a `handoffModel: null` fit scores over-limit items by the no-hand-off
row, and say which unit produces that candidate.

**m3 (minor; the coordinator's coverage question) — No copy is required to pin the
policy-bounds refusal.**
- Model-bench's `apply_policy` now refuses out-of-range bands (`_policy_bounds_problem`, the
  uncommitted fix diff). v1.3 §3.2.2's `apply-policy.jsonl` has no refusal row, and impl Pass 3
  Claim 2 rightly kept it so. Neither plan2 nor plan3 names the refusal in U5/U6's or U9's brief.
- **The exposure is real at one copy.** `fit.py` searches only `l ≤ h` (note §Q2.1), so its copy
  never sees an out-of-range band. But enabling a point in falkor-chat is a **human edit** of
  `models.json` (v1.3 §3.7). A swapped `yesAtOrAbove`/`noBelow` would reach an unmeasured,
  order-dependent policy. Plan3 already gives U9 "config validation" for the one-way rule; the
  bounds rule belongs beside it.

*Fix (§4.1 U9, U5/U6):* U9's `DecisionPointConfig` validation refuses out-of-range bounds, with a
test. U5/U6's copy raises the same refusal, pinned by its own suite. A shared
`decision-contract/policy-bounds.jsonl` (policy → `refused`) is the better vehicle, because three
copies now exist, but it is optional. Leave `apply-policy.jsonl` unchanged.

**m4 (minor) — R-15's "enforced in the bench" overstates U2's check.**
- §3.7's refusal tests **co-residency** after both warm-ups, not how each model was loaded.
- An explicitly loaded `qwen` plus a JIT-loaded `laya` passes it. The probe showed that a JIT load
  evicts the previous JIT model; no second JIT load happens here.

*Fix (§3.7, §6):* say "co-residency is enforced; explicit loading is the operator's, and B1's
mechanism". If B1 picks option (a), the runner can enforce explicitness itself.

**m5 (minor) — U7's confirm step has no executable means.**
- "Send one over-limit `decide` call and check that `is_over_limit` recognizes the raise". U7 is
  a no-code unit, and no CLI command issues a single `decide` call.

*Fix (§4.1 U1d or U1e, U7):* add one `-m live` test (for example
`test_live_over_limit_body_is_recognised`) that posts a >512-token input through the real `decide`
and asserts `is_over_limit`. U7's step runs it and records the attested `lmStudioAppVersion` in
HISTORY.

**m6 (minor) — `reason` reaches only `classification.score_answer`.**
- §3.3 adds `reason` (no default) to classification's keyword set. It does not add it to the v1.3
  §3.3 `ItemScorer.score_answer` Protocol hook, which plan2 extended with `failed_call`.
- It also does not add it to U4's `agreement.py` scorer, which must write `reason="over-limit"`
  on new-ability items.

*Fix (§3.3, U4):* name both.

**m7 (minor) — The no-retry test does not reach the code it protects.**
- §3.2 says the three fronts must never be wrapped in `call_with_retry`. U9's test checks only
  that `falkorchat/decisions.py` imports no `retry`.
- The fronts live in `app.py` (`_DecisionFrontedGuardJudge`), the responder (`JoinInDecider`) and
  `background._safe_classify`. A wrapper added at any of those sites passes the test.

*Fix (§4.1 U9–U12, §5):* a behavioural test per front. Raise one over-limit `ProviderCallError`
from a fake `DecisionClient`, and assert the client was called exactly once.

**m8 (minor) — Restated facts and a stale risk.**
- §2's "The limit" bullet restates the probe document and note §Q4.3/§Q6. Cite them.
- §3.2's "Plan2 §3.2's pin (… = 180, the `--request-timeout` refusal and the `compare` backstop)
  stands" restates plan2. Under Extends, an unnamed rule stands by default.
- "The hand-off chat call … uses the 180 s pin" is already plan2's "`run` uses it for every chat
  call". Keep only "never `decisionTimeoutSeconds`".
- Plan2 R-14 ("U1e is late", schema-2 bump for runtime fields) is void with the `/api/v0` branch.
  §6 replaces R-13 but leaves R-14 standing.

**m9 (minor) — The fixture asserts an app version nobody observed.**
- U1e's fixture sets `lmStudioAppVersion` from the current `host.json`, and `_provenance` admits
  the value is reconstructed. The body document says the version "was not captured at probe
  time".
- A plausible but unverified value beside a verbatim body is the defaulted-field shape that
  model-bench's FR-7 refuses elsewhere.

*Fix (§4.1 U1e):* `lmStudioAppVersion: null` with `_provenance: "not captured at probe time"`.
U7's confirm step (m5) records the version recognition was confirmed on.

**Nits.**
- §3.7's refusal is an environment state, which elsewhere exits 3 (warm-up and probe failures,
  `runner.py:799`, `:830-836`), not 4, which is for pack and candidate refusals.
- `LMStudioCallFailed(…, body=None)` defaults where `status` deliberately does not (its
  docstring). A future HTTP-error raise site could omit `body` and silently disable recognition.
  That fails safe, but it is the shape the docstring argues against.
- falkor-chat's `detail` is capped at 500 characters (`transport.py:63`, `:107`). The marker starts at
  character 65, so recognition is safe. "The text is already structured there" (§2)
  should still say "truncated".
- The body document's ingestion `status` is `failed` (embedding or extraction). Its `text` is
  intact, and the bytes match.

### What's solid

- **The body pin is exact.** The body document's text between the markers is 232 bytes of UTF-8
  with no trailing newline, matching both §4.1 U1e and the document's own "232 bytes".
  `_raw_post`'s `errors="replace"` decode round-trips it, so "`LMStudioCallFailed.body` equals
  it" is reachable through the real `decide`.
- **The premises check out against code:**
  - `_raw_post` puts the error body only into a truncated message (`lmstudio.py:653-662`);
  - `ProviderCallError(kind="http", status, detail)` exists (`transport.py:147-153`);
  - `is_transient` classes every 5xx as transient (`retry.py:87-92`);
  - `call_with_retry` is called only from `ingestion.py:138` and `embedding.py:259`.
- **Recognition is one rule** (`status == 500` and a case-sensitive marker), held by a vector
  family that every copy byte-asserts. A wording change fails safe, and R-16 names that.
- **The one-way rule fixes the right contract,** at the right unit (U1d), with the right test flip.
- **§3.8 gives `eval_set_hash` a home** that matches the built code byte for byte, including the
  trailing `\n` and the default `json.dumps` separators.
- **No retry on decision paths** follows the runtime. It closes the trap where `is_transient`
  would retry a deterministic over-limit 500.

### Open questions

1. **`architect` (possibly with `devops`), for B1:** should the runner load the decision model
   explicitly itself (option (a)), or should the fingerprint gain load-identity evidence (option
   (b))? Either needs a short probe: whether LM Studio exposes a REST load route on this build, or
   what `/api/v0/models` reports about a load.
