# Ministral re-probe — K-027 item 5 (D13 finding 2), post-item-1/item-2 fixes

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-027 (item 5)

## 1. The question and the decision it serves

D13's original probe (`docs/archive/plans/m3-capability-probe-ml.md`) found Ministral-3B beat the
shipped Qwen3-4B at the terminal `post_message` tool call (native `tool_calls`, 3/3 replay) but
scored far worse as the fuzzy-guard judge (fence-fixed advance-recall 0.364 vs Qwen's 0.818). That
probe ran **before** two fixes now shipped: K-027 item 1 (judge parse robustness — fenced/prose
JSON now parses) and K-027 item 2 (the `config.requiredTools` engine-level must-post contract).
K-027 item 5 asks: **re-probe against current code — is Ministral now viable as the guard judge, as
the agent/step model, or neither, and does this change D13's practical relevance?** The caller
(`teco`, coordinating K-027) acts on the verdict directly; nothing here changes shipped behavior —
`config/models.json`, `config/opencode.example.json`, `reference`, and `ws:acme` are all untouched.

## 2. Model identity (first step, live-verified)

LM Studio at `localhost:1234` lists **two** Ministral catalog entries: `mistralai_ministral-3-3b-instruct-2512`
(publisher `bartowski`) and `mistralai/ministral-3-3b` (publisher `mistralai`) — both `arch:
mistral3`, `quantization: Q8_0`, `max_context_length: 262144`, both `state: not-loaded` at rest.
Confirmed **live**, via direct `curl`/`urllib` against `/v1/chat/completions` and
`/api/v0/models` (not assumed):

- At `temperature: 0`, both ids returned **byte-identical** completions to the same probe prompt
  (`"Reply with the single word: ok"` → `"Understood! 😊"`, `prompt_tokens=542` both times — the
  large count is LM Studio applying its own default system-prompt preset for this model, unrelated
  to falkor-chat).
- Both self-identify identically: *"My name is **Ministral-3-3B-Instruct**, a 3 billion parameter
  conversational AI model developed by Mistral AI."*
- `/api/v0/models`'s `state` field **flips between the two ids depending on which was called
  last** — calling `mistralai_ministral-3-3b-instruct-2512` left `mistralai/ministral-3-3b`
  reporting `state: loaded` and the other `not-loaded`. LM Studio is aliasing two catalog
  registrations onto **one loaded model slot**, not serving two separate weight sets.

**Conclusion: the same underlying weights under two catalog ids.** Used
`lmstudio/mistralai/ministral-3-3b` as the ref throughout (mirrors the existing `qwen/qwen3-4b-2507`
naming shape); the other id is interchangeable.

## 3. Judge-calibration measurement (item 1 of the brief)

**Method — reused, not reimplemented.** A new throwaway driver,
`server/tests/eval/probe_ministral_judge.py` (not `test_*`-named, never pytest-collected), imports
`server/tests/eval/guard_calibration.py` unmodified and monkeypatches only its module-global
`build_call` (rebinding the name from the probe script, not editing the file) to inject
`run = {"ws": "ws:ministral-probe", "modelOverrides": {"guardModel": "lmstudio/mistralai/ministral-3-3b"}}`
in place of the shipped `{"ws": "ws:golden-eval", "modelOverrides": {}}`. Because `run_case` looks
up `build_call` via its own module's globals at call time, this redirection reaches every case
without touching the reviewed file. Every other function — `RecordingJudge`, `run_case`,
`load_golden_guards`, `false_advance_rate`, `advance_recall`, `cohens_kappa`,
`confusion_matrix`, `coercion_flip_rate`, `flip_rate`, `per_path_breakdown`,
`materiality_probe_failed` — is called exactly as `test_guard_calibration_live.py` calls it,
unmodified. `guard.evaluate_guard`'s only judge-model input is `run["modelOverrides"]["guardModel"]`,
which resolves via `ModelGateway.resolve()`'s workspace-override rung (`_workspace_override_ref`
returns from the `overrides=` branch whenever `overrides is not None`, before `ws=` is ever
consulted) — so `"ws:ministral-probe"` is a **label only**, never a graph key any code on this path
selects, reads, or writes. Confirmed by reading the resolution path end to end; no graph was
created, and `GRAPH.DELETE` cleanup was therefore unnecessary (there was nothing to delete).

Same fixture, same protocol, same gates as item 3's live run: `golden_guards.jsonl`
(sha256 `35061c79aa9ae93f5e2350d30e4543a1a64f72b49a4cb0409319cd431d6776b4` — **byte-identical to
the fixture the 2026-08-17 Qwen report used**, confirmed by recomputing it), 26 cases × k=3
replicates = 78 real judge calls, G1 gate ≤10% false-advance, G2 gate ≥80% advance-recall.

**Provenance caveat (report honestly, not glossed over):** `git status` showed
`falkorchat/guards.py`/`tests/test_guards.py` **uncommitted and mid-edit** at run time — this is
U1's parallel work on the K-027 carried findings (m-1 negation clause-boundary fix, m-2
filter-before-slice fix), confirmed by diffing the working tree against HEAD (`ad9e4ff`). Both
changes are narrow (a negator-across-a-clause-boundary edge case; a malformed-row-in-the-tail edge
case) and this run's own **coercion-flip rate was 0.0% throughout** (0/78 calls), consistent with
neither edge case being hit by any golden-set case — but the numbers below are not from a pinned,
clean commit. **If exact reproducibility is later required, re-run once U1's diff lands and the
tree is clean.**

**Second caveat: no temperature pin.** Qwen's calibration run pins `temperature: 0` via
`config/models.json`'s `models["lmstudio/qwen/qwen3-4b-2507"]` entry — the brief forbids editing
that file, and there is no override-time mechanism to set a per-call sampling param without it, so
Ministral ran at LM Studio's **default sampling** (`resolved.params == {}`, confirmed printed by
the probe). This is a real, acknowledged confound against the Qwen numbers below — not fatal (both
runs are single-draw-per-replicate reports, not a claim of tight determinism), but not apples-to-
apples either.

### Results (real, observed — 2026-08-20, LM Studio `mistralai/ministral-3-3b`, quantization `Q8_0`)

**G1 false-advance = 0.0% (n=10 cases / 30 calls) · G2 advance-recall = 45.5% (n=11 cases, 5
advanced) · VERDICT: block** (gate: G1 ≤10% → pass; G2 ≥80% → **fail**).

> This gate is a one-sided screen at n=21 hand-labeled cases. A failure is strong evidence the
> judge is unfit. A pass means only that no large defect was detected at a sample size that could
> not have detected a small one. It is not a calibration certificate.

`FAR_all` (all 15 `expected:false` cases, reported not gated): 0.0% (0/45 calls).

**Cohen's kappa — diagnostic only:** κ = 0.442 (n=21, po=0.714, pe=0.488); rater A (human)
positive rate/prevalence = 52.4%; rater B (judge, per-case majority) positive rate = 23.8%.

**Boundary stratum (n=5):** conservatism (share suspended) = 100.0% — confusion tp=0 fp=0 tn=5
fn=0. Fully conservative, as intended.

**Per-path breakdown:** `understanding` n=15, accuracy 66.7% (tp=3 fp=0 tn=7 fn=5) — 5 of 8
clear-advance cases on this path were false-negatives (missed advances). `turns` n=6, accuracy
83.3% (tp=2 fp=0 tn=3 fn=1).

**Coercion-flip rate:** 0.0% overall (0/78), 0.0% on `r1_probe` cases (0/15) — the judge's raw
decision and `_coerce_verdict`'s final decision never disagreed.

**Flip rate (replicate instability):** 3.8% (1/26 cases — `ca-02`, whose 3 replicates split
True/True/False).

**Materiality-probe check (`ca-04`/`ca-05`/`ca-08` vs. `cs-04`):** bloc criterion **not
triggered** (`cs-04`, the control, correctly suspended — same as Qwen's run), so this is not a
blocker by the protocol's own rule. **Named plainly anyway, mirroring the Qwen report's own
practice:** all three individual materiality probes (`ca-04`, `ca-05`, `ca-08`) suspended — worse
than Qwen's 2-of-3 miss (Qwen correctly advanced `ca-05`). Ministral's rationale for each echoes
the `missing` field near-verbatim rather than reasoning about research-sufficiency, same pattern
Qwen showed on its own two misses.

**A residual parse failure was observed even post-fix:** `cs-07` replicate 2's raw rationale was
literally `"unparseable judge output"` — one of 78 calls. The K-027 item 1 fix (fence/prose-JSON
tolerance) measurably helped (see below) but did not eliminate every parse failure for this
model's output style.

### Comparison to the D13 baseline and to Qwen

| | D13 (pre-fix, bare `json.loads`) | D13 (pre-fix, fence-tolerant re-parse) | **This probe (item-1-fixed, real `evaluate_guard`)** | Qwen, current code (2026-08-17) |
|---|---|---|---|---|
| G2 advance-recall | 0/11 = 0.0% | 4/11 = 36.4% | **5/11 = 45.5%** | 9/11 = 81.8% |
| G1 false-advance | — (26/26 unparseable) | — | **0/30 = 0.0%** | 0/30 = 0.0% |

The K-027 item 1 parse fix **measurably helped** Ministral (0.364 → 0.455, +1 case) — it is now
running through the real `guards.evaluate_guard` path rather than a bare, judge-bypassing
`json.loads` re-parse, and gained one more correct advance. **It did not close the gap.** Ministral
still fails the G2 gate by a wide margin (45.5% vs. the 80% bar), and G1 stays at a clean 0% in
both arms — the failure mode is exactly what D13 predicted: *"even after fence-tolerant
re-parsing... it genuinely over-suspends on clean clear-advance cases, demanding logs/traces
that aren't the gate's bar"* — a reasoning-quality gap the parser layer cannot fix, and did not.

**Judge verdict: block.** Ministral is not viable as the guard judge under this gate, current code
included.

## 4. Terminal tool-call / agent-step measurement (item 2 of the brief)

**A structural finding changed the plan for this measurement, and is the more important result of
this probe.**

### 4.1 `test_workflow_live.py` does not use the workspace-override mechanism

Read fully, per the brief. Unlike the guard-calibration harness, `test_workflow_live.py` builds
`OpenAICompatibleLLM`/`OpenAICompatibleEmbedder` **directly from env-var literals**
(`FALKORCHAT_LIVE_LLM_MODEL`, `FALKORCHAT_LIVE_LLM_BASE_URL`) — its own docstring calls this "D7
mechanism 2," explicitly chosen because a full live e2e run needs no config file. There is no
`run["modelOverrides"]` in this path at all. The brief's premise ("via the same workspace-override
mechanism") does not match what this file actually does — worth naming plainly rather than forcing
a mechanism that isn't there. Pointing it at Ministral is instead a one-line env swap:
`FALKORCHAT_LIVE_LLM_MODEL=mistralai/ministral-3-3b`. Both the `step`/`agent` LLM and the guard
judge share the same client in `_build_live_stack`, so this swap would drive all three kinds at
once (matching D13's own "the judge is the same model as the generator" caveat).

### 4.2 Why the full live e2e was not run: a live-verified crash, checked before running it

Before spending a multi-minute live e2e run, I checked whether the swap would even produce an
informative result, by replaying the **exact** first-call shape the real `intake` node sends
(`executor._assemble_messages`: `systemPrompt`, then thread turns role-mapped as `user`/`assistant`
with `"{speaker}: {text}"`, then an **unconditionally appended** final `user`-role `"CONTEXT:\n..."`
block) against Ministral directly:

```
messages = [
  {"role": "system", "content": <intake systemPrompt, verbatim from scripts/seed_workflows.sh>},
  {"role": "user", "content": "alice: @assistant I need help figuring out what happened with our checkout service."},
  {"role": "user", "content": "CONTEXT:\n{}"},
]
```

**This returns `HTTP 400`** from LM Studio, with the underlying Jinja template error surfaced
verbatim: *`"After the optional system message, conversation roles must alternate user and
assistant roles except for tool calls and results."`* Confirmed the same failure independently on
a minimal 3-message repro (`system`, `user`, `user`) against **both** Ministral catalog ids. The
**identical** message shape against `qwen/qwen3-4b-2507` succeeds cleanly (`finish_reason:
tool_calls`).

**Why this fires on the very first intake call, every time, not just occasionally.** The trigger
message that starts a run is itself a `user`-role thread turn; `_assemble_messages` then appends
one more `user`-role `CONTEXT` block unconditionally — two consecutive `user` messages by
construction on turn 1, before the model gets any say. The same shape recurs structurally for
`research`/`answer`: neither `intake` (parked at `waiting`) nor `research` (granted only
`graphrag_retrieve`, never `post_message`) posts an assistant-visible thread message before
`answer` runs in the same drive, so the thread's last turn reaching `answer`'s own
`_assemble_messages` call is very likely still `user`-authored too.

**What this does at the engine level, traced through the code (not assumed):**
`falkorchat/transport.py`'s `urllib.error.HTTPError` rung wraps the 400 into a
`ProviderCallError`; `executor._drive`'s `try/except HumanHandoffSignal / except Exception`
(`executor.py:440-449`) catches it, calls `_fail_with_note` (`fail_run`, with the exception message
as the note) — **and then re-raises**. So a live Ministral-backed triage run would `fail_run`
loudly on its very first LLM call, and the exception would also propagate to whatever wraps
`trigger.maybe_trigger` (the background task in production; the test function itself in
`test_workflow_live.py`, which asserts no such exception). Running the full live e2e would not
"measure a low AC-4 rate" — it would **crash the test with an unhandled exception before any
guard, any node, or any tool call is exercised**, exactly as uninformative as D12/D13's own R2
caveat warns against ("a harness/format artifact masquerading as a capability result").

**This is the brief's own sanctioned fallback, taken deliberately, not as a shortcut:** given a
confirmed crash on the very first call, I used the narrower, D13-style scripted replay of just the
`post_message` schema call instead (`docs/archive/plans/m3-capability-probe-ml.md` §4's own
protocol; its own `scratchpad/replay_answer.py` no longer exists in the tree — a small new
throwaway script, `scratchpad/replay_answer_ministral.py`, in **this session's scratchpad**, not
`server/tests/eval/`, since this one is genuinely one-off and not meant to be re-run/maintained the
way the judge-calibration probe is).

### 4.3 Terminal tool-call capability, isolated from the alternation crash

To answer "can Ministral call `post_message`" without the confound above, the replay used a
**single merged `user` turn** (thread context + `CONTEXT` block folded together, never two
consecutive same-role messages) — the exact current `answer`-node `systemPrompt` (verbatim from
`scripts/seed_workflows.sh`, including the "MUST post"/"never pass `mentions`" Defect-C
mitigation already shipped) and the exact `PostMessageTool.schema` (verbatim from
`falkorchat/tools.py`), against a synthetic grounded research-findings context (the same checkout
v4.2/connection-pool scenario `test_workflow_live.py`'s own `CORPUS` uses, for realism). **n=5
draws** (D13 used n=3; affordable to go slightly wider given each draw is a few seconds, not a
multi-minute e2e run):

**Ministral: 5/5 draws called `post_message` natively**, correctly schemad (`text` only, no
`mentions`), with clean, grounded, on-topic answers (`finish_reason: tool_calls` every time). This
**reconfirms D13's 3/3 finding** under current code and current prompt.

**Same-session Qwen comparison** (identical prompt, schema, message shape, run right after):
**0/5 draws called `post_message`** — every draw was `finish_reason: stop` with plain prose
content instead. This reconfirms the exact Defect-C failure mode the "MUST post"/"never pass
mentions" prompt mitigation was written to prevent, and shows the mitigation **does not reliably
prevent it** even now — a live, current-code data point that the already-shipped K-039 implicit-
dispatch fallback is not a leftover safety net for a rare case, but the thing actually carrying
Qwen's AC-4 outcome on this exact request shape.

### 4.4 Why K-039 changes what "terminal-post reliability" even means now

`executor._run_agent_node` (`executor.py:723-754`) already contains an **implicit-dispatch
fallback**, shipped 2026-07-31 as K-039 — predating this task but **postdating D13** (2026-07-18).
When a node granted `post_message` ends its turn with non-empty text and no tool call, the
executor dispatches `post_message` **itself**, using that text as the `text` argument. K-027 item 2
(2026-08-16) layers a `must_post_violation` **detector** on top, unchanged, catching only the
narrower residual case: the node ends with **empty** text and no tool call (or exhausts
`maxIterations` with nothing).

**Practical consequence:** since 2026-07-31, a model's native tool-calling weakness on the terminal
node is, for AC-4 purposes, **already compensated** as long as the model (a) produces non-empty
text and (b) does not crash the request outright. Qwen's measured 0/5 native-call rate on the exact
current prompt (§4.3) is therefore not the number that determines whether the user sees an answer
— K-039's fallback is. Ministral's superior native tool-calling (5/5, reconfirmed) buys it nothing
over Qwen on this axis in practice, **and the alternation crash (§4.2) means Ministral cannot even
reach the point where either K-039 or K-027 item 2 would get a chance to help** — the request never
completes.

## 5. Verdict

| Axis | Verdict | Basis |
|---|---|---|
| **Guard judge** (`kind: guard`) | **block** | G2 advance-recall 45.5% < 80% gate, real `evaluate_guard` run, current code. Same directional result as D13 (over-suspending on clear-advance cases); the item-1 parse fix improved the number (0.364→0.455) but did not close the gap to Qwen's 0.818. |
| **Agent/step model** (`kind: agent`/`step`) | **block — worse than "loses," structurally broken** | The current `_assemble_messages` convention (unconditional trailing `user`-role `CONTEXT` block after thread turns) sends two consecutive `user`-role messages on the very first `intake` call and, structurally, again for `research`/`answer` — Ministral's LM Studio-served chat template hard-rejects that shape (`HTTP 400`), which `executor._drive`'s fault net turns into an unhandled `fail_run` + re-raised exception. Confirmed live, both catalog ids, both the raw 3-message repro and the real `intake` prompt. Not measurable as a capability gap at all until this is fixed — and fixing it is an `executor.py` change, out of this unit's scope (and not one of the K-027 carried findings U1 is already fixing — a **new** finding to route separately). |

**Does this change D13's practical relevance? No — it reconfirms and sharpens it, on a mechanism
D13 never tested.** D13's expectation ("Qwen3-4B is the best that fits") holds, but not for the
reason D13 measured: Ministral is **not** weaker at the terminal tool call (5/5 native vs. Qwen's
reconfirmed 0/5 — Ministral remains the better-behaved model on that narrow axis, exactly as D13
found). It loses because (a) the now-shipped K-039 fallback already neutralizes Qwen's
tool-calling weakness for AC-4 purposes, making the axis D13 measured close to moot in practice,
and (b) Ministral introduces a **new, more severe** failure mode — a hard per-request crash from a
chat-template incompatibility with the current message-assembly shape — that Qwen does not have.
On the judge axis, the item-1 parse fix helped but did not change the underlying verdict: Ministral
was blocked pre-fix and remains blocked post-fix, on the same reasoning-quality gap D13 diagnosed,
which a parser cannot repair. **Recommendation: do not wire Ministral for either kind under the
current codebase. The judge-quality gap is a model-capability limit (matches D13, no action
implied beyond what K-027 item 3 already decided for Qwen). The alternation crash is a genuine,
new, currently-undocumented defect in `_assemble_messages`'s message-shape assumptions that is
worth its own backlog item if multi-vendor model portability (beyond Qwen's own tolerant template)
is ever a goal — flagged here, not fixed here (`guards.py`/`executor.py`/`app.py` are out of this
unit's scope per the brief).**

## 6. What could not be measured, and why

- **No temperature pin for Ministral's judge run** (§3, second caveat) — `config/models.json`
  cannot be edited per the brief; the run used LM Studio's default sampling. A real, disclosed
  confound against Qwen's temperature-0-pinned run, not fatal to the directional conclusion (a
  45.5%-vs-80% gap is far outside single-run sampling noise).
- **`guards.py` was mid-edit (U1, concurrent unit) at run time** (§3, first caveat) — numbers are
  not from a pinned clean commit; the run's own 0.0% coercion-flip rate is consistent with neither
  in-flight fix mattering to this fixture, but a fully reproducible provenance record needs a
  re-run once U1 lands.
- **No full live e2e run** — deliberately not attempted after confirming (live, not assumed) that
  it would crash on the first LLM call; §4.2 documents the direct verification in place of the run.
  D13's own M1 "intake advancement over n runs" measure is therefore **not reproduced here** — it
  is not measurable at all under the current message-assembly code, for either arm's comparison to
  be meaningful, until the alternation issue is fixed.
- **AC-4/terminal-post rate is not on record for Qwen post-K-027** in `HISTORY.md`'s 2026-08-16/17
  entries (only the guard-calibration numbers are) — the §4.3 same-session 0/5 Qwen replay is this
  probe's own fresh substitute, not a citation of an existing number.

## 7. Artifacts

- `server/tests/eval/probe_ministral_judge.py` — the judge-calibration driver (git-tracked, reusable
  for a future re-probe of any other model via the same `run["modelOverrides"]["guardModel"]`
  pattern; not pytest-collected).
- Raw run output: printed to stdout by the probe above (not written to `docs/test-reports/` — that
  directory's `Status:`-flip ownership is `qa-engineer`'s per root `AGENTS.md`; the numbers in §3
  are transcribed by hand from the real run's own printed output, not estimated).
- `scratchpad/replay_answer_ministral.py` (session scratchpad, not git-tracked) — the terminal
  tool-call replay, §4.3.

## 8. Cleanup

No throwaway FalkorDB graph state was ever created — confirmed by reading the resolution path
(§3): `"ws:ministral-probe"` never reaches a graph selector on the judge-calibration path, and the
terminal-tool-call replays (§4.2/§4.3) never touch FalkorDB at all (raw `urllib` calls to LM Studio
only). `GRAPH.DELETE` was therefore not needed and was not run.
