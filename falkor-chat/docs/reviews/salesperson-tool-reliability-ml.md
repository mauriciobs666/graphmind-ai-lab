# `salesperson` scaffold — live tool-call reliability diagnostic (D-1 follow-up)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-052, K-056 (M6) — informs
> K-053/K-054/K-055 sequencing

## Verdict

**Superseded by §8 (Pass 2, 2026-08-29) as the operative recommendation** — a controlled eval
(n=40 conversations / 280 turns) confirms the skip-and-fabricate mechanism is a **near-certain,
deterministic-onset capability ceiling for `qwen/qwen3-4b-2507`** in this role, not a
probabilistic soft-degradation a scaffold tweak can dampen (both scaffold-level mitigations
this note named — `tool_choice` forcing here, a replayed-history breadcrumb tried afterward —
are independently falsified; see §8's own recap). §8 also tested three alternative local
models and found one, `mistralai/ministral-3-3b`, with **zero** instances of this mechanism
across 176 further turns — a same-size-class, immediately-available candidate, though it
carries its own distinct, real defect (§8.4) that still needs closing before it ships. Read §8
for the numbers and the recommendation this now drives; §1-§7 below are Pass 1's original
root-cause finding, left intact as the mechanism analysis that still holds.

### Pass 1 verdict (2026-08-27, historical — see §8 for the current call)

**Proceed with K-053/K-054/K-055 as planned, gated on one cheap, model-independent
mitigation: ship an observability signal for "answered a factual question without dispatching
a lookup tool this turn," generalized to every sibling capability's own tools.** Do **not**
adopt `tool_choice` forcing as a fix without its own controlled eval — I tested it directly
against a known-failing prompt on this exact stack and it neither forced a tool call nor was
safe (see §4.2). This is not a hold: nothing found here is specific to K-052's own code, and
delaying K-053+ would not itself de-risk anything — the useful next steps (the eval in §4.2,
the observability signal in §4.1) are independent of whether those three capabilities are
already landed.

The evidence (§2, two independent live repros with `trace=True`) **resolves QA's open
question definitively**: on every fabricating turn, in both runs, **the model never invoked the
tool at all** — the very first LLM turn of that node execution emitted plain text, zero
`tool_calls`. This is not "tool called with stale/wrong arguments," and not "correct tool
result overridden by the model" — it is a clean skip straight to pattern-completed prose. The
`requiredTools` engine safety net (`["post_message"]`) does not, and structurally cannot,
catch this: it guards that *a reply gets delivered*, not that the reply was grounded in a
tool call — `lookup_product_fact`/`filter_products` are not (and cannot generically be) in
`requiredTools`, so a fabricated reply sails through every existing guard clean.

## 1. The question and the decision it serves

QA's D-1 (`docs/test-reports/workflow-catalog-lookup-report.md`) found the live
`qwen/qwen3-4b-2507` model reproducibly fabricates catalog facts on later turns of an
extended `salesperson@v1` conversation, while identical questions succeed as turn 1 of a
fresh conversation. QA ruled out the repository/service/tool layer (ground-truth Cypher
matches in every case) but explicitly could not determine, without `trace=True`, whether the
tool was invoked with bad arguments on the failing turns or not invoked at all — that
distinction, plus a severity/sequencing call for K-053/K-054/K-055 (three sibling
capabilities about to add more tools to this exact scaffold and model), is what this note
answers.

## 2. Method

Reproduced D-1's turn sequence twice, independently, driving `services.start_workflow_run`
(turn 1) / `services.resume_workflow_run` (turns 2-9) directly in-process — the same pattern
`server/tests/test_workflow_live.py::_build_live_stack` uses for `triage`, pointed at
`salesperson@v1` instead — with `WorkflowTrigger(..., trace=True)` (the run-level trace flag
persists across every later `resume` on that run, so only turn 1 needs the explicit flag).
This bypasses the `@mention` REST path's `trace=False` default (confirmed in
`falkorchat/app.py:396-399` — `WorkflowTrigger` is constructed with no `trace=` kwarg there)
**without modifying any shipped code**, exactly QA's own suggested next step (report
§"Feedback & recommendations" item 1).

Workspace: fresh throwaway `ws:ds-tool-reliability` (`bootstrap_schema.sh` →
`seed_demo.sh` → `seed_catalog.sh` → `seed_salesperson.sh`, `EMBEDDING_DIM=1024`), against
the real `falkordb-dev` instance and real LM Studio (`localhost:1234`,
`qwen/qwen3-4b-2507`) — no gateway-IP connectivity issue this pass (`localhost:1234` answers
directly). Each turn's trace was read back via `GraphTracer`'s `TraceEvent` chain
(`MATCH (StepRun)-[:TRACED]->(TraceEvent)`), dumping `llm_prompt`/`llm_response`/`tool_call`/
`tool_result` per iteration. **Teardown:** `GRAPH.DELETE ws:ds-tool-reliability` after the
pass — nothing was left in the shared graph; `reference` was never written to (catalog/def
seeding was idempotent no-ops against already-present data).

Two independent fresh threads/runs were driven through the full 9-turn D-1 sequence
(turns 1-2 correct baseline, 3-4 price-range-filter repro, 5-6 abstention controls, 7-9
exact-name-lookup repro).

## 3. Trace evidence

**Run 1** — tool-call status per turn (`✓`=tool invoked and correct, `✗`=no tool call,
fabricated/wrong, `✓*`=no tool call but happened to be correct by luck):

| Turn | Question shape | Tool invoked? | Result |
|---|---|---|---|
| 1 | exact-name lookup (Wireless Mouse Pro) | ✓ `lookup_product_fact` | correct ($29.99) |
| 2 | category filter (Wearables) | ✓ `filter_products` | correct (2 items) |
| 3 | price-range filter (<$30) | ✗ none | **fabricated** (invented 2 products) |
| 4 | rerun of 3 | ✗ none | same fabrication repeated verbatim |
| 5 | exact-name lookup (nonexistent product) | ✓ `lookup_product_fact` | correct abstention |
| 6 | category filter (nonexistent category) | ✗ none | correct abstention, but **not tool-verified** |
| 7 | exact-name lookup (Portable SSD 1TB) | ✗ none | **fabricated** ($149.99 vs. true $109.99) |
| 8 | rerun of 7 | ✗ none | same fabrication repeated |
| 9 | rephrase of 7 | ✗ none | same fabrication repeated |

**Run 2** (independent fresh thread, same sequence) — the collapse point moved (turn 4
instead of turn 3) but the shape is identical and, past onset, **never recovered** across
all 6 remaining turns:

| Turn | Question shape | Tool invoked? | Result |
|---|---|---|---|
| 1 | exact-name lookup | ✓ `lookup_product_fact` | correct |
| 2 | category filter | ✓ `filter_products` | correct |
| 3 | price-range filter (<$30) | ✓ `filter_products` | correct (all 3 real items) |
| 4 | rerun of 3 | ✗ none | happened to repeat 3's correct text verbatim — **not re-verified** |
| 5 | exact-name lookup (nonexistent product) | ✗ none | **fabricated a price** ($149.99) for a product that does not exist — worse than run 1's abstention at the analogous slot |
| 6 | category filter (nonexistent category) | ✗ none | correct abstention, not tool-verified |
| 7 | exact-name lookup (Portable SSD 1TB) | ✗ none | **fabricated** ($149.99 — same number run 1 fabricated for the same product) |
| 8 | rerun of 7 | ✗ none | same fabrication repeated |
| 9 | rephrase of 7 | ✗ none | same fabrication repeated |

Representative trace excerpt (run 1, turn 3 — the first-observed skip):

```
[e6d4be82 seq=1] llm_prompt: iter 1/8: 6 msgs, 3 tool(s)
[e6d4be82 seq=2] llm_response: text: Assistant: Here are the products we have under $30: ...
```
— iteration 1 of 8, straight to text, zero `tool_calls`. Compare turn 1's first iteration:
```
[22ae730f seq=1] llm_prompt: iter 1/8: 2 msgs, 3 tool(s)
[22ae730f seq=2] llm_response: tool_calls: lookup_product_fact
```

**The exact fabricated price, $149.99, recurred identically across both independent runs**
for two different real products (Portable SSD 1TB in run 1 and run 2; also the nonexistent
Quantum Toaster 3000 in run 2) — consistent with the model falling back to some
plausible/salient number from its own prior context or training distribution rather than
any retrieved value, not independent random confabulation each time.

## 4. Analysis

### 4.1 Mechanism

This is **not** context-window/token-budget pressure in any literal sense — the absolute
context at the collapse point is tiny (6-8 short messages, well under any 4B model's context
limit) — `executor._assemble_messages` rebuilds the prompt fresh every turn from
`_read_thread_context` (capped at `THREAD_CONTEXT_WINDOW=20` messages) and **only replays
each prior turn's final text**, never the tool-call/tool-result scaffolding used internally
during that turn's own execution (that scratch space is discarded once the node returns).
From the model's point of view, by turn 3-4 the visible conversation history is 2-3 clean
"user asks, assistant answers" text exchanges with **no visible evidence a tool was ever
used** — the system prompt says "never guess... using your catalog tools," but the model's
own in-context precedent (its own prior turns, replayed as ordinary chat) increasingly looks
like a conversation that gets answered directly. By turn 3-4 that precedent dominates the
instruction, and the model starts pattern-completing a plausible-sounding answer on the very
first LLM turn instead of reaching for a tool. This is a documented category of small-model
failure — weak **instruction-vs-in-context-precedent robustness**: a 4B-class model is far
more susceptible than a frontier model to a visible pattern in its own context overriding an
explicit system instruction, especially when the instruction and the pattern diverge. It is
distinct from (and should not be conflated with) classic long-context degradation (needle-
in-a-haystack recall loss) — the context here is short; what degrades is *instruction
adherence under self-generated precedent*, which is a scaffold-shape issue (what the replay
shows the model) as much as a raw-capability one.

The one partial counter-example (run 1 turn 5: an exact-name lookup still triggered a real
tool call, mid-collapse) is a plausible but unconfirmed further data point for this same
theory — turn 5's phrasing/shape most closely mirrors turn 1's own (the strongest, earliest,
and only fully unambiguous single-tool-call precedent in the transcript at that point),
so the model may have reverted to imitating that closer precedent rather than the more
recent no-tool one. Flagged as a hypothesis, not a confirmed finding — n=1 occurrence.

### 4.2 Tested mitigation: `tool_choice` forcing — falsified, do not adopt as-is

`OpenAICompatibleLLM.chat()` (`falkorchat/llm.py:145-175`) sends `tools` with no
`tool_choice` field at all — the wire request implicitly gets the OpenAI-compatible default
(`"auto"`), meaning the model is always free to skip every tool. The obvious cheap fix is
forcing `tool_choice: "required"` on the wire. **I tested this directly against the exact
prompt that failed live** (run 2's turn 4, reconstructed byte-identically via the shipped
`WorkflowExecutor._assemble_messages`, replayed with `OpenAICompatibleLLM(..., params=
{"tool_choice": "required"})` against the same live LM Studio instance): it **did not force
a tool call** (`is_tool_call=False`, `tool_calls=[]`, identical text to the unforced
control) — LM Studio/this Qwen3 GGUF appears to silently accept-and-ignore the field for
this request shape — **and additionally triggered a separate degenerate failure mode**: the
model entered a runaway repetition loop ("I want to see everything. I want to see the full
details. I want to see the complete list. ..." repeated hundreds of times) instead of a
clean reply. That is a worse outcome than the original bug (a garbage/looping
`post_message` payload rather than a plausible-looking wrong answer), so **`tool_choice`
forcing must not be shipped on this stack without its own controlled eval** — this was a
single-prompt test (n=1), not a characterization of the failure rate, but it is a hard
falsification of "just add `tool_choice: required`" as a drop-in fix.

### 4.3 Other candidate mitigations, briefly

- **System-prompt reinforcement** — already maximally fresh (`_assemble_messages` re-sends
  the full system prompt, unabridged, every single turn; it is never "stale"), and the
  prompt already explicitly instructs "never guess... using your catalog tools." The
  fabrication happens anyway, so more/stronger wording in the same slot is a weak lever,
  not a promising fix on its own.
- **Periodic conversation summarization** — targets true context-overflow, not the observed
  mechanism (collapse onset at 6-8 short messages, far from any overflow). Not indicated
  for D-1 specifically.
- **Lowering `SALESPERSON_MAX_STEPS`** — would just truncate legitimate long demos sooner;
  does not address root cause and actively fights the scaffold's intended usage pattern.
- **Reinforcing the *replayed history* itself** (untested, plausible given §4.1's
  mechanism) — instead of replaying only plain "speaker: text" turns, fold a one-line
  breadcrumb into each replayed assistant turn that used a tool (e.g. "[verified via
  lookup_product_fact]"), so the model's own in-context precedent shows tool use happening,
  not just answers appearing. This targets the actual observed mechanism rather than
  fighting sampling parameters, but needs its own implementation and eval — proposed as
  scope for the new backlog item below, not something to ship on this note's evidence alone.

### 4.4 Severity/risk for K-053/K-054/K-055 sequencing

**This gets worse with more tools and more turns, not turn-count-independent noise.** Two
independent live reproductions both collapsed within the first 3-4 agent turns — a
conversation window K-053/K-054/K-055 will each reach in ordinary use, not an edge case —
and once collapsed, the failure was **persistent for the remainder of the run in both
reproductions bar one exception** (run 1 turn 5). More tools compound the opportunity for
the same skip-and-fabricate pattern (K-053 cart/order actions and K-054 durable-profile
writes turn a fabricated *reply* into a fabricated *state mutation* if the model similarly
skips a write-tool and narrates a plausible-sounding confirmation instead — materially
higher-severity than a wrong catalog fact). This is squarely inside the risk this pass was
asked to characterize before those three ship.

## 5. Proposed new backlog item (for `teco` to file — `docs/BACKLOG.md` is `teco`-owned)

**Title:** `salesperson` scaffold — observe and mitigate live tool-call skip-and-fabricate
degradation under extended conversations (small local model)

**Scope (one paragraph):** Trace-instrumented live reproduction (this note,
`docs/reviews/salesperson-tool-reliability-ml.md`) confirms `qwen/qwen3-4b-2507` served via
LM Studio reproducibly (2/2 independent runs) stops invoking `lookup_product_fact`/
`filter_products` after 2-3 successful tool-calling turns within one `salesperson@v1`
conversation — the model's very first LLM turn skips straight to pattern-completed text,
zero `tool_calls` — and fabricates catalog facts the existing `requiredTools` engine
guard cannot catch (it only requires `post_message`). A first candidate fix, forcing
`tool_choice: "required"` on the wire, was tested directly against a known-failing prompt
on this exact stack and neither forced a tool call nor was safe (triggered a separate
runaway-repetition failure). Scope: (1) ship a cheap, model-independent observability
signal — log/trace a warning whenever an agent-node turn's final answer looks fact-bearing
(references a specific catalog term/price-like token) but no catalog-fact tool was
dispatched during that node's own execution, generalized to whatever tools K-053/K-054/
K-055 add; (2) run a proper controlled eval (repeated live conversations varying turn/tool
count) to replace this note's n=2 anecdote with an actual rate estimate, and to evaluate the
untested "tool-use breadcrumb in replayed history" mitigation (§4.3) against a baseline;
(3) decide, on real data, whether a scaffold-level fix exists or this is a capability
ceiling for a 4B local model in this role, in which case the recommendation becomes "route
this role to a larger model" rather than "keep patching the 4B path."

## 6. Correction to QA's report

`docs/test-reports/workflow-catalog-lookup-report.md` states D-1 is "already flagged as a
known open epic (K-027)." This is incorrect: `docs/BACKLOG.md`/`docs/HISTORY.md` show
**K-027 closed 2026-08-21**, and its scope was tool-call *parsing* precedence bugs (bare-call
vs. JSON-envelope ambiguity in `llm._parse_content_tool_calls`, golden-set expansion,
guard-judge calibration) — a different mechanism entirely from D-1's live conversational
degradation. No existing backlog item covers D-1's actual failure mode; §5 above is the
proposed new item.

## 7. Artifacts

Two throwaway diagnostic scripts (not part of the shipped test suite, not committed):
`ds_trace_repro.py` (the 9-turn D-1 repro driver, `trace=True`) and
`ds_tool_choice_probe.py` (the `tool_choice` falsification test) — left in this session's
scratchpad, not in the repo. `ws:ds-tool-reliability` was `GRAPH.DELETE`d after the pass;
`reference` was never mutated (catalog/def seeding was idempotent no-ops).

## 8. Pass 2 (2026-08-29) — controlled eval, detector precision/recall, model alternatives

**What this pass answers, and for whom.** Pass 1 (§1-§7) diagnosed the mechanism from n=2
anecdotes and named three open items (§5): a real rate estimate, a precision/recall check on
the shipped observability signal at scale, and a scaffold-fix-vs-larger-model decision. Since
Pass 1, two scaffold-level mitigations were tried and fell to other units — `tool_choice`
forcing (§4.2, this note) and a tool-use breadcrumb folded into replayed history (U37/U38/U39,
`docs/BACKLOG.md` K-056) — the second live-verified 2/2 to not resolve the fabrication and to
introduce a worse side effect (the model imitating the breadcrumb's surface format without
calling a tool), reverted. **Neither is re-tested here; both stay closed.** This pass runs the
controlled eval, checks the detector against ground truth at scale, and — per a mid-run steer
from the coordinator, expanding this pass's own original scope — checks every alternative
local model actually loadable in this environment (not only larger ones) for the same failure
mode: `openai/gpt-oss-20b` (larger) and `mistralai/ministral-3-3b` (same/smaller size class,
marketed on tool-calling instruction-following).

### 8.1 Method

Same in-process harness precedent as Pass 1 (`_build_live_stack`'s wiring, not the shipped
`@mention` REST path), against a throwaway `ws:eval-k056` (+ one per alternative model probed,
all `GRAPH.DELETE`d after this pass — none is `ws:acme`). Three fixed conversation scripts
against the real, published `salesperson@v2`/real LM Studio, each run **independently and
fresh** (new customer id, new thread, new run) so every turn is a genuine trial, not a
continuation:

- **Condition A** (9 turns, read-only catalog) — reproduces Pass 1's own D-1 sequence almost
  verbatim (exact-name lookup, category filter, price-range filter + repeat, an abstention
  pair, then a second exact-name lookup repeated/rephrased three times).
- **Condition B** (7 turns, **write-mutating** — the escalated risk this pass was specifically
  asked to characterize) — add 2×Wireless Mouse Pro, add 1×Portable SSD 1TB, view cart, remove
  1×Wireless Mouse Pro, remove Portable SSD 1TB entirely, view cart, add 1×Bluetooth Speaker
  Mini.
- **Condition C** (4 turns, read-only, short) — a turn-count-sensitivity probe: does a short
  conversation show a materially lower failure rate than A/B, or does the same onset point
  recur regardless of total planned length?

Two ground-truth signals per turn, both read directly off what the engine itself already
records (no reliance on the shipped signal for either):
1. **Tool-invocation ground truth** — `repository.read_trace`'s per-turn slice (iteration 1's
   `llm_response` trace payload: `tool_calls: …` vs `text: …`), the same unambiguous signal
   Pass 1's own `trace=True` repro used, at n=40 conversations instead of n=2.
2. **Correctness ground truth** — for a catalog fact, exact-match containment of the seed
   catalog's real price/category-derived expected set against the posted reply (mirrors
   `workflow-nl-query-generation-ml.md` §3/§5's own value-containment convention, this lab's
   established pattern for a closed-form-fact check, not a judge); for a cart turn, the
   **live `Cart`/`CartItem` graph state read directly after the turn** (byte-identical
   methodology to D-1's own repro in `docs/test-reports/workflow-cart-and-totals-report.md`).

The shipped `_note_possible_fabrication` signal is read from the same trace, independently of
both ground-truth signals above, so its agreement with them is a real measurement, not an
assumption. Baseline sample: **40 independent conversations / 280 turns** against
`qwen/qwen3-4b-2507` (A×15, B×15, C×10) — chosen after a smoke test (n=1 per condition) showed
each conversation completes in ~1.3s/turn on this instance, making a much larger n than Pass
1's cheap enough to run in a few minutes rather than a genuine cost/time trade-off. Alternative
models were probed at a smaller sample (`gpt-oss-20b`: A×4/B×4, capped by a discovered
instability — see §8.3; `ministral-3-3b`: A×10/B×10/C×4, extended from an initial A×4/B×4 once
the first batch showed a clean, striking result worth firming up). All Wilson 95% CIs below use
z=1.96, this lab's own established convention (`workflow-nl-query-generation-ml.md` §5).

### 8.2 Baseline rate estimate — `qwen/qwen3-4b-2507`

**The failure is not a gradually-increasing-probability function of conversation length — it
is a near-deterministic collapse at a fixed turn position, independent of turn content, tool
type, or planned conversation length.** Skip rate (no domain tool call at all this turn) by
turn position, pooled across all three conditions' independent runs:

| Turn | A (n=15) | B (n=15) | C (n=10) |
|---|---|---|---|
| 1-2 | 0/15 | 0/15 | 0/10 |
| 3 | 0/15 | 1/15 | 0/10 |
| **4** | **15/15** | **15/15** | **10/10** |
| 5-9 (where scripted) | 15/15 each | 15/15 each | — |

**39 of 40 independent conversations (97.5%, Wilson 95% CI 87.1-99.6%) collapse at exactly the
4th user turn; the one exception collapses one turn earlier (turn 3).** This holds whether
turn 4 is a *repeat* of turn 3's question (condition A), a *brand-new* write-mutating
instruction (condition B's `remove_from_cart`), or the *last* turn of a short 4-turn script
(condition C) — the onset point is not correlated with what the turn asks, only with its
position. This sharpens, rather than contradicts, Pass 1's §4.1 "instruction-vs-in-context-
precedent robustness" mechanism: it is not that longer context makes fabrication *more likely*
in a soft sense, it is that **exactly three prior "user asks, assistant answers" text
exchanges with no visible tool evidence** is enough to flip this model's completion pattern on
the very next turn, essentially every time.

**Persistence is total: zero recoveries in 121 post-onset turns (100%, Wilson 95% CI 96.9-
100%).** Pass 1 §4.1 flagged one partial counter-example (run 1 turn 5, a real tool call
mid-collapse) as "a hypothesis, not a confirmed finding — n=1." At n=40 (121 turns after the
first observed skip in each run), that hypothesis is **disconfirmed** — once a conversation
skips once, every later turn skips too, with no exception in this sample.

**Correctness given tool-called vs. skipped confirms the defect is 100% conversational, 0%
code.** Every turn where a domain tool actually fired was correct: 119/119 (100%, Wilson 95%
CI 96.9-100%) — the repository/service/tool layer (already gated at K-052/K-053's own
`analyst`/`qa-engineer` passes) contributes zero failures. Turns where the tool was skipped:
56/161 (34.8%, Wilson 95% CI 27.9-42.4%) were "correct" only by luck — a repeated still-valid
fact, or a genuine abstention with no fabricated content — the remaining **65.2% of all
skip-turns were concretely wrong**: an invented product, a wrong price, or (condition B) a
claimed cart mutation with the graph unchanged.

**The write-mutating escalation is not a rare tail risk — it is the modal outcome.** Every one
of 15 independent condition-B conversations (100%, Wilson 95% CI 79.6-100%) produced at least
one turn where `remove_from_cart` (or a repeat `add_to_cart`) was silently skipped and narrated
as succeeded while `Cart`/`CartItem` ground truth showed **zero change** — the exact D-1
mechanism from `docs/test-reports/workflow-cart-and-totals-report.md`, now reproduced on demand
rather than observed once. Given K-056's near-total onset by the 4th customer turn regardless
of which tool is at stake, **K-054's `save_profile` is expected to fail identically under
ordinary multi-turn use on the current model** — this is not a hypothetical extrapolation, it
follows directly from the turn-position-only onset finding above.

### 8.3 Detector precision/recall — `_note_possible_fabrication` at scale

The shipped signal was **live-verified to fire correctly** at U37 (2/2 passes) but never
checked against ground truth at a scale that could estimate its error rate. At n=280 turns:

| Check | Result |
|---|---|
| False-positive rate (fires while a domain tool *was* dispatched this turn) | **0/119 = 0%** (Wilson 95% CI 0-3.1%) — never cries wolf on a genuinely tool-verified turn. |
| Recall on the failure class that actually matters (skipped **and** ground-truth wrong) | **103/105 = 98.1%** (Wilson 95% CI 93.3-99.5%) |
| Recall on *all* ground-truth skips (including benign "lucky" abstentions/repeats) | 129/161 = 80.1% (95% CI 73.2-85.6%) — the gap from 98.1% is explained entirely by the signal correctly staying silent on skip-turns that were not fact-bearing (a genuine abstention "we don't have that in Robotics") — appropriate, not a miss. |

**The signal has one confirmed, specific blind spot, both instances write-mutating.** The two
misses in the "matters" row above are both condition-B turns where the model's fabricated
confirmation contained **no price-shaped token at all** — `"The Portable SSD 1TB has been
successfully removed from your cart. Your cart is now empty."` — `_looks_fact_bearing`'s regex
(currency-prefixed or two-decimal numbers, `executor.py:290-306`) has nothing to match. This is
exactly the failure shape the brief flagged as a risk (a bare mutation-confirmation with no
number to key on) and it is now a confirmed, not merely hypothesized, gap: **the observability
signal is weaker on exactly the escalated write-mutating class than on the read-only catalog
class it was originally designed against.** Not a blocker to the signal's continued value (98%
recall on the class that matters is still a strong ops-alerting number, and its 0% false-
positive rate makes it cheap to trust when it *does* fire) — but a follow-up worth naming: a
second, narrower regex/heuristic keyed on a bare confirmation verb ("removed", "added",
"cleared") with no accompanying tool dispatch this turn, independent of whether a price
appears, would close this specific gap. Not implemented here (this is a diagnostic pass, not
an implementation one) — routes to `tdd-engineer` if the team wants it closed.

### 8.4 Alternative-model check (per the coordinator's mid-run scope expansion)

This environment's LM Studio instance (queried directly via `/v1/models`, not inferred from
`config/models.json`) has substantially more than the currently-pinned model loaded/available:
`prism-ml/bonsai-27b`, `mistralai/ministral-3-3b` (+ an older `-instruct-2512` download of the
same model), `qwen/qwen3-4b-thinking-2507`, `google/gemma-4-12b` (+ `-qat`), `google/gemma-3-4b`
/`-12b`, a few uncensored/distilled derivatives, `qwen/qwen3.5-9b`, `nvidia/nemotron-3-nano-4b`,
`openai/gpt-oss-20b`. A single-turn, cold-context tool-calling sanity check (bare system
prompt + one tool schema, no conversation history) was run against several before committing to
a probe: `google/gemma-3-12b` **failed even this trivial check** (fabricated a price on turn 1
of a brand-new conversation, worse baseline behavior than the pinned model's own turn-1
success rate) and was dropped without a full probe; `openai/gpt-oss-20b`, `qwen/qwen3.5-9b`,
and both `ministral-3-3b` variants passed and were carried forward.

**`openai/gpt-oss-20b` (larger, ~5x parameter count) — not usable on this serving stack as-is,
directionally encouraging where it ran.** Probed at A×4/B×4 (intended 64 turns): **6 of 8
conversations (75%, Wilson 95% CI 40.9-92.7%) crashed outright on an early turn** with a
server-side error from LM Studio itself (`HTTP 400`, `"the model produced output that does not
match the expected peg-native format"`) — a grammar/parser incompatibility between this LM
Studio build and gpt-oss's harmony response format under a multi-tool schema, not a model-
capability failure (confirmed via direct `WorkflowRun.ctx` inspection: the *first* customer-
facing turn had already posted a **correct**, tool-grounded reply and, for condition B,
correctly mutated the `Cart` graph, before a **second**, follow-up LLM call inside the same
node crashed and left the engine's own `StepRun`/`TraceEvent` audit trail entirely unwritten
for that turn — a distinct, worse-than-K-056 observability gap: not a false negative, a total
absence of any record for an action that actually happened). The two conversations that
survived past turn 1 (reaching stepCount 6 and 1 respectively before crashing later) showed
**zero skip-and-fabricate instances across every completed turn** — including a full add→add→
view→remove→remove→view cart sequence, well past the pinned model's own turn-4 collapse point
— but also a severe, distinct defect: on every reply, the model called `post_message` again and
again (up to the full 8-iteration budget) re-posting near-duplicate confirmations of an
already-completed action instead of stopping cleanly, a customer-facing message-spam problem
independent of the crash. **Verdict: not a viable fallback candidate today** — the crash rate
alone blocks a real measurement of its underlying capability, and the repetition defect would
need its own fix regardless. A `devops` investigation into the LM Studio/gpt-oss compatibility
issue (template/grammar config, or an LM Studio version bump) is a reasonable, named follow-up;
not something this pass can action or unblock.

**`mistralai/ministral-3-3b` (same/smaller size class, tool-calling-optimized) — the strongest
result in this pass, immediately available, but not risk-free.** Probed at A×10/B×10/C×4 (176
turns, 24 independent conversations): **zero skip-and-fabricate instances at any turn
position in any condition** — 0/176 (Wilson 95% CI 0-2.1%), including every turn past the
pinned model's own turn-4 collapse point in both the 9-turn and 7-turn (write-mutating)
scripts. The shipped detector fired zero false positives across the same 176 turns (consistent
with a genuinely clean sample, not signal failure). **This directly contradicts a "bigger
fixes it" framing** — a *smaller* (3B vs. 4B) but differently-trained/tuned model shows
categorically better tool-call-vs-in-context-precedent robustness on this exact mechanism,
consistent with the coordinator's own steer that this may be a model-specific robustness trait
rather than a capability-scale one.

**Ministral is not, however, a risk-free swap: it has its own distinct, real defect.** In 3 of
10 (30%, Wilson 95% CI 10.8-60.3% — wide at this n, but not negligible) condition-B
conversations, turn 2 (`"Also add 1 Portable SSD 1TB"`, following turn 1's `"add 2 Wireless
Mouse Pro"`) triggered the model **re-issuing turn 1's already-completed `add_to_cart` call a
second time** (`add_to_cart(Wireless Mouse Pro, quantity=2)` dispatched again, confirmed via the
raw tool-call trace) alongside the actually-requested new item — silently doubling the mouse
quantity the customer never asked to add again. This is a **categorically different** failure
from K-056: every tool call is real and dispatched, the graph mutation is real, and every reply
is honestly grounded in the resulting (if wrong) cart state — there is no fabricated-success-
with-zero-effect gap, and the shipped detector correctly stays silent throughout (it isn't
designed to catch this class, and shouldn't be expected to). But it is still a genuine
customer-impacting risk (a silently inflated order) at a rate too high to ignore. **Not
resolved or investigated further here** — this pass's scope was K-056's specific mechanism, and
this is a different one, surfaced as a byproduct of testing the alternative candidate. Flagged
as a precondition on adopting Ministral, not a reason to prefer the status quo (`qwen3-4b`'s
own defect is both more frequent — near-100% by turn 4 vs. 30% on a specific instruction
pattern — and more severe — a total fabrication vs. an honestly-reported over-execution).

### 8.5 Recommendation

**Do not ship K-054/K-055 on `qwen/qwen3-4b-2507` as currently scaffolded without accepting a
disclosed, quantified, near-certain (87-100% CI) risk that any real conversation reaching a
4th customer turn silently fabricates at least one write-mutating action.** This is a
confirmed capability ceiling for this exact model in this conversational role, not a scaffold
defect — both scaffold-level mitigation candidates this note's own lineage produced (`tool_choice`
forcing, the replayed-history breadcrumb) are independently falsified, and this pass's own
turn-position-only onset finding (§8.2) argues against a third scaffold attempt: the collapse
is too sharp and too content-independent to be a "soft precedent-weighting" issue a prompt or
replay tweak would plausibly fix. **Recommend piloting `mistralai/ministral-3-3b` as the
same-cost, immediately-available replacement candidate for this role** — it is the only
alternative tested here with a clean result on K-056's specific mechanism at a meaningful
sample size, requires no larger-model cost/latency trade-off, and is already loadable in this
environment. This is conditional, not unconditional: before routing K-054/K-055 (or re-routing
K-052/K-053) onto it, close or explicitly accept §8.4's duplicate-instruction defect — a
scoped, cheap follow-up eval targeting exactly that pattern (a small golden set of "does a
follow-up instruction ever re-trigger an earlier one" turns), not a full re-run of this pass.
**Do not pursue `gpt-oss-20b` further until its LM Studio-side crash is independently fixed**
(`devops`/environment scope, not `data-scientist`'s to action) — it remains a secondary,
currently-unusable option whose underlying capability this pass could not cleanly measure.
**If the team instead wants to proceed on the current model**, the shipped observability
signal (§8.3) is a legitimate, cheap ops-alerting backstop — 98% recall on the failure class
that matters, 0% false-positive rate — but it does not prevent the fabrication, only surfaces
it after the fact in logs/trace, and has a confirmed blind spot on bare (price-less)
write-mutation confirmations.

**What would sharpen this further, if the team wants more before deciding:** a Ministral probe
at the same n as the `qwen3-4b` baseline (40 conversations) would tighten the 0/176 result's
CI further and increase the chance of surfacing a rarer failure mode not yet seen; a working
(non-crashing) `gpt-oss-20b` re-probe, if `devops` resolves the LM Studio compatibility issue,
would let a genuine larger-model comparison happen at all. Neither is required to act on §8.5's
recommendation — the Ministral evidence already clears a much lower bar than "prove it's
perfect," which is "does it show the specific, already-confirmed-near-certain defect blocking
K-054/K-055 today," and the answer at n=176 is no.

### 8.6 Artifacts

Two throwaway scripts, not part of the shipped test suite, not committed, left in this
session's scratchpad: `ds_k056_eval.py` (the conversation-driver + per-turn ground-truth
harness, parameterized by model/condition/n via env vars) and `ds_k056_analyze.py` (Wilson-CI
aggregation over the driver's JSONL output). Five throwaway workspaces
(`ws:eval-k056{,-fallback,-ministral,-ministral2,-smoke}`) were `GRAPH.DELETE`d after this
pass; `ws:acme`/`reference` were never written to, and were independently re-verified in sync
(`verify_salesperson.sh acme`, `verify_catalog.sh`, `verify_workflows.sh acme`, all `OK`) before
finishing.
