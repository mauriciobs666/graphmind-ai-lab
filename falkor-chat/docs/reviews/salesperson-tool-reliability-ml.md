# `salesperson` scaffold — live tool-call reliability diagnostic (D-1 follow-up)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-052, K-056, K-057 (M6) —
> informs K-053/K-054/K-055 sequencing

## Verdict

**Superseded by §8 (Pass 2, 2026-08-29) as the operative recommendation** — a controlled eval
(n=40 conversations / 280 turns) confirms the skip-and-fabricate mechanism is a **near-certain,
deterministic-onset capability ceiling for `qwen/qwen3-4b-2507`** in this role, not a
probabilistic soft-degradation a scaffold tweak can dampen (both scaffold-level mitigations
this note named — `tool_choice` forcing here, a replayed-history breadcrumb tried afterward —
are independently falsified; see §8's own recap). §8 also tested three alternative local
models and found one, `mistralai/ministral-3-3b`, with **zero** instances of this mechanism
across 176 further turns — a same-size-class, immediately-available candidate. It carries its
own distinct, real (but categorically less severe, intermittent, self-disclosing) defect,
scoped in a follow-up pass at §9: **not a blocker** — §9.5's verdict is Go, conditional on
disclosure and a near-term mitigation pass, not on closing it to zero first. Read §8 for the
numbers and §9 for the Ministral-specific follow-up; §1-§7 below are Pass 1's original
root-cause finding, left intact as the mechanism analysis that still holds. §10 is a separate,
not-yet-live-tested web survey of other small local models worth considering next.

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

**§8.5's condition on piloting Ministral — closing/accepting the duplicate-instruction
defect — is addressed by §9 below**, a scoped follow-up eval (not a re-run of this pass's own
baseline). Verdict: not a blocker, ship conditional on a named (untested) dispatch-time
mitigation or an explicitly accepted, disclosed low-rate risk — see §9.5.

### 8.6 Artifacts

Two throwaway scripts, not part of the shipped test suite, not committed, left in this
session's scratchpad: `ds_k056_eval.py` (the conversation-driver + per-turn ground-truth
harness, parameterized by model/condition/n via env vars) and `ds_k056_analyze.py` (Wilson-CI
aggregation over the driver's JSONL output). Five throwaway workspaces
(`ws:eval-k056{,-fallback,-ministral,-ministral2,-smoke}`) were `GRAPH.DELETE`d after this
pass; `ws:acme`/`reference` were never written to, and were independently re-verified in sync
(`verify_salesperson.sh acme`, `verify_catalog.sh`, `verify_workflows.sh acme`, all `OK`) before
finishing.

## 9. Ministral duplicate-instruction follow-up eval (2026-08-29)

**What this section answers, and for whom.** §8.4 surfaced a real, distinct defect on
`mistralai/ministral-3-3b` as a byproduct of the model-alternative check: a second cart
instruction sometimes caused the model to silently re-issue an earlier, already-completed
`add_to_cart` call — every call honest and dispatched, no fabrication, but a silently
inflated cart line. §8.5 made piloting Ministral **conditional** on closing or explicitly
accepting that defect first. This section is that scoped follow-up — not a re-run of §8's
own baseline, not a re-test of the (separately, already closed) K-056 mechanism — sized to
meaningfully narrow §8.4's n=10 estimate (Wilson 95% CI 10.8-60.3%) and to characterize the
pattern across the axes the coordinator asked for: instruction similarity, turn spacing, and
tool variety.

### 9.1 Method

Same in-process harness precedent as §2/§8 (`services.start_workflow_run`/
`resume_workflow_run` driven directly, `WorkflowTrigger(..., trace=True)`, real
`salesperson@v2`, real LM Studio) against a fresh throwaway `ws:eval-ministral-dup`
(`bootstrap_schema.sh` → `seed_demo.sh` → `seed_catalog.sh` → `seed_salesperson.sh`,
`EMBEDDING_DIM=1024`). `mistralai/ministral-3-3b` re-confirmed loaded/responsive via a direct
`/v1/chat/completions` probe immediately before starting (U40 had confirmed it minutes
earlier; re-checked fresh for this pass rather than assumed). Each conversation is fresh
(new customer id → its own `Customer`/`Cart` anchor via `ctx.actor`, new thread, new run) —
32 independent conversations across six conditions, each varying one axis against a
minimal 2-3-turn script built around the brief's general pattern ("does a follow-up
instruction ever cause the model to re-issue an earlier, already-completed tool call"):

| Condition | n | Turns | Tests |
|---|---|---|---|
| `distinct-immediate` | 6 | add Wireless Mouse Pro → add Portable SSD 1TB | §8.4's exact recipe (distinct categories: Peripherals/Storage), back-to-back |
| `same-category-immediate` | 6 | add Wireless Mouse Pro → add Mechanical Keyboard K200 | same category (both Peripherals), back-to-back |
| `distinct-spaced` | 6 | add mouse → *"What's my cart total so far?"* → add SSD | distinct categories, one intervening read-only turn |
| `same-category-spaced` | 6 | add mouse → intervening read-only turn → add keyboard | same category, one intervening turn |
| `remove-retrigger` | 4 | add mouse+SSD → remove mouse → *"Actually also remove"* SSD | a second **removal** instruction re-triggering the first removal, not an add |
| `place-order-retrigger` | 4 | add mouse → place order → add keyboard (new, post-checkout instruction) | a **different tool** (`place_order`, not cart-line writes) re-firing on an unrelated follow-up instruction |

**Tool-surface scope note (the brief's point (c)):** K-053's tool surface is cart/order-only
(`view_cart`, `add_to_cart`, `remove_from_cart`, `clear_cart`, `place_order` — no
`save_profile`/other non-cart write tool exists yet; K-054 is unbuilt). `place_order` is the
one write tool in reach that is mechanistically distinct from a cart-line add/remove (it
snapshots the cart into a new `Order` and clears it, rather than adjusting a `CartItem`
count) — used as the "different tool" condition rather than a hypothetical tool that doesn't
exist yet.

Two ground-truth signals per conversation, same discipline as §8.1 — neither trusts reply
text:
1. **Tool-dispatch ground truth** — the raw `TraceEvent` chain
   (`repository.read_trace`), flattened to the ordered sequence of `tool_calls` names +
   arguments actually dispatched across every LLM iteration of every turn in the run.
2. **State ground truth** — `Cart`/`CartItem` read directly via Cypher
   (`repository.read_cart`, byte-identical to §8.1's/D-1's own method) after the
   conversation completes, plus an `Order` count (`MATCH (c:Customer)-[:PLACED]->(o:Order)`)
   for the `place-order-retrigger` condition. A duplicate is scored **only** on this ground
   truth (an item's actual graph quantity exceeding what was ever explicitly requested, an
   extra `place_order` dispatch, or a non-empty post-clear cart) — never on the model's own
   reply wording.

All Wilson 95% CIs below use z=1.96, this lab's established convention.

### 9.2 Result

**32/32 conversations completed cleanly** (every run reached `waiting`, no crashes, no
harness errors) — a first useful data point on its own: Ministral did not exhibit K-056's
skip-and-fabricate mechanism, the runaway-repetition failure mode `tool_choice` forcing
triggered on `qwen3-4b` (§4.2), or `gpt-oss-20b`'s message-spam defect (§8.4) anywhere in
this sample, consistent with §8.4's own clean read on the *other* mechanism.

**One confirmed duplicate, independently re-verified against ground-truth Cypher and the raw
trace, not just the harness's own read** — `same-category-immediate` case #3
(customer `cust-9-643f5a`, run `d3b401d734e147fca1e47a3d28d38d8b`). Turn 1 ("Add 1 Wireless
Mouse Pro") correctly added one line; turn 2 ("Also add 1 Mechanical Keyboard K200")
**correctly added the keyboard on its first tool-calling iteration, then, on its own very
next iteration of the same turn, re-issued `add_to_cart(Wireless Mouse Pro, quantity=1)` a
second time** — a call whose target product is not mentioned anywhere in turn 2's own text.
Re-verified directly:

```
MATCH (cart:Cart {customerId: 'cust-9-643f5a'})-[:HAS_ITEM]->(item:CartItem)
RETURN item.productId, item.quantity, item.addedAt ORDER BY item.addedAt
→ wireless-mouse-pro | 2 | ...
   mechanical-keyboard-k200 | 1 | ...
```

— the customer asked for 1 mouse, ever, across the whole conversation; the graph shows 2.
The raw trace confirms the mechanism precisely: the re-issued call is byte-identical in
shape to the one U40 found (`add_to_cart({"productName": "Wireless Mouse Pro", "quantity":
1})`, fired a second time), and it happens **within the follow-up turn's own multi-iteration
tool loop**, not as a separate later message — the model's second turn does one correct,
newly-requested write, then spontaneously repeats an unrelated, already-completed one before
finishing that same turn.

Rates by condition (Wilson 95% CI, all wide at this n — no cell is large enough to stand
alone):

| Condition | Rate | Wilson 95% CI |
|---|---|---|
| `distinct-immediate` | 0/6 | 0.0-39.0% |
| `same-category-immediate` | 1/6 | 3.0-56.4% |
| `distinct-spaced` | 0/6 | 0.0-39.0% |
| `same-category-spaced` | 0/6 | 0.0-39.0% |
| `remove-retrigger` | 0/4 | 0.0-49.0% |
| `place-order-retrigger` | 0/4 | 0.0-49.0% |
| **pooled, add-retrigger conditions (first 4 rows, n=24)** | **1/24 (4.2%)** | **0.7-20.2%** |
| **pooled, all 32** | **1/32 (3.1%)** | **0.6-15.7%** |

**No axis in this pass shows a statistically distinguishable effect** — every per-condition
CI overlaps every other one, including the two axes the brief specifically asked to probe:
category-similarity (`distinct` 0/12 pooled vs. `same-category` 1/12 pooled, CIs
0.0-24.3% vs. 1.5-35.4% — overlapping) and turn spacing (`immediate` 1/12 pooled vs.
`spaced` 0/12 pooled, CIs 1.5-35.4% vs. 0.0-24.3% — overlapping). The one occurrence landing
in `same-category-immediate` is consistent with chance at this n, not a confirmed pattern —
reporting it plainly rather than reading a story into a single event. `remove-retrigger` and
`place-order-retrigger` (0/4 each) are too small to support any claim beyond "not observed at
this n" — worth a larger follow-up sample if the team wants more confidence specifically on
those two, non-add mechanisms.

**This does not contradict §8.4's 3/10 (30%) finding — the two studies do not measure the
identical quantity.** §8.4's condition B was a 7-turn script (add, add, view, remove, remove,
view, add) — several independent opportunities per conversation for this exact mechanism to
fire; this pass's four "add" conditions are deliberately minimal 2-3-turn scripts isolating
one add-after-add pair per conversation, to cleanly attribute a duplicate to the specific
follow-up instruction under test. A lower per-conversation rate on a shorter script with
fewer opportunities is the expected result of that design choice, not evidence the underlying
per-opportunity rate dropped. Pooling loosely — §8.4's 3/10 conversation-level finding and
this pass's 1/24 opportunity-level finding — both point at a real, non-negligible,
non-zero rate whose overlapping CIs (10.8-60.3% and 0.7-20.2%) are statistically consistent
with a shared true rate somewhere in the high-single-digits to twenties-percent range; this
pass narrows the estimate's floor (ruling out "vanishingly rare") without pinning down a
precise point value, which was the brief's own stated bar ("narrow U40's wide CI," not
"resolve it to a point estimate").

### 9.3 Severity, contrasted with K-056

**Categorically less severe than `qwen3-4b`'s K-056 fabrication, on two independent
grounds, not just a lower rate.** First, every occurrence here is **honestly grounded** — the
tool call is real, dispatched, and the reply is generated from the resulting (if wrong) cart
state, unlike K-056's zero-tool-call fabrication where the reply is disconnected from any
real state entirely. Second, and more consequential for real-world risk: **the error is
self-disclosing in the very reply the customer reads.** In the confirmed case, the assistant's
own turn-2 reply reads *"Your cart now includes: Wireless Mouse Pro × 2 ... Mechanical
Keyboard K200 × 1 ... Total: $149.97"* — the doubled quantity is stated plainly, not hidden,
giving an attentive customer (or a `view_cart`/checkout-review step) a real chance to catch
it before `place_order` freezes it into an `Order`. K-056's fabricated "successfully removed"
replies carried **no such signal** — nothing in the reply or the engine's guards indicated
anything was wrong. This is a materially different risk profile even before the rate
difference: intermittent-and-visible vs. near-certain-and-silent.

**It is still a real, disclosed risk worth closing, not one to wave away.** A customer who
does not re-read their own cart summary carefully (plausible — the summary in the confirmed
case is not visually flagged as anomalous, just longer than expected) could still complete
checkout on an inflated order. `place_order`'s own idempotency guard (a caller-minted
`order_id`, `repository.py:2936-2949`) does **not** cover this — each `place_order` tool call
mints a fresh id, so it only protects against a literal retried call with the same id, not
two independently-decided calls with different args/state; the `place-order-retrigger`
condition here (0/4) did not surface a duplicate *order*, but the sample is small and the
mechanism (an extra write dispatched for a state already-established earlier in the same
run) is the same shape.

### 9.4 Candidate mitigation (named, not implemented — same posture as §4.3/§8.3)

**A blind cross-turn duplicate-call suppression (dedup-by-signature) is not safe and should
not be the fix.** The naive fix — "if this exact `(tool, args)` pair was already dispatched
earlier in this run, skip it" — would incorrectly block a customer's own legitimate later
request to add the same product again (e.g. "add another mouse" three turns later is a real,
intended repeat, not a bug). Ruling this out explicitly, the same way §4.2 ruled out
`tool_choice` forcing by direct test rather than by assumption.

**A more targeted candidate, suggested by this pass's own trace evidence:** the confirmed
duplicate is, precisely, a write tool call whose resolved argument (the product name) does
not appear anywhere in the *current* turn's own raw user text — the customer's turn-2
message said "Mechanical Keyboard K200," never "Wireless Mouse Pro," yet the second
iteration's tool call targeted the mouse. A **dispatch-time sanity check**, immediately
before executing a write-mutating tool call: does the tool's own resolved target (here,
`productName`) appear (case/normalization-insensitive, mirroring the existing
`nameNormalized`/`categoryNormalized` precedent) in the current turn's own trigger/reply
text? If not, hold the call (surface it to the observability signal from §8.3, or require an
explicit confirmation round-trip) rather than silently dispatching it. This targets the
observed mechanism directly (an uninstructed write appended onto an otherwise-correct turn)
without needing cross-turn state tracking or risking a false-suppress on a genuine repeat
instruction. **Untested — a candidate, not a fix.** It needs its own implementation +
targeted eval (mutation-test: force an off-turn-text write, confirm the gate holds it) before
it should ship, per this note's own standing discipline of never blessing an unverified
mitigation (§4.2's `tool_choice` lesson).

### 9.5 Recommendation

**Go — pilot `mistralai/ministral-3-3b`, not blocked by this defect.** The duplicate-
instruction pattern is real (independently reproduced and ground-truth-confirmed again here,
not a one-off from §8.4 alone) and not to be dismissed, but at this combined sample
(§8.4's n=10 + this pass's n=32) it is **intermittent** (point estimates from 3.1% to 30%
depending on pooling, CIs overlapping and none anywhere near certainty) and **categorically
less severe** (honestly-grounded, self-disclosing in the reply) than the K-056 fabrication
ceiling this whole pilot exists to route around (near-deterministic, 87-100% CI, silent).
Treating a low-double-digit-at-worst, self-disclosing state error as a hard blocker while the
status-quo model carries a near-certain, silent fabrication risk would invert the actual risk
ordering.

**Conditional on disclosure and a near-term mitigation pass, not on closing this to zero
first.** Recommend: (1) proceed with re-pointing `salesperson@v2` at `ministral-3-3b` per the
already-agreed plan (K-052/K-053 QA re-run, then K-054/K-055); (2) file the §9.4 dispatch-time
sanity-check candidate as a follow-up implementation + eval item, not a pre-pilot gate — it is
cheap, targeted, and model-independent, but unproven; (3) disclose the residual risk
explicitly wherever the pilot decision is recorded (`docs/BACKLOG.md`/`docs/HISTORY.md`): a
low-but-nonzero rate of an extra, uninstructed write-tool call duplicating an earlier action
within a follow-up turn, currently uncaught by any guard (the shipped `_note_possible_
fabrication` signal, §8.3, does not and should not fire here — it targets ungrounded replies,
and every reply here is grounded).

**What would sharpen this further, if wanted before K-054/K-055 land:** a larger,
combined-condition sample (n≈40-60, weighted toward the `add`-conditions where the only
confirmed instance occurred) would tighten the pooled CI meaningfully; a live-implemented and
eval'd version of §9.4's gate would let its own effectiveness be measured directly rather than
argued from mechanism alone. Neither blocks the go decision above.

### 9.6 Artifacts

One throwaway script, not part of the shipped test suite, not committed, left in this
session's scratchpad: `ds_ministral_dup_eval.py` (conversation-driver + ground-truth harness
for the six conditions above; results written to a sibling
`ministral_dup_eval_results.jsonl`). `ws:eval-ministral-dup` was `GRAPH.DELETE`d after this
pass; `ws:acme`/`reference` were never written to, and were independently re-verified in sync
(`verify_salesperson.sh acme`, `verify_catalog.sh`, `verify_workflows.sh acme`, all `OK`)
before finishing.

## 10. Model landscape survey (web research, not live-tested)

**What this section answers, and for whom.** §8.4 live-tested four models already
loaded/available in this environment's LM Studio instance. The user separately asked: what
*other* small (~2-8B) local models does the current market/literature recommend specifically
for reliable tool/function calling — not general chat quality — that this lab hasn't tried yet?
This is a **survey, not a live test**: nothing below is verified against this stack, this
scaffold, or this failure mode. It exists to shortlist what a follow-up live-test unit should
try next, not to replace §8's live evidence.

**The one finding that should shape how every number below is read.** A model's published
tool-calling benchmark score and its behavior in a long, multi-turn conversation with replayed
history are **different questions**, and the gap between them is not a small-model quirk this
lab happened to discover — it is a documented, general-purpose-LLM phenomenon at every scale.
Laban et al., "LLMs Get Lost In Multi-Turn Conversation" (ICLR 2026 Outstanding Paper,
arXiv:2505.06120), tested 15 frontier and near-frontier models (GPT-4.1, Claude 3.7 Sonnet,
Gemini 2.5 Pro, and others) across single- vs. multi-turn task completion and found a **39%
average accuracy drop** and **112% higher unreliability** (same task, wildly different outcomes
run to run) purely from spreading the same instructions across a conversation instead of giving
them all at once — with four named mechanisms: premature answers on partial context, "answer
bloat" that compounds an early mistake instead of reconsidering, a lost-in-the-middle effect on
turns that aren't first or last, and **no self-correction after an early misstep** ("no
recovery"). That last mechanism — an early wrong turn, once made, is never recovered from for
the rest of the conversation — is structurally the closest published analogue to this lab's own
turn-4 collapse-and-never-recover finding (§8.2: 0/121 post-onset recoveries), even though the
paper's task suite is general instruction-following, not tool-calling specifically, and its
tested models are two to three orders of magnitude larger than `qwen3-4b`/`ministral-3b`. Read
together with this lab's own finding, this argues the mechanism (in-context precedent + no
recovery) may be a general LLM behavior that a 4B-class local model simply has far less headroom
to resist — not something unique to this stack.

**No benchmark or model card found in this survey tests the exact K-056 shape** — a model
completing a real conversational turn, with only *prior turns' final text* replayed (no tool-call
scaffolding, no explicit multi-step task framing), silently choosing prose over a tool call it
was instructed to always use. The closest published proxy is BFCL's own "multi-turn" category
(cited per-model below), which tests whether a model correctly chains several *dependent*
function calls within one scripted multi-step task — a real and relevant signal (a model that
can't track state across calls is unlikely to fare better at not skipping a call at all), but a
**different construct**: it does not vary conversation length as a hidden lever, does not test
whether the model reverts to un-grounded prose once it "feels" a rapport-like text exchange has
been established, and is run with the tool schema and instructions freshly re-presented every
step rather than diluted across many replayed plain-text turns the way `_assemble_messages`
does here. A high BFCL multi-turn score is evidence worth weighting, not proof a model would
avoid this lab's specific failure mode — the only way to know is to run this lab's own §8
harness against it, exactly as done for `ministral-3-3b`/`gpt-oss-20b`.

### 10.1 Candidates surveyed

| Model | Params | GGUF for LM Studio | License | Tool-calling evidence found | Notes |
|---|---|---|---|---|---|
| **Salesforce xLAM-2-3b-fc-r** | 3B | Yes (`Salesforce/xLAM-2-3b-fc-r-gguf`, official) | **CC-BY-NC-4.0** (non-commercial) | BFCL overall 65.74%; live/AST 81.03%/88.22%; **multi-turn 55.62%** — the highest multi-turn BFCL figure found among small models in this survey [xLAM-2 GGUF card](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r-gguf), [xLAM paper](https://arxiv.org/pdf/2409.03215) | Purpose-built "Large Action Model" line, specifically trained for agentic tool use (not a general chat model retrofitted). Same size class as `ministral-3-3b`. Non-commercial license is a real constraint if this ever leaves an internal/research posture — fine for this lab's current dev use, worth flagging before any productization. |
| **Salesforce Llama-xLAM-2-8b-fc-r** | 8B | Yes (`Salesforce/Llama-xLAM-2-8b-fc-r-gguf`) | CC-BY-NC-4.0 | Same xLAM-2 lineage/training, larger backbone (Llama 3.1 8B) | Above-band alternative if the 3B variant's capacity turns out to be the limiting factor rather than its training. |
| **IBM Granite 4.1-3b / 4.0-H-Tiny** | 3B (4.1) / 7B-with-1B-active MoE (4.0-H-Tiny) | Yes (`ibm-granite/granite-4.0-h-tiny-GGUF`, `unsloth/granite-4.1-3b-GGUF`) | **Apache-2.0** | Vendor claims "improved instruction-following and tool-calling" as a named 4.0/4.1 design goal, explicitly marketed for "function calling, simple RAG, fine-tuning on smaller GPUs" [IBM Granite 4.1 docs](https://unsloth.ai/docs/models/ibm-granite-4.1), [Granite 4.0 docs](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0) | No independent BFCL number found for this exact variant in this pass (vendor claim only, not third-party-verified) — weaker evidence than xLAM/Hammer, but the **only Apache-2.0-licensed candidate surveyed**, which matters if the non-commercial licenses above are ever a blocker. Worth a quick live smoke-test before investing in a full probe, given the thin evidence base. |
| **Microsoft Phi-4-mini** | 3.8B | Yes (multiple community GGUFs: `bartowski`, `unsloth`, `llmware`) | MIT (per Microsoft's usual Phi release pattern — verify at adoption, not independently re-confirmed here) | Marketed explicitly for "chat with function calling and tool use" as a named use case; 128K context [LM Studio model page](https://lmstudio.ai/models/microsoft/phi-4-mini) | Differently-trained (Microsoft, not Qwen/Mistral lineage) — a genuinely distinct architecture/training-data family from every model tried in §8.4, which is exactly the kind of diversity worth probing if the hypothesis is "this is a training-recipe trait, not a scale trait" (§8.4's own framing for why Ministral beat Qwen3-4B despite being smaller). No independent BFCL figure surfaced in this pass. |
| **Hammer2.1-3b / -7b** (MadeAgents) | 3B / 7B | Yes (`mradermacher/Hammer2.1-3b-GGUF`, `eaddario/Hammer2.1-7b-GGUF`) | **Qwen-research license** (non-commercial-leaning, restrictive — same posture concern as xLAM's CC-BY-NC) | BFCL 45.0 (3B variant) [Hammer BFCL/model card discussion](https://huggingface.co/MadeAgents/Hammer2.1-3b) | Built on Qwen2.5-Coder backbone with "function masking" specifically targeting spurious/hallucinated tool calls — thematically adjacent to K-056 (fabrication under uncertainty) but the technique targets *wrong-tool selection*, not *skip-tool-entirely-after-N-turns*; unclear a priori it transfers. |
| **Llama-3-Groq-8B-Tool-Use** | 8B | Yes (widely mirrored: `bartowski`, `lmstudio-community`, `QuantFactory`, others) | Meta Llama 3 Community License | Reported 89.06% BFCL — cited as the highest published small-class BFCL score in one practitioner roundup [localaimaster.com roundup](https://localaimaster.com/blog/best-ollama-models-tool-calling) | **Caveat on the number itself:** this is a mid-2024 (Groq/Glaive) release scored against an earlier BFCL leaderboard generation (v1/v2-era); BFCL's own methodology has been revised since (v3/v4 add multi-turn, hallucination-avoidance categories), so this figure is not directly comparable to the BFCL-v3/v4-era numbers quoted for Qwen3-4B/xLAM-2 above — flagged as likely favorable-vintage bias, not a like-for-like "best in class" claim. |
| **ToolACE-8B / Watt-Tool-8B** | 8B | Yes (community GGUF mirrors) | Llama-3.1 base license (research-community fine-tunes, terms generally follow the Llama 3.1 Community License) | Vendor/community claims of BFCL-v2 SOTA-for-size; no third-party number independently re-confirmed in this pass | Same caveat as Llama-3-Groq-8B-Tool-Use — BFCL-v2-era claims, not verified against current leaderboard revisions. |
| **Qwen2.5-Instruct (3B/7B)** | 3B / 7B | Yes (widely available) | Apache-2.0 (2.5 series; note this is a **different license posture** from the Qwen3 line already tested) | BFCL: 3B ≈ 35.7%, 7B ≈ 44.7% [BFCL-derived scores via arXiv survey citations] | **Deprioritized for follow-up testing**: both scores sit well below `qwen3-4b`'s own already-measured-clean-at-turn-1-through-3 BFCL figure (~62% overall per the `llm-stats`/`pricepertoken` BFCL leaderboard mirrors), and the model already in this failure mode's own lineage (Qwen3) is a newer, better-benchmarking generation from the same vendor — no reason to expect Qwen2.5 to do *better* on K-056's mechanism than Qwen3-4B already measured to fail near-100% of the time. Named because the brief asked for it explicitly, not because it's a promising candidate. |
| **Meta Llama 3.2 Instruct (3B/8B)** | 3B / 8B | Yes (widely available) | Meta Llama 3.2 Community License | Native pythonic/JSON tool-call format, "single, nested, parallel, and multi-turn function calling" per Meta's own release notes; anecdotal community reports cluster around ~80% task success in informal tool-calling write-ups (not a rigorous benchmark figure) [Novita blog](https://blogs.novita.ai/does-llama-3-2-support-function-calling/), [llama.cpp function-calling docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md) | Weakest evidence base of the group (no BFCL figure surfaced), but the most permissively-licensed, most widely-supported baseline in this size class — worth including as a broad-compatibility comparator, not a top pick on capability evidence alone. |
| **Google Gemma 3 family — general caveat, not a per-variant recommendation** | 1B/4B/12B/27B | Yes | Gemma license (permissive-ish, custom) | Gemma 3 **has no dedicated tool-call special tokens/training** — function calling is prompted-in, not natively trained, per Google's own developer-facing material; Google's response was a **separate, purpose-built fine-tune, FunctionGemma (270M)**, released specifically because "the number one request" from developers was native function calling that Gemma 3 itself lacks [Google FunctionGemma announcement](https://blog.google/innovation-and-ai/technology/developers-tools/functiongemma/), [Gemma function-calling walkthrough](https://www.philschmid.de/gemma-function-calling) | This corroborates and generalizes this lab's own quick sanity-check finding (`gemma-3-12b` failed even a trivial cold-context tool call, §8.4) — **it is plausibly a family-wide characteristic of Gemma 3, not a fluke of the 12B checkpoint specifically**, since the gap is architectural/training-recipe (no tool-call tokens at all) rather than a capacity threshold a smaller or larger Gemma-3 checkpoint would cross differently. Not worth probing further variants of Gemma 3 for this role; Gemma's newer generation (referenced in passing above as "Gemma 4" in one comparison piece) or the dedicated FunctionGemma line would be the vendor's own recommended path if this family is revisited — out of scope to verify further here. |

### 10.2 Shortlist for a follow-up live-test unit

In priority order, weighing (a) how close the available evidence gets to K-056's actual
mechanism, (b) license cleanliness, and (c) training-lineage diversity from what's already
failed/passed in §8.4:

1. **`Salesforce/xLAM-2-3b-fc-r` (GGUF)** — top priority. Same size class as the already-clean
   `ministral-3-3b`, but with the best multi-turn BFCL figure found (55.62%) and purpose-built
   agentic training rather than a general chat model repurposed for tool use. The CC-BY-NC-4.0
   license is not a blocker for this lab's current internal/dev posture but should be named
   explicitly if this ever ships past that.
2. **`ibm-granite/granite-4.1-3b` or `granite-4.0-h-tiny` (GGUF)** — second priority, specifically
   *because* its evidence is thinner (vendor claim, no independently-verified BFCL number) but it
   is the only Apache-2.0 candidate surveyed — worth a cheap smoke test (single-turn sanity check,
   same triage this lab already did on Gemma) before committing to a full §8-style probe, purely
   to see if a license-clean option is viable at all.
3. **`microsoft/phi-4-mini` (GGUF)** — third priority, chosen specifically for training-lineage
   diversity: every model tried in §8.4 so far is Qwen- or Mistral-family; Phi is a genuinely
   different vendor/recipe, which matters if the working hypothesis (§8.4) is "this is a
   training-recipe trait, not a parameter-count trait."
4. **`MadeAgents/Hammer2.1-3b` (GGUF)** — fourth, lower priority given the restrictive
   Qwen-research license and a BFCL score (45.0) below xLAM-2-3b's — include only if the team
   wants a same-size, differently-fine-tuned Qwen2.5-Coder-lineage comparator specifically because
   its "function masking" technique targets spurious tool-call behavior, thematically (not
   mechanistically) adjacent to K-056.

**Not recommended for follow-up testing:** Qwen2.5-Instruct (weaker BFCL than the
already-tested, already-failing Qwen3-4B — no reason to expect improvement), any further Gemma 3
size variant (family-wide caveat above), and the 8B-class BFCL-leaderboard-topping models
(Llama-3-Groq-8B-Tool-Use, ToolACE-8B, Watt-Tool-8B) unless the team decides the "larger model"
path is back on the table — their BFCL numbers are on a vintage generation of the leaderboard not
directly comparable to the v3/v4-era figures quoted for xLAM-2/Qwen3/Ministral above, so they're
weaker evidence than they look at first glance, and this lab's own §8.4 already found the
one larger model it could actually load (`gpt-oss-20b`) blocked by an unrelated serving-stack
bug rather than a capability question — an 8B live probe would need its own LM Studio
compatibility check first, same caveat.

### 10.3 Sources

- Laban, Hayashi, Zhou, Neville — "LLMs Get Lost In Multi-Turn Conversation," ICLR 2026
  Outstanding Paper, [arXiv:2505.06120](https://arxiv.org/abs/2505.06120) (fetched via
  [secondary summary](https://beam.ai/agentic-insights/iclr-2026-llms-lose-accuracy-in-multi-turn-conversations))
- Berkeley Function-Calling Leaderboard (BFCL) v3/v4 — [Gorilla project leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), score mirrors via [llm-stats.com](https://llm-stats.com/benchmarks/bfcl) and [pricepertoken.com](https://pricepertoken.com/leaderboards/benchmark/bfcl-v3)
- xLAM: Zhang et al., "xLAM: A Family of Large Action Models to Empower AI Agent Systems," [arXiv:2409.03215](https://arxiv.org/pdf/2409.03215); model cards: [xLAM-2-3b-fc-r-gguf](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r-gguf), [Llama-xLAM-2-8b-fc-r-gguf](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r-gguf)
- IBM Granite 4.0/4.1 docs: [Granite 4.0 tutorial](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0), [Granite 4.1 docs](https://unsloth.ai/docs/models/ibm-granite-4.1), model card: [granite-4.1-3b-GGUF](https://huggingface.co/unsloth/granite-4.1-3b-GGUF)
- Phi-4-mini: [LM Studio model page](https://lmstudio.ai/models/microsoft/phi-4-mini), [bartowski GGUF](https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF)
- Hammer2.1: [MadeAgents/Hammer2.1-3b model card](https://huggingface.co/MadeAgents/Hammer2.1-3b), [Hammer paper](https://arxiv.org/html/2410.04587v2), GGUF: [mradermacher/Hammer2.1-3b-GGUF](https://huggingface.co/mradermacher/Hammer2.1-3b-GGUF)
- Llama-3-Groq-8B-Tool-Use / ToolACE-8B / Watt-Tool-8B / Mistral-7B-v0.3 roundup:
  [localaimaster.com, "Best Local LLMs for Tool & Function Calling (2026 Tested)"](https://localaimaster.com/blog/best-ollama-models-tool-calling)
- Qwen2.5-Instruct BFCL figures: cited via arXiv survey papers referencing BFCL score tables
  (e.g. [R2IF](https://arxiv.org/pdf/2604.20316), [FunReason](https://arxiv.org/pdf/2505.20192))
- Llama 3.2 tool-calling support: [Meta/Novita blog walkthrough](https://blogs.novita.ai/does-llama-3-2-support-function-calling/), [llama.cpp function-calling docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
- Gemma 3 tool-calling caveat / FunctionGemma: [Google FunctionGemma announcement](https://blog.google/innovation-and-ai/technology/developers-tools/functiongemma/), [Gemma function-calling walkthrough](https://www.philschmid.de/gemma-function-calling)

## 11. K-057 compound-filter answer-conflation diagnosis (2026-08-30)

**What this section answers, and for whom.** DEF-01 (`docs/test-reports/
workflow-nl-query-generation2-report.md`) found `@assistant Which peripherals cost less than
$60?` gave a correct, complete answer on one live attempt and a self-contradictory, incomplete
one on an identical repeat — the failing attempt called both `filter_products` and
`query_graph_data`, the passing one called `filter_products` alone. K-057
(`docs/BACKLOG.md`) filed this as worth a controlled rate estimate before acting, and named a
leading hypothesis: an **orchestration-layer** conflation of two tool results, resting on the
premise that `filter_products`'s category filter "has no price predicate." This section
**tests that hypothesis directly against live code and a live-reproduced sample, rather than
accepting it** — the same discipline this note's own §8/§9 and
`docs/reviews/workflow-nl-query-generation-rca.md` apply elsewhere in this codebase. The
premise turns out to be **factually wrong**, and the dominant failure mode is a **different,
larger, and previously unmeasured defect**.

### 11.1 Premise check (static, before any live run)

`FilterProductsTool.schema` (`server/falkorchat/tools.py:450-479`) declares `minPrice`/
`maxPrice` as ordinary optional parameters alongside `category`; `services.filter_products` →
`repository.filter_products` (`server/falkorchat/repository.py:2724-2769`) compiles all three
into one sargable `WHERE` clause (`p.price >= coalesce($minPrice, -1.0) AND p.price <=
coalesce($maxPrice, 1e9)`). `git log -p` on this method's introduction (commit `14891c9`, K-052)
shows `minPrice`/`maxPrice` present from the tool's very first version — **not** a capability
added later. **`filter_products` has always been able to express a compound category+price
predicate; the backlog's stated reason two tools get called is not supported by the code.**
This alone does not tell us why the model sometimes calls `query_graph_data` too, or why the
combination sometimes goes wrong — that needs the live probe below — but it rules out "the
model reaches for a second tool because the first one structurally can't do the job" as the
mechanism, before spending a single live LLM call on it.

### 11.2 Method

Same in-process harness precedent as §2/§8/§9 (`services.start_workflow_run` via
`WorkflowTrigger.maybe_trigger`, `trace=True`), but with one **load-bearing departure from
§8/§9's own convention**, not a stylistic choice: a **real `ModelGateway.from_env()`**, not the
`StaticModelGateway` single-injected-client sugar §8/§9 used. Reason: `v4` (unlike the `v2`/
`v2.1` defs §8/§9 tested) adds `query_graph_data`, and `QueryGraphDataTool.run()` resolves its
own internal structured-completion call via `self._models.llm("step", ws=ctx.ws)` with **no
`requested=` override** — under a real gateway this resolves to the shared `step`-kind role
**default** (`config/models.json`: `"step": "lmstudio/qwen/qwen3-4b-2507"`), independently of
the `assistant` step's own `config.model` pin to `mistralai/ministral-3-3b`
(`_run_agent_node` passes `requested=config.get("model")` only for its *own* resolution,
`server/falkorchat/executor.py:741-745`). A `StaticModelGateway` wrapping one injected client
would silently erase this exact divergence and could not have surfaced what §11.4 below found.
Confirmed by reading `app.py:361-393` (production wires exactly one shared `ModelGateway` across
`build_builtin_registry` and `WorkflowExecutor`) and `tests/eval/run_nlq_golden_set_eval.py`
(the golden-set harness itself already uses a real `ModelGateway.from_env()` for the identical
reason, one more precedent for this being the right seam, not a novel choice).

`FALKORCHAT_OPENCODE_CONFIG` was pointed at the repo's own `config/opencode.example.json`
(`localhost:1234/v1`) rather than the shared `$HOME/.config/opencode/opencode.json`, whose
`lmstudio` provider resolves to a LAN host (`192.168.0.69:1234`) that did not answer in this
session (`curl` returned nothing before timeout; `localhost:1234/v1/models` answered
immediately with both `mistralai/ministral-3-3b` and `qwen/qwen3-4b-2507` loaded) — the same
override `run_nlq_golden_set_eval.py` already uses, for the same documented reason, not an
ad hoc substitution. `FALKORCHAT_MODEL_CONFIG` was left at its shipped default
(`config/models.json`) — real production overlay, unmodified.

Fresh throwaway `ws:ds-k057` (`EMBEDDING_DIM=1024 bootstrap_schema.sh` → `seed_demo.sh` →
`seed_catalog.sh` → `seed_salesperson.sh`), real `falkordb-dev`. **20 independent
conversations** (backlog's own suggested 10-20 range, upper bound taken given how much
behavioral diversity turned up early), each a fresh customer id / thread / run, posting
DEF-01's exact reproduction text verbatim: `@assistant Which peripherals cost less than $60?`.
A thin `LoggingToolRegistry` (subclasses the shipped `ToolRegistry`, overrides `dispatch` to
record every call's full, untruncated arguments and raw JSON result before returning) supplied
the ground-truth seam the live REST surface lacks and DEF-01 itself flagged as missing — the
executor's own trace payloads truncate tool results at 200 chars (`executor._short`), too short
to ground-truth a 6-8-item JSON list, so this wrapper (not the trace) is this pass's primary
evidence source; `repo.read_trace` was still read per rep as a cross-check. Ground truth for
the question itself, re-verified live immediately before the run:

```
MATCH (p:Product) WHERE p.category='Peripherals' AND p.price<60 RETURN p.name, p.price
→ Gaming Mouse Pad XL 19.99 · Wireless Mouse Pro 29.99 · Webcam HD 1080p 59.99
```

`ws:ds-k057` was `GRAPH.DELETE`d after the pass; `reference`/`ws:acme` were never written to and
were independently re-verified in sync (`verify_salesperson.sh acme`, `verify_catalog.sh`,
`verify_workflows.sh acme`, all `OK`) before finishing. Script: `ds_k057_probe.py`, throwaway,
not committed, left in this session's scratchpad, parameterized by `K057_N`.

### 11.3 Result

**Both tools were called in 4/20 runs (20%, Wilson 95% CI 8.1-41.6%) — DEF-01's own observed
pattern, reproduced, but not the majority behavior and not, on inspection, driven by a belief
that `filter_products` lacks a price predicate.** In all 4, the model's **first** `filter_products`
call used `{"minPrice": 60}` — the *complement* of the question (products at $60 **and up**),
an apparent direction/framing slip, not a missing-capability workaround (three of the four calls
that followed reused `minPrice`/`maxPrice` correctly). `query_graph_data` was reached for
**after** that mis-scoped first call, consistent with a self-correction/second-opinion
attempt, not a belief filter_products is inadequate for the compound shape.

**The dominant, previously unmeasured failure mode is a numeric-boundary translation error
in `filter_products`'s own arguments, independent of whether `query_graph_data` is ever
called.** Across single-`filter_products`-call runs (`n=16`), the model split cleanly into two
groups by **one number**:

| `maxPrice` argument used | n | Reply |
|---|---|---|
| `59.99` (correct: ≤ the boundary item's real price) | 7 | All 3 items, correct, complete |
| `59` (wrong: excludes the $59.99 boundary item) | 9 | Only 2 items — `Webcam HD 1080p` silently dropped |

**9/16 (56.2%, Wilson 95% CI 33.2-76.9%) of single-tool-call runs were wrong purely from this
rounding**, with zero orchestration involved — no second tool, no conflicting result to
synthesize. `filter_products`'s own schema documents `maxPrice` correctly ("only products
priced at or below this amount"); the model's own translation of "less than $60" into that
inclusive bound is the unreliable step, not the tool's semantics or its capability. Every
single-call reply, right or wrong, correctly narrowed to genuinely-`Peripherals` items **despite
`category` never once being passed** (0/20 runs passed `category` to `filter_products` at
all) — the model reliably re-derives the category filter from its own knowledge of the returned
rows' `category` field at synthesis time in every observed case, so category omission was not,
in this sample, itself a source of wrong answers — only the price-boundary translation was.

**DEF-01's exact self-contradiction shape reproduced once: 1/20 (5.0%, Wilson 95% CI
0.9-23.6%).** Full trace, `ws:ds-k057` rep 5 — `filter_products({"minPrice": 60})` [wrong
direction] → `filter_products({"category": "Peripherals", "minPrice": 60})` → correctly returns
only `Mechanical Keyboard K200` ($89.99, genuinely ≥$60) → `query_graph_data("How many products
are there in the catalog?")` → correctly `15` → `filter_products({"category": "Peripherals",
"maxPrice": 60})` → correctly returns all 3 ground-truth items → **final reply: "There are no
peripherals priced under $60 in the catalog. Here's what we do have under $60: [lists all 3
correct items]."** The mechanism is not "two tool results synthesized badly" (there is no
`query_graph_data`-vs-`filter_products` conflict here — `query_graph_data` answered an unrelated
count question, cleanly) — it is **failure to revise an earlier, self-stated conclusion once a
later, correct tool call inside the same turn's own multi-iteration loop contradicts it**. This
is the same "no self-correction after an early misstep" mechanism §10's own literature survey
names (Laban et al., arXiv:2505.06120) as a general, scale-independent LLM behavior — observed
here, for the first time in this note's own evidence, **within one turn's tool loop**, not
across conversation turns as §8.2's collapse-and-never-recover finding showed.

**`query_graph_data` itself failed both times it was asked a live paraphrase of the exact
reproduction question, independently of the outer model's synthesis.** Rep 7:
`query_graph_data("Which peripherals cost less than $60?")` → `{"items": [], "finding": "no
matching data found"}` — wrong; the correct 3-row answer exists and `filter_products`'s own
concurrent, correctly-scoped call in the same run proves it. Rep 20:
`query_graph_data("Which peripherals are priced under $60?")` → same abstention, same wrong.
**This does not contradict `docs/test-reports/workflow-nl-query-generation-report.md`'s 100%
compound-filter accuracy (n=3) — it exposes a coverage gap the backlog's framing did not
account for.** Reading `server/tests/eval/nlq_golden_set.jsonl` directly: the three
`compound-filter`-shape catalog pairs (`nlq-10/11/12`) use thresholds ($50, $30, $200) that sit
comfortably clear of any product's actual price — **none is a boundary-adjacent case the way
DEF-01's $60-vs-$59.99 pairing is**. "100% on this exact shape" was true for the shape
taxonomy label the golden set tests; it was never evidence about boundary-adjacent thresholds
specifically, which is exactly the sub-case DEF-01's own reproduction question happens to be.
Both live occurrences of this exact abstention rebut "not a `querygen` DSL defect on current
evidence" as originally stated in the backlog — it should read "not evidenced as a DSL defect by
the golden set's existing cases," a narrower and more accurate claim. Both `query_graph_data`
abstentions were also, in both runs, correctly ignored by the outer model in favor of a
concurrent correct `filter_products` result (rep 7 gave the fully correct answer despite the
abstention; rep 20 gave a still-incomplete-but-non-contradictory answer, itself a `maxPrice=59`
rounding case) — so this abstention defect did not itself cause DEF-01's self-contradiction
pattern in this sample, but it is a real, independently-confirmed, previously-unflagged
`query_graph_data` reliability gap on a shape the golden set does not cover.

Also observed, off the reproduction question's exact path but on the same live sample: rep 4's
`query_graph_data("How many products are in the peripherals category?")` returned
`{"count(p.name)": 0}` — wrong (ground truth is 4, per
`docs/test-reports/workflow-catalog-lookup-report.md` TP-03) — an aggregate-with-category-filter
shape the golden set's own `nlq_golden_set.jsonl` does not carry at all (its `aggregation` pairs
are unfiltered counts). Named for completeness, not scored into any rate above — one occurrence,
outside DEF-01's own question shape, flagged as a candidate for the golden set's own future
expansion rather than something this pass sizes.

**Net correctness: 9/20 fully correct and complete (45.0%, Wilson 95% CI 25.8-65.8%); 11/20
wrong in some way (55.0%, Wilson 95% CI 34.2-74.2%), of which 10/20 (50.0%) are the silent
boundary-drop and 1/20 (5.0%) is DEF-01's self-contradiction shape.** These two failure modes do
not overlap in this sample (the one self-contradictory run's underlying `filter_products` calls
used the *correct* `59.99`-equivalent boundary once corrected — its problem was the
uncorrected earlier statement, not the final numbers).

### 11.4 Severity assessment

**Wrong, but not orchestration-layer in the sense the backlog named, and not primarily about
`query_graph_data` at all.** The single largest, best-evidenced defect (50% of all runs, present
in the *majority* of `filter_products`-alone runs with zero second tool involved) is
`mistralai/ministral-3-3b`'s own unreliable conversion of a natural-language "less than $X"
into `filter_products`'s documented-inclusive `maxPrice` bound — a **model-capability /
prompt-guidance gap**, not a tool defect, not a synthesis-of-two-results defect, and not fixable
by steering the model away from `query_graph_data` (the backlog's own named candidate angle):
that fix targets a mechanism (tool selection) that is not what causes most of the wrong answers
observed here. DEF-01's own self-contradiction pattern is real (reproduced once, ground-truth
confirmed) but is the **minority** failure mode in this sample (5% vs. 50%), and its own
mechanism — failure to revise an earlier stated conclusion after a later corrective tool call in
the same turn — is closer in shape to §8.2's "no recovery" finding and to K-058's
duplicate-instruction pattern (an uncorrected earlier action/belief persisting past a later,
correct one) than to a two-results-conflated-badly story. Severity: **MAJOR**, same as DEF-01's
own QA-assigned severity — a customer asking this ordinary question gets a wrong or incomplete
answer roughly half the time on the current model/scaffold, materially worse than DEF-01's own
1-of-2 anecdote suggested, just for a different reason than DEF-01 guessed.

### 11.5 Recommendation

**Fix now, and the fix is `systemPrompt`/tool-description wording, not a scaffold or model
change — route as a fast, narrow follow-up, not a full architect/coder cycle.** Two independent,
targeted wording additions, each aimed at a distinct confirmed mechanism above:

1. **Boundary-translation guidance (targets the 50% failure — priority one).** Add one sentence
   to `filter_products`'s tool description or the `assistant` step's `systemPrompt` making the
   inclusive-bound semantics and the translation rule explicit, e.g.: *"`minPrice`/`maxPrice` are
   inclusive ('at or above'/'at or below'). For 'less than $X', use a `maxPrice` just under X
   (e.g. `X - 0.01`), never round down to the nearest whole dollar — a product priced at exactly
   `X - 0.01` must still match."* This targets the mechanism directly confirmed by trace evidence
   (§11.3's `59` vs. `59.99` split), not a guess dressed as a finding.
2. **Non-revision guidance (targets the 5% shape — priority two, smaller payoff but cheap to
   bundle).** A short instruction not to state a conclusion before every planned tool call for
   the turn has returned, e.g.: *"Do not tell the customer a filter came up empty or that
   nothing matches until you have made your last catalog lookup for this question — if an
   earlier attempt looked empty or wrong, a later corrected one wins; never report both."*
3. **Do not adopt the backlog's originally-named candidate** ("steer `systemPrompt` toward
   `query_graph_data` alone once a question carries a price predicate `filter_products` can't
   express") **as written** — its premise is false (§11.1) and it would not address either
   confirmed mechanism above; §11.3 shows `filter_products` alone is wrong more often (56.2% of
   its own single-call runs) than the two-tool path is self-contradictory (25%, 1/4) — steering
   *toward* single-tool use without also fixing the boundary-translation bug would not help, and
   could plausibly remove the self-correction path that let reps 4/7 recover to a correct answer.

**Evaluation to confirm a fix, before shipping:** re-run this section's own harness
(`ds_k057_probe.py` pattern — throwaway workspace, `LoggingToolRegistry`, DEF-01's exact
question) at the same or a larger n against the `systemPrompt`-patched `v5` def, scoring the
same three outcomes (correct-complete / silent-boundary-drop / self-contradictory) plus the
`maxPrice` value actually dispatched on every single-call run. **Acceptance bar:** the `59` vs.
`59.99` split should collapse to ≥90% `59.99`-or-equivalent-correct at n≥20 (a one-sided
improvement claim against this pass's 43.8% baseline is easy to detect at this n — Wilson lower
bound at, e.g., 18/20 correct is 71.7%, cleanly above the 43.8% point estimate measured here);
the self-contradiction shape should not reappear in the same n (it was already rare — absence at
n=20 is expected either way and not itself proof of a fix, only consistent with one; a
larger confirming sample, n≈40-60, is worth running only if the team wants to distinguish
"fixed" from "still ~5%, just not observed again," not required to ship). A **golden-set
addition** is also recommended, independent of the prompt fix: add one `compound-filter` catalog
pair at a genuine boundary-adjacent threshold (mirroring DEF-01's $60-vs-$59.99 pairing) to
`nlq_golden_set.jsonl`, closing the coverage gap §11.3 identified — this is cheap, durable, and
prevents this exact gap from recurring for the next capability that reads "100% on this shape"
as blanket assurance.

### 11.6 Artifacts

One throwaway script, not part of the shipped test suite, not committed, left in this session's
scratchpad: `ds_k057_probe.py` (`LoggingToolRegistry`-instrumented conversation driver, real
`ModelGateway.from_env()`, `K057_N` env-parameterized rep count; results in a sibling
`k057_probe_results.jsonl`). `ws:ds-k057` was `GRAPH.DELETE`d after this pass;
`ws:acme`/`reference` were never written to and were independently re-verified in sync
(`verify_salesperson.sh acme`, `verify_catalog.sh`, `verify_workflows.sh acme`, all `OK`) before
finishing.
