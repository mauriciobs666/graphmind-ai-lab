# `salesperson` scaffold — live tool-call reliability diagnostic (D-1 follow-up)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-052 (M6) — informs
> K-053/K-054/K-055 sequencing; no dedicated K-item exists yet for this defect (see §5, a new
> item is proposed for `teco` to file)

## Verdict

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
