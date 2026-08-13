# LM Studio / local small-model realism — this lab's stack

> **Live-verified knowledge base for `data-scientist`.** Facts confirmed by direct testing against
> this lab's LM Studio server, not assumed from model cards. Model behavior across versions/quants
> is perishable — treat entries as **verified for the cited model tag and date**, re-check before
> leaning on one for a live decision.
>
> **This is a cache, not the source of truth.** Origin: distilled 2026-08-11 from the
> `data-scientist` agent's learnings inbox via `agent-maintenance` skill §5.

## On a terminal tool-call schema, Ministral-3B was MORE reliable than Qwen3-4B — native `tool_calls` vs. prose

Direct replay of the answer-node `post_message` schema against LM Studio (`:1234`):
`mistralai_ministral-3-3b-instruct-2512` emitted a native OpenAI `tool_calls` `post_message` 3/3
draws (parsed cleanly by this lab's `llm.py`); `qwen/qwen3-4b-2507` emitted plain prose with **no**
tool call 3/3 draws. LM Studio's OpenAI-compat layer surfaced Ministral's tool call correctly on
this path — a documented risk that "Mistral's tool-call format won't parse" did **not** materialize
on the native path.

**Consequence:** the naive prior "smaller parameter count ⇒ worse at structured tool calls" did not
hold for this specific pair on this specific schema — verify per model/schema rather than ranking
models by size alone for tool-calling reliability. (Ministral never reached the live answer node in
the coordination this was observed in, so this is a capability-probe datapoint, not a banked
production result — re-verify before treating it as decided.)

**Context:** K-022 D13 Qwen-vs-Ministral capability probe (falkor-chat live-triage reliability
work), classifying Defect-C / D4 (genuine no-post).
