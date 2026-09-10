# A raising `ToolEnvironment.dispatch` — record, refuse, or something else

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** —

## 1. The question and the decision it serves

`docs/reviews/small-model-benchmarking-impl.md` **P17-3** and its **§6 open question 1**: when a
pack's `dispatch(name, arguments)` raises an unexpected exception, should `drive` abort (pack
defect, fail closed) or record the call as *undispatchable* and drive past it (§4 S2's replay
clause, `-ml` §4.1's never-skip-a-turn rule)? The decision blocks the future unit that writes
`tools/sim.py` (not yet dispatched — a teco mislabeling in this unit's dispatch brief called it
"U106," but that id is already assigned in the coordination ledger to a different, unrelated unit:
`packs.py`'s manifest→`PromptConfig` validation route), because a sim author writes `raise KeyError`
without thinking about it. Naming the exception is already in flight and is independent of this;
what follows is a small addition on top of that naming.

**Ruling, in one line: neither, as filed.** Record the failure, keep every observation already
taken, and **end the conversation at that turn** — the conversation, not the turn and not the run —
routing it to `-ml` §4.1's **`unrunnable`**, never to *undispatchable*. **No sixth disposition
member is needed.** The primary defence is upstream of all of this: a sim's `dispatch` is
**total**, and after that rule the ruling governs a residual rather than a design path.

## 2. Findings from the real system

1. **`undispatchable` is a model-failure bucket by construction, and its downstream effect is
   scored.** Its three reasons are `missing-function-name` and `unparseable-arguments`
   (`convo.py:404`, `:406`) — both the **model's** failure — plus `no-dispatch-record` (`:410`), an
   environment-contract fallback P17-2's fix makes unreachable from `drive`. So the bucket as built
   means *the model emitted a call that cannot be executed*. An undispatchable call
   contributes nothing to `E(t)`, so a turn whose only call was undispatchable has `|E(t)| = 0` and
   partitions as **`no_attempt`** under `-ml` §4.2(a) — *a failure charged to the model*. Putting a
   pack's `KeyError` there records a well-formed native tool call as *the model did not attempt
   one*. That is §4.3's laundering rule violated in the direction the note exists to prevent, it is
   silent, and — the caller's own concern, arriving inside the option meant to answer it — it is
   **model-correlated and signed**: it penalises exactly the arm more likely to trigger it.
2. **The environment is stateful per conversation and its state after a raise is unknown.** §3.8.4
   makes the sim *"deterministic and stateful within a conversation"*, and FR-10's ground truth is
   the dispatch trace plus the final state. A raise part-way through `add_to_cart` may or may not
   have mutated the cart and may or may not have recorded a `DispatchRecord`; `drive` cannot tell
   (this is the same class `TraceContractViolated` was built for). Every later turn of that
   conversation is then measured against ground truth the harness cannot vouch for.
3. **The plan has already ruled the isomorphic case, in the model-caused direction.** `-ml` §4.3
   rule 4 routes `no-response`/`server-rejected` to `unrunnable` on the discriminator *the harness
   holds no observation of the model at all* — and it explicitly contemplates a **model-caused**
   harness failure: *"a `400` produced by a model's runaway message list and a `400` produced by a
   malformed harness payload are the same status code"*. §8.4's `gpt-oss-20b` lost **6 of 8**
   conversations that way. A `KeyError` from a sim that did not anticipate a model-chosen argument
   is that same phenomenon one channel over: partly model-caused, unattributable, no observation.
4. **The machinery that makes such a loss honest is already specified and already gated.** Rule 5's
   censoring (risk set, `c_t`, observed `n`), the headline's third state, the funnel head, the
   paired intersection and `asymmetry`, and `verdict()`'s refusal below `observable_floor`
   recomputed on the *surviving* `n` (§3.4 Rule 7). Nothing new is required to keep a censored
   conversation visible.
5. **A pack fix cannot be mixed with pre-fix data.** `tools/sim.py` is inside `packContentHash`
   (§3.3), which is `REQUIRED_NONEMPTY` and drives AC-3's mismatch banner. So fixing a sim forces
   both arms to be re-run — which is what stops the fix-and-rerun loop from becoming an
   arm-conditioned researcher-degrees-of-freedom channel.

## 3. The caller's hypothesis, tested

*"Unconditional abort makes the harness's failure mode correlate with model quality and filters the
sample toward stronger models."* **Directionally right, materially right at run granularity,
wrong about the remedy.**

Right: the per-call hazard is arm-dependent, so the probability of producing a run at all is a
decreasing function of arm weakness — non-ignorable missingness at the run level, and a
survivorship sample across models.

Its size, and this is the only assumption-free part: **the blast-radius ratio is exactly 12:1** —
for any per-call hazard `p`, aborting the run loses ~12× what aborting the conversation loses,
because the pack is 12 scripts × 1 replicate and the environment is per-conversation.
*Illustrative arithmetic on an assumed `p`, not a measurement*: §4.3 rule 3's funnel shape (167
dispatched calls / 360 turns) over §4.5.2's ~80 turns per run puts ≈37 dispatched calls in a run
and ≈3 in a conversation, so `1 − (1 − p)^K` gives P(run lost) ≈ 17% against P(conversation lost)
≈ 1.5% at `p = 0.005`, ≈53% against ≈6% at `p = 0.02`. The first real `tool-caller` run replaces
these with an observed count.

Where I overrule it: **run-level abort is loud, and record-as-undispatchable is not.** A lost run
produces no artifact and no comparison, so it cannot be misread as a measurement; finding 1 can be,
and is signed against the weaker arm. Between a bias you cannot see and a refusal you cannot miss,
the refusal is the safer error — so "record it" does **not** follow from "abort is biased". What
follows is: shrink the abort to the unit actually contaminated (the environment, i.e. the
conversation) and route the loss through machinery that already prices it.

## 4. The recommendation

**(a) The sim-author rule — the load-bearing half. `dispatch(name, arguments)` is total over
`(str, dict)`: it never raises on anything the model can produce.**

- Every input-shaped problem is a **returned value**, not an exception — unknown tool name, missing
  or extra or wrong-typed argument, a product not in the catalog, quantity `0`, removing an item
  that is not in the cart. Return `{"error": "<code>", ...}` and record the `DispatchRecord` like
  any other. A returned tool error is *the thing the pack measures* (does the model recover?); a
  raise is the harness admitting it cannot execute the turn.
- `raise` is reserved for a condition that **does not depend on the model's arguments at all** (the
  catalog file failed to load; an internal invariant broke).
- One-line test the author can apply without re-deriving any of this: **if the model's arguments
  can change whether it raises, it must not raise.**
- Corollary for `tools/sim.py`'s future author and for `drive`: **do not schema-validate arguments before dispatching.**
  `drive` dispatches anything that parses as a JSON object (`convo.py:_parse_tool_arguments`), and
  it must keep doing so — refusing a wrong-typed argument would make it *undispatchable*, hence
  `|E(t)| = 0`, hence `no_attempt`, which destroys FR-8(d)'s measurement of argument correctness.
  The sim therefore tolerates schema-invalid input; it does not get to assume validation.

**(b) When it happens anyway.** `drive` wraps the `env.dispatch` call site, raises the named class
(the half already being implemented) carrying **the completed turns' `ConversationTrace`**, the
turn index, the tool name and the parsed arguments, and the original exception. The runner catches
it, stores the conversation **censored at `t`** — turns `1 … t−1` kept, `t` and later absent — and
proceeds to the next script with a fresh environment. Rule 5's censoring then applies unchanged:
out of the hazard's risk set at `≥ t`, out of §4.4's per-position `n` at `≥ t`, and — with `H = 4`
— out of the headline's denominator and into its `n/a` tally when `t ≤ H`.

**One deliberate divergence from rule 5**, and it needs stating because it is the only place this
ruling is not a straight application: rule 5's carve-out keeps §4.2's turn-level counts (a)–(g) for
turns after an `unrunnable` one, on the reasoning that *a turn answered against a shortened history
is still an observation*. That reasoning is about **history** contamination and does not survive
**state** contamination — after a raise, FR-10's ground truth itself is unreliable, so (a)–(g) are
not observations either. Not driving those turns is the cheapest correct implementation of that,
and it also returns their inference budget, which §4.5.2 shows is the binding constraint.

**(c) Why no sixth disposition member.** `TurnDisposition` is a vocabulary for *how a turn's model
loop ended*; all five members name an LLM-channel event. A dispatch raise is **conversation-scoped**
— it invalidates the environment, not just the turn — so naming it as a turn mechanism is the
two-vocabularies collision this plan has refused three times (P13-1's `unrunnable`-as-member,
P14-1's fourth column, P15-9's fifth `outcome` member). It also does not fit: a turn could both
reach the cap and have had a raise, and one enum slot cannot carry both. Carrying it as a
**conversation-level censoring marker** keeps the closed enum closed, keeps its guard untouched,
and puts the fact at the scope where it is true. Cost, stated: the funnel's `unrunnable` head count
gains a **second source** (the conversation marker) beside `turnDisposition ∈ {no-response,
server-rejected}` — an S5 addition, and S5 is unbuilt, so it costs nothing already shipped.

**(d) The fail-closed teeth the abort side is right to want.** The record is written, then `run`
**exits `4`** — the existing *invalid pack* code, which is exactly what a raising `dispatch`
proves — printing a `PACK DISPATCH FAILURES` block at the funnel head with
`(scriptId, turn, tool, reason)`. `compare` prints the per-arm count and otherwise behaves
normally, on the `INVALID RESULTS EXCLUDED` precedent that a data-quality finding is a report, not
an operational failure. **No new verdict gate**: `observable_floor` on the surviving `n` already
refuses a collapsed denominator (§3.4 Rule 7). The net property is that the ruling **degrades
continuously with the defect's size** — one raise costs one conversation and says so, a
systematically broken pack costs all twelve and `verdict()` refuses — where both filed options are
step functions.

**Rejected alternatives.** *Abort the run* — loses 12× the data for the same defect, produces no
comparison rather than a reduced one, and its only advantage (loudness) is bought by (d) for free.
*Record as `undispatchable` and continue the turn* — finding 1 (silent, signed, model-correlated
mis-scoring) plus finding 2 (drives on against an environment whose state is unknown). *End the
turn but continue the conversation* — finding 2 again, one turn later; and it silently keeps
contaminated turns inside (a)–(g) via rule 5's carve-out.

## 5. Evaluation design

| # | What it proves | Data | Threshold |
|---|---|---|---|
| **E1** | The sim contract holds — `dispatch` is total | Property test over each tool in `schemas()` × a fixed adversarial-but-JSON-object argument set (absent key, extra key, wrong type per declared field, `""`, `0`, `-1`, a large int, unicode, nested object, `{}`) | **Zero raises**, one `DispatchRecord` per call. A hard gate, correctly: this is a totality claim, not a rate, so one raise falsifies it and small `n` is not a limitation |
| **E2** | The censoring is wired, and is not the headline's rule | Synthetic: 9-turn script, sim raises at `t = 4`, beside three clean conversations, `H = 4` | `drive` raises the named class carrying **3** `TurnTrace`s; conversation stored censored at 4; funnel prints 1; headline denominator **3** with 1 in `n/a`; hazard risk set 3 at `t ≥ 4` with `c_4 == 1`; (a)–(g) receive turns 1–3 and **nothing** after |
| **E3** | The two rules are not collapsed into one | The same fixture with the raise at `t = 5` | **In** `cleanThroughTurn4`'s denominator, **out** of the hazard from `t = 5` — the discriminating pair §4 S5 item (3a) already gates, extended to this cause |
| **E4** | Negative control — censoring ≠ scored failure | The E2 fixture against one where turn 4 is a normal scored failure | Headline, hazard and per-position figures **differ**. If they are equal, the censoring is not wired and E2 passed on a tautology |
| **E5** | Disclosure survives storage | A stored run with ≥1 dispatch failure, re-read by `compare` | Per-arm count printed; `run` exited `4`; the record is present and complete for turns `1 … t−1` |

## 6. What it costs, per document

- **`modelbench/convo.py` / `runner`** — the named exception gains a payload (partial trace, turn
  index, call) and `drive`'s docstring gains the conversation-granular ruling; runner catch, store,
  continue. Small, on top of the naming already in flight. **`TurnDisposition` and
  `TURN_DISPOSITIONS` are untouched**, and so is their three-way probe.
- **`docs/plans/small-model-benchmarking.md`** — the expensive item, and I am flagging it as such.
  Three edits: **(i)** §4 S2's replay clause *"(no name, or the dispatch raised)"* loses the four
  words — under this ruling a raised dispatch is never replayed, so the parenthetical stops being
  merely stale (Pass 17 Q1) and becomes wrong; **(ii)** §3.3's `tools/sim.py` bullet gains §4(a)'s
  totality rule; **(iii)** §3.8.4 gains the conversation-censoring ruling, §4 S5's *Done when*
  gains E1–E4 and the funnel's second `unrunnable` source, and the exit-code paragraph gains one
  clause allowing `4` after artifacts are written. All three are additions to an **unbuilt** stage
  or a four-word deletion; none reopens a ruling the gate closed. If reopening the plan is refused,
  the fallback that preserves the method is (ii) alone plus the code change, at the price of a
  contract owed by prose and gated by nothing — which is precisely what P14-3 and P15-1 cost.
- **`docs/plans/small-model-benchmarking-ml.md` (mine, v1.22 → v1.23)** — §4.1's `unrunnable`
  gains the tool channel; §4.3 rule 4 gains the row and the discriminator sentence; rule 5 gains
  the stated exception in §4(b) (state contamination reaches (a)–(g); history contamination does
  not); the funnel gains its line. `ITERATION_SUMMARY_DISPOSITIONS` / `_EXCLUDED` and their union
  assertion are **unchanged** — a consequence of taking no sixth member.
- **`model-bench/AGENTS.md` or the pack README** — §4(a)'s three bullets, verbatim, as the pack
  authoring rule. One place a sim author will actually read.

## 7. Risks and open questions

- **This ruling rests on one contract the plan implies and never states, and it should be gated
  with the rest.** §3.8.4 makes the sim *"deterministic and stateful within a conversation"*, and
  `drive` takes `env` as a parameter, so the **runner** must call `build_environment()` **once per
  conversation**. Nothing in §4 S2, §4 S5 or `tooling.py` says so. If one environment were reused
  across the twelve scripts, the blast radius of a raise is the run again — and, independently of
  this ruling, cart and order state would leak from script to script and falsify FR-10's ground
  truth on every conversation after the first. Cheap gate: a runner test asserting twelve distinct
  environment instances, and one asserting conversation `k+1` opens with an empty cart after
  conversation `k` placed an order.
- **The raise hazard is unmeasured.** Every figure in §3 is arithmetic on an assumed `p`. The first
  real `tool-caller` run measures it; if it is not ~0 after §4(a), the sim contract is not being
  followed and that is the finding, not the statistic.
- **Adaptive fixing is narrowed, not closed.** An author fixes a sim knowing which arm triggered
  it, and *what* the fixed tool returns is still a judgment. §4(a) narrows it to one mechanical
  answer, and finding 5's content hash forbids mixing pre- and post-fix data. Residual: the error
  *value* can affect FR-8(g)'s containment check. Mitigation to weigh at S5 — declare error returns
  in the pack rather than inventing them in `sim.py`.
- **At `n = 12` one censored conversation moves the headline 8.3 pp**, which §4.3 rule 4 already
  names as enough to flip a McNemar discordant pair. That is an argument for §4(a), not against
  §4(b): the ruling makes the loss visible and priced; it cannot make it cheap.
- **Filename, no action requested:** `-ml` is a closed role suffix and here it sits mid-basename.
  The path is the caller's and is used as given.
