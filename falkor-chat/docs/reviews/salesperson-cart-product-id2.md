# Review — cart writes by product ID, S7 fixes (filter-miss recovery, customer speaker label)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-078

## Pass 1

**Reviewed:** `falkor-chat/docs/plans/salesperson-cart-product-id2.md` (D8–D12, §4 budget, §5–§6),
the plan-gate for RCA fix items 2–3.

**Baseline:**
- the tree at `e4cdba9f` (the devops 2-worker cap is already in);
- the RCA `falkor-chat/docs/reviews/salesperson-cart-product-id-rca.md`;
- S7's raw traffic (session scratchpad `…/k078-s7/`, `…/rca/`).

**Verdict: approve with suggestions.** No blocker. One major (M1) needs a plan edit with the
stopping condition below. The rest are minor or nits.

CPG: considered, not relevant — the change touches five functions (`FilterProductsTool.run`,
`Repository.filter_products`, `Services.catalog_index`, `_assemble_messages`,
`_build_default_app`), each read directly; `cpg_falkorchat` is stale per the original plan.

### Answers to the five check points

**1. Budget: the note is right, and my RCA figure was wrong.**

I re-derived the worst 2-way demand two ways:

| Definition | S7 (`v9`) | Q3 (`v8`) |
|---|---|---|
| Proxy: any two time-overlapping successful calls from different threads, prompt + completion | **7,074** (C1 es + pt-BR, both turn 4, iteration 7) | — |
| Runtime log: any two concurrently active slots at a launch instant, final `n_tokens` | **7,072** | 5,602 |
| Hard bound: 2 × the largest single sequence (3,595) | 7,190 | — |
| RCA's "~6.4k": the worst pair *inside the 3-call failure groups* | 6.4k | — |

- The first two definitions agree to within 2 tokens.
- The RCA's 6.4k used the wrong population. I have corrected RCA §5.1/§5.2 in place (dated line).
- **Size of D8's result.** The architect's ≈ +430 is slightly low. Built from the real catalog and
  the D8 strings, the result is 1,394 chars against the lookup miss's 1,211 (397 tokens). That puts
  it at **≈ 445–457 tokens**, a net ≈ +440 over the 16-token empty it replaces.

**Does the 2-worker cap plus D8 clear 8,192?**

| Scenario | Demand | Margin |
|---|---|---|
| Worst observed pair + D8 on one sequence | ≈ 7,515 | ≈ 680 (8.3%) |
| The same, with D9 | ≈ 7,450 | ≈ 740 (9%) |
| The note's "expected" row (no cascade) | ≈ 6.1–6.2k | ≈ 24% |

- **Yes, it fits,** though below my own 10% review line on the worst case.
- Every pair above 7,000 is a turn-4 re-add (iteration 7) after a dead turn. Under the cap, a dead
  turn needs an exogenous cause: a third LM Studio caller, or a DEF-6-style cold start.
- A turn can't outlive its worker: running futures are never cancelled, and the 180 s agent timeout
  is ≫ the 10.8 s maximum observed call. So the cap really does bound in-flight sequences to 2.

**Is D11 still right? Yes.**
- I verified `_collect_target_values` (`executor.py:393`) collects only values under a `productId`
  or `*.productId` key. So `[id, name]` pairs would stop grounding, and compaction is an executor
  change.
- Its ~100-token saving doesn't change the margin class anyway.
- The 7,373 trigger is falsifiable and sits between the expected (≈ 6.2k) and the cascade
  (≈ 7.5k) cases, so it fires exactly when a cascade recurs. That is the right signal.

**2. D9's seam: confirmed.**
- **Single human per thread.** Role comes from the actor kind (`services.py:909`: `User` → `user`,
  else `assistant`). The only `User` who posts into a storefront thread is its participant:
  `POST /shop/api/messages` posts to `who.thread_id` (`storefront_api.py:1233`), and the presenter
  routes post nothing.
- **Other agent nodes in that process: none in practice.**
  - The trigger is pinned to `salesperson@v9`, and the responder fall-through is off.
  - `order-fulfillment@v1` has no agent step.
  - `sweep_due_workflow_runs` resumes only timer-declaring steps, and salesperson declares none.
  - `/mcp` and the legacy router are unmounted.
  - `query_graph_data`'s internal call takes no thread rows.
- **Two things outside D9's reach** (see nit n1):
  - `guards.py` `_recent_turns` still renders `displayName` for an LLM judge. That is unreachable
    today because the guard is `cmp`.
  - If an operator re-points `TRIGGER_DEF_KEY` in storefront mode, that def also gets `Customer:`.
    That is harmless, since its threads are single-human too.

**3. `has_category`: correct as specified.**
- `Repository.filter_products` (`repository.py:3558`) normalizes with
  `normalize_name(category) or None`. Passing the raw value through therefore gives
  `has_category("")` → whole catalog → `True`, which matches §5's sketch.
- A non-string `category` raises inside the *first* `filter_products` call, as it does today, so
  `has_category` adds no new failure.
- The case/whitespace behaviour inherits from
  `test_filter_products_category_is_case_and_whitespace_insensitive` (`test_repository.py:4067`,
  `wf_repo`; not run, per the constraint).
- **The hint against the frozen v9 prompt: consistent.** The prompt's add rule is "copied exactly
  from a `lookup_product_fact` or `filter_products` result", and the `catalog` sits inside a
  `filter_products` result. The prompt's pick instruction covers only a lookup miss, which is why
  the hint has to carry it. It contains no "again" and no price.

**4. Mutants: three are behaviour-equivalent, and are killed only by call-shape spies** (minor m3).
- **N1:** `min_price=0` matches every product (all prices > 0), and `limit` ≠ 1 gives the same
  `bool`.
- **N2, the normalize-in-Python variant:** `normalize_name` is idempotent
  (`strip` → whitespace-collapse → `casefold`), so normalizing twice is invisible.
- **N6:** `has_category(None)` → whole catalog → `True` → the old shape, plus one extra read.

The other nine (N3–N5, N7–N12) change observable output and are killed by the tests named. I
checked that the cited existing tests and harness exist: `test_workflow_wiring_is_gated_on_workflow_enabled`
(`test_app.py:238`), `StubServices` (`test_tools.py:58`),
`test_filter_products_abstains_when_nothing_matches` (`:553`), `StubThreadServices` and
`test_agent_node_folds_thread_messages_into_prompt` (`test_executor_agent.py:108`/`:365`).

**5. Interactions.** These are findings M1, m1 and m2 below. The probe's customer text is English in
every language. After D9, turn 0 ("Hi there!") carries no in-line non-English cue at all.

### Findings

**M1 — Major — D9 removes the only in-line language cue, and the re-run cannot attribute a
wrong-language rise to it** (§6.2, §8 R-B).
- **What changes.** Every S7, U3 and K-065 measurement of 0 wrong-language replies was taken with
  `…-pt-BR:`/`…-es:` on every customer line. The probe's text is always English
  (`language_salience_probe.py` `SCRIPTED_TURNS`). After D9, the user content reads
  `Customer: Hi there!`: an English label on English text, with `CONTEXT.language` the only cue.
- **What the evidence covers.** K-065's language fix (CONTEXT plus two prompt sentences) has never
  been measured without the suffix, so "no code reads it" doesn't bound the model's behaviour.
- **What is missing.** The plan keeps the ≤ 3/375 gate but has no early stop and no attribution
  rule. A fail would arrive after a ~45-min run, confounded with D8.
- **What makes attribution possible.** D8 cannot touch turns 0–1: S7 had 0 `filter_products`
  calls there.
- **Suggested change:** add to §6.2
  - (i) a **pre-check before C1**: the probe at `--trials 12`, all three languages, turns 0–1 only
    (or a full short run read at those turns);
  - (ii) wrong-language reported **by turn index × language**, with a stated rule: a rise confined
    to turns 0–1 is D9's.
- **Stopping condition:** any pt-BR or es wrong-language reply in the pre-check stops the run
  before C1 and goes back to `architect`. **Done** when §6.2 contains both items and the rule.

**m1 — Minor — D8's grounding widening is said to be measured by "C2 pt-BR", but C2 never takes
the filter path** (§8 R-F, §6.2).
- **Evidence:** S7's C2 and confirm runs contain **0** `filter_products` calls in any language. All
  37 product-name filters were C1 (36 pt-BR, 1 en), at turn indices 2–3, after the
  `query_graph_data` turn.
- **Consequence:** C2 cannot see a re-add grounded by a category-miss list.
- **Suggested change:** replace "C2 pt-BR" with C1 M2/M3 plus a new reported count: C1 turns where
  a category-miss `catalog` was followed by an add of a product the turn's customer line didn't
  name.

**m2 — Minor — D8 lengthens pt-BR add turns, and the latency prior ignores it under the 2-worker
queue** (§6.2 priors).
- **Evidence:** today all 35 DEF-2 turns are one LLM call (filter → text). D8 makes them ≈ 3 (filter
  → add or lookup → add → post). Combined with 2 workers, the trial's third participant waits
  longer.
- **Suggested change:** name pt-BR alongside en as a latency risk in the priors, and report queue
  wait per language (measure 2 already reports queue wait).

**m3 — Minor — N1, N2 (normalize variant) and N6 are behaviour-equivalent** (§6.1 mutant table).
- **Why it matters:** they are killed by tests 28/29/33 asserting exact repository call arguments.
  That is legitimate as a contract pin (one read, raw pass-through, no read on price-only misses),
  but the table presents them as behaviour kills.
- **Suggested change:** mark the three "contract pin (behaviour-equivalent)" in the table, so a
  later refactor that legitimately changes the call shape doesn't read as a regression.

**m4 — Minor — the reversal trigger for rejecting a `categories` list can never fire in the planned
re-run** (D8, "Reversal trigger: translated category-browse questions ending without an answer").
- **Why:** neither probe asks a category-browse question.
- **What the branch now does to one:** a real "show me Periféricos" will get every product, labelled
  in the hint "every product in the catalog instead", with no categories. The model may then present
  all 15 as the category's contents.
- **Suggested change:** either add one scripted browse turn to the confirm variant
  (e.g. "Do you have any periféricos?", 12 × pt-BR/es), or record the trigger as unmeasured in
  BACKLOG with D11.

**n1 — Nit.** D9 leaves `guards.py` `_recent_turns` naming speakers by `displayName`.
Unreachable now (`cmp` guard); one docstring line at D9 saying so avoids a future LLM-guard storefront
def rediscovering it.

**n2 — Nit.** §4's D8 row: use ≈ +445–457 (built from the real strings, §1 above), not ≈ +430.
The margins shift by ~25 tokens, and no conclusion changes.

**n3 — Nit.** Test 40 reads `services._executor`. The existing `fake_create_app(services=None,
**kwargs)` captures only `kwargs`, so the test must also capture `services`. That is a one-line
detail for the implementer.

### What's solid

- D8.1 decides "unknown category" in the repository with no second normalizer. That is exactly
  K-058's rule, and the blank-category edge falls out correctly.
- D8.2's own hint (no false "again") and the dropped "no matching products found" finding both
  target R2's observed shape.
- D9's constructor seam avoids the create-only `config` trap and leaves every non-storefront
  deployment byte-identical (test 37 pins a literal).
- D11's added reason (compaction breaks grounding) is correct and was missed by the RCA.
- §6.2's instrument note on the RCA's `LSP-` parser is right.
- Re-deriving my budget rather than inheriting it caught a real error.

### Open questions

None for the stakeholder. M1's pre-check costs about 5 minutes of LM Studio time, which `teco`
schedules.
